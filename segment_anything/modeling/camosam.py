# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
import numpy as np
from metrics import batched_jaccard, f_measure
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch
from torch import nn
from torch.nn import functional as F

from typing import Any, Dict, List, Optional, Tuple

import lightning as L
import torchmetrics
import wandb
import gc

from torchvision.transforms.functional import resize, to_pil_image  # type: ignore

class CamoSam(L.LightningModule):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        config,
        model
    ) -> None:
        """
        SAM predicts object masks from an image and input prompts.
        """
        super().__init__()
        self.model = model
        for param in self.model.parameters():
            param.requires_grad = False
        
        for param in self.model.propagation_module.parameters():
            param.requires_grad = True
    
        self.cfg = config
        self.batch_freq = self.cfg.metric_train_eval_interval

        self.train_jaccard = torchmetrics.JaccardIndex(task="multiclass", num_classes=self.cfg.dataset.max_num_obj)
        self.val_jaccard = torchmetrics.JaccardIndex(task="multiclass", num_classes=self.cfg.dataset.max_num_obj)
        self.train_dice = torchmetrics.Dice(num_classes=self.cfg.dataset.max_num_obj)
        self.val_dice = torchmetrics.Dice(num_classes=self.cfg.dataset.max_num_obj)

    def forward(
        self,
        batched_input: List[Dict[str, Any]],
        multimask_output: bool,
    ) -> List[Dict[str, torch.Tensor]]:
        
        return self.model(batched_input, multimask_output)
    
    def check_frequency(self, check_idx):
        return check_idx % self.batch_freq == 0
    
    @torch.no_grad()
    def log_images(self,img_list, mask_list, gt_list, batch_idx, train=True):
        table = wandb.Table(columns=['ID', 'Image'])

        for id, (img, gt, pred) in enumerate(zip(img_list, gt_list, mask_list)):
            mask_img = wandb.Image(img, masks = {
                "prediction" : { "mask_data" : pred,}, "ground truth" : {"mask_data" : gt}
            })
            
            table.add_data(id, mask_img)
        if train:
            wandb.log({f"Images/Epoch:{self.current_epoch}/{batch_idx}_Train" : table})
        else:
            wandb.log({f"Images/Epoch:{self.current_epoch}/{batch_idx}_Validation" : table})

    def sigmoid_focal_loss(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        p: torch.Tensor,
        alpha: float = 0.25, # optimal based on https://arxiv.org/pdf/1708.02002.pdf
        gamma: float = 2,
        reduction: str = "none",
    ) -> torch.Tensor:
        """
        Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

        Args:
            inputs (Tensor): A float tensor of arbitrary shape.
                    The predictions for each example.
            targets (Tensor): A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
            alpha (float): Weighting factor in range (0,1) to balance
                    positive vs negative examples or -1 for ignore. Default: ``0.25``.
            gamma (float): Exponent of the modulating factor (1 - p_t) to
                    balance easy vs hard examples. Default: ``2``.
            reduction (string): ``'none'`` | ``'mean'`` | ``'sum'``
                    ``'none'``: No reduction will be applied to the output.
                    ``'mean'``: The output will be averaged.
                    ``'sum'``: The output will be summed. Default: ``'none'``.
        Returns:
            Loss tensor with the reduction option applied.
        """
        # Original implementation from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py

        # p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** gamma) # [1, H, W]
        if alpha >= 0:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            loss = alpha_t * loss

        # Check reduction option and return loss accordingly
        if reduction == "none":
            pass
        elif reduction == "mean":
            loss = loss.mean()
        elif reduction == "sum":
            loss = loss.sum()
        else:
            raise ValueError(
                f"Invalid Value for arg 'reduction': '{reduction} \n Supported reduction modes: 'none', 'mean', 'sum'"
            )
        return loss

    def dice_loss(self, inputs, targets):
        """
        Compute the DICE loss, similar to generalized IOU for masks
        Args:
            inputs: A float tensor of arbitrary shape.
                    The predictions for each example.
            targets: A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
        """
        # 2ypˆ+ 1 /(y + ˆp + 1)
  
        # inputs = inputs.sigmoid()
        inputs = inputs.flatten(1)	 # [1, HxW]
        targets = targets.flatten(1) # [1, HxW]
        numerator = (inputs * targets).sum(-1, keepdim=True)
        denominator = inputs.sum(-1, keepdim=True) + targets.sum(-1, keepdim=True) - numerator
        loss = 1 - (numerator + 1) / (denominator + 1)
        return loss.mean()
    
    def iou(self, inputs, targets):
        """
        Compute the IOU
        Args:
            inputs: A float tensor of arbitrary shape.
                    The predictions for each example.
            targets: A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
        """
        # ypˆ+ 1 /(y + ˆp - ypˆ+ 1)
  
        # inputs = inputs.sigmoid()
        inputs = inputs.flatten(1)	 # [1, HxW]
        targets = targets.flatten(1) # [1, HxW]
        numerator = (inputs * targets).sum(-1, keepdim=True)
        denominator = inputs.sum(-1, keepdim=True) + targets.sum(-1, keepdim=True) - numerator
        iou = (numerator + 1) / (denominator + 1)
        return iou.mean(dim=1)

    def lr_lambda(self, step):
        if step < self.cfg.opt.warmup_steps:
            return step / self.cfg.opt.warmup_steps
        elif step < self.cfg.opt.steps[0]:
            return 1.0
        elif step < self.cfg.opt.steps[1]:
            return 1 / self.cfg.opt.decay_factor
        else:
            return 1 / (self.cfg.opt.decay_factor**2)

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.opt.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=self.cfg.opt.weight_decay,
            amsgrad=False,
        )
        # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, self.lr_lambda)
        return [optimizer] #, [scheduler]

    def training_step(self, batch, batch_idx):
        output = self(batch, False)
        bs = len(output)
        loss_focal = 0
        loss_dice = 0
        loss_iou = 0

        pred_masks_list = []
        for each_output, gt_mask, selector in zip(output, batch['gt_mask'], batch['selector']):
            pred_masks = each_output["masks"].squeeze()
            pred_masks_list.append(pred_masks.detach())
            pred_masks = pred_masks[selector] #[num_true_obj, H, W]
            gt_mask = gt_mask.squeeze()[selector] #[num_true_obj, H, W]

            pred_masks_sigmoid = torch.sigmoid(pred_masks)
            
            loss_focal += self.sigmoid_focal_loss(pred_masks, gt_mask, pred_masks_sigmoid, reduction="mean")
            loss_dice += self.dice_loss(pred_masks_sigmoid, gt_mask)
            loss_iou += F.mse_loss(
                each_output["iou_predictions"][selector].reshape(-1), self.iou(pred_masks_sigmoid, gt_mask), reduction="mean"
            )

        # Ex: focal - tensor(0.5012, device='cuda:0') dice - tensor(1.9991, device='cuda:0') iou - tensor(1.7245e-05, device='cuda:0')
        loss_total = (20.0 * loss_focal + loss_dice + loss_iou) / bs
        avg_focal = loss_focal / bs
        avg_dice = loss_dice / bs
        avg_iou = loss_iou / bs

        self.log_dict({"Loss/train_total_loss" : loss_total, "Loss/train_focal_loss" : avg_focal, "Loss/train_dice_loss" : avg_dice, "Loss/train_iou_loss" : avg_iou}, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        return {'loss': loss_total, 'masks': pred_masks_list}
    
    def validation_step(self, batch, batch_idx):
        output = self(batch, False)
        bs = len(output)
        loss_focal = 0
        loss_dice = 0
        loss_iou = 0

        pred_masks_list = []
        for each_output, gt_mask, selector in zip(output, batch['gt_mask'], batch['selector']):
            pred_masks = each_output["masks"].squeeze()
            pred_masks_list.append(pred_masks.detach())
            pred_masks = pred_masks[selector] #[num_true_obj, H, W]
            gt_mask = gt_mask.squeeze()[selector] #[num_true_obj, H, W]

            pred_masks_sigmoid = torch.sigmoid(pred_masks)
            
            loss_focal += self.sigmoid_focal_loss(pred_masks, gt_mask, pred_masks_sigmoid, reduction="mean")
            loss_dice += self.dice_loss(pred_masks_sigmoid, gt_mask)
            loss_iou += F.mse_loss(
                each_output["iou_predictions"][selector].reshape(-1), self.iou(pred_masks_sigmoid, gt_mask), reduction="mean"
            )

        # Ex: focal - tensor(0.5012, device='cuda:0') dice - tensor(1.9991, device='cuda:0') iou - tensor(1.7245e-05, device='cuda:0')
        loss_total = (20.0 * loss_focal + loss_dice + loss_iou) / bs
        avg_focal = loss_focal / bs
        avg_dice = loss_dice / bs
        avg_iou = loss_iou / bs

        self.log_dict({"Loss/val_total_loss" : loss_total, "Loss/val_focal_loss" : avg_focal, "Loss/val_dice_loss" : avg_dice, "Loss/val_iou_loss" : avg_iou}, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        return {'loss': loss_total, 'masks': pred_masks_list}
    
    def on_train_batch_end(self, output, batch, batch_idx):
        if self.check_frequency(batch_idx):
            img_list = []
            mask_list = []
            gt_mask_list = []

            for each_output, gt_mask, copped_img in zip(output["masks"], batch['gt_mask'], batch['cropped_img']):
                pred_masks = torch.argmax(each_output, dim=0).detach() #[H, W]

                gt_mask = torch.argmax(gt_mask, dim=0).squeeze().detach() #[H, W]
                
                mask_list.append(pred_masks)
                gt_mask_list.append(gt_mask)
                img = copped_img
                img_list.append(img)

            mask_list = torch.stack(mask_list)
            gt_mask_list = torch.stack(gt_mask_list)
            self.train_jaccard(mask_list, gt_mask_list)
            self.train_dice(mask_list, gt_mask_list)
            self.log('train_jaccard', self.train_jaccard, on_step=True, on_epoch=False)
            self.log('train_dice', self.train_dice, on_step=True, on_epoch=False)
            self.log_images(img_list, mask_list.cpu().numpy(), gt_mask_list.cpu().numpy(), batch_idx=batch_idx, train=True)

    
    def on_validation_batch_end(self, output, batch, batch_idx):
        img_list = []
        mask_list = []
        gt_mask_list = []

        for each_output, gt_mask, copped_img in zip(output["masks"], batch['gt_mask'], batch['cropped_img']):
            pred_masks = torch.argmax(each_output, dim=0).detach() #[H, W]

            gt_mask = torch.argmax(gt_mask, dim=0).squeeze().detach() #[H, W]
            
            mask_list.append(pred_masks)
            gt_mask_list.append(gt_mask)
            img = copped_img
            img_list.append(img)

        mask_list = torch.stack(mask_list)
        gt_mask_list = torch.stack(gt_mask_list)
        self.train_jaccard(mask_list, gt_mask_list)
        self.train_dice(mask_list, gt_mask_list)
        self.log('train_jaccard', self.train_jaccard, on_step=True, on_epoch=False)
        self.log('train_dice', self.train_dice, on_step=True, on_epoch=False)
        self.log_images(img_list, mask_list.cpu().numpy(), gt_mask_list.cpu().numpy(), batch_idx=batch_idx, train=True)