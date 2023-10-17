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
        model,
        learning_rate = False, #auto_lr
        ckpt = None
    ) -> None:
        """
        SAM predicts object masks from an image and input prompts.
        """
        super().__init__()
        self.ckpt = ckpt
        self.model = model
        if ckpt is not None:
            self.model.propagation_module.load_state_dict(ckpt['model_state_dict'])
            print("Loaded checkpoint for propagation module")

        for param in self.model.parameters():
            param.requires_grad = False

        for p in self.model.propagation_module.parameters():
            p.requires_grad = True
    
        self.cfg = config
        self.epoch_freq = self.cfg.metric_train_eval_interval
        
        self.train_jaccard_sep_0 = torchmetrics.JaccardIndex(task="binary")
        self.val_jaccard_sep_0 = torchmetrics.JaccardIndex(task="binary")
        self.train_dice_sep_0 = torchmetrics.Dice()
        self.val_dice_sep_0 = torchmetrics.Dice()

        self.train_jaccard_sep_1 = torchmetrics.JaccardIndex(task="binary")
        self.val_jaccard_sep_1 = torchmetrics.JaccardIndex(task="binary")
        self.train_dice_sep_1 = torchmetrics.Dice()
        self.val_dice_sep_1 = torchmetrics.Dice()
        
        self.train_benchmark = []
        self.val_benchmark = []
        
        self.learning_rate = learning_rate #auto_lr_find = True

    def forward(
        self,
        batched_input: List[Dict[str, Any]],
        multimask_output: bool,
    ) -> List[Dict[str, torch.Tensor]]:
        
        return self.model(batched_input, multimask_output)
    
    def check_frequency(self, check_idx):
        return check_idx % self.epoch_freq == 0

    @torch.no_grad()
    def log_images_0(self, names, img_list, mask_list, gt_list, batch_idx, train=True):
        for name, img, gt, pred in zip(names, img_list, gt_list, mask_list):
            mask_dict = {}
            mask_dict["ground_truth"] = {"mask_data" : gt.cpu().numpy()}
            for obj_index, (pred_obj) in enumerate(pred):
                mask_dict[f"prediction_{obj_index + 1}"] = {"mask_data" : pred_obj.cpu().numpy() * (obj_index + 1)}
            self.logger.log_image(f"Images/{'train' if train else 'val'}/{name}", [img], step=self.global_step, masks=[mask_dict], caption=[f"Epoch_{self.current_epoch}"])

    @torch.no_grad()
    def log_images_1(self, names, img_list_0, img_list_1, mask_list_0, gt_list_0, mask_list_1, gt_list_1, batch_idx, train=True):
        for name, img_0, img_1, gt_0, gt_1, pred_0, pred_1 in zip(names, img_list_0, img_list_1, gt_list_0, gt_list_1, mask_list_0, mask_list_1):
            mask_dict_0 = {}
            mask_dict_0["ground_truth"] = {"mask_data" : gt_0.cpu().numpy()}
            mask_dict_1 = {}
            mask_dict_1["ground_truth"] = {"mask_data" : gt_1.cpu().numpy()}
            for obj_index, (pred_obj_0, pred_obj_1) in enumerate(zip(pred_0, pred_1)):
                mask_dict_0[f"prediction_{obj_index + 1}"] = {"mask_data" : pred_obj_0.cpu().numpy() * (obj_index + 1)}
                mask_dict_1[f"prediction_{obj_index + 1}"] = {"mask_data" : pred_obj_1.cpu().numpy() * (obj_index + 1)}
            
            self.logger.log_image(f"Images/{'train' if train else 'val'}/{name}", [img_0, img_1], step=self.global_step, masks=[mask_dict_0, mask_dict_1], caption=[f"Epoch_{self.current_epoch}_Frame_0", f"Epoch_{self.current_epoch}_Frame_1"])
   
    def sigmoid_focal_loss(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        p: torch.Tensor,
        alpha: float = 0.25, # optimal based on https://arxiv.org/pdf/1708.02002.pdf
        gamma: float = 2,
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
        inputs = inputs.flatten(1)	 # [num_true_obj, HxW]
        targets = targets.flatten(1) # [num_true_obj, HxW]
        p = p.flatten(1) # [num_true_obj, HxW]
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** gamma) # [1, H, W]
        if alpha >= 0:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            loss = alpha_t * loss

        return loss.mean(dim=1).sum()

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
        inputs = inputs.flatten(1)	 # [num_true_obj, HxW]
        targets = targets.flatten(1) # [num_true_obj, HxW]
        numerator = 2 * (inputs * targets).sum(-1)
        denominator = inputs.sum(-1) + targets.sum(-1)
        loss = 1 - (numerator + 1) / (denominator + 1)
        return loss.sum()
    
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
        inputs = inputs.flatten(1)	 # [num_true_obj, HxW]
        targets = targets.flatten(1) # [num_true_obj, HxW]
        numerator = (inputs * targets).sum(-1)
        denominator = inputs.sum(-1) + targets.sum(-1) - numerator
        iou = (numerator + 1) / (denominator + 1)
        return iou

    def configure_optimizers(self) -> Any:
        print("Using Learning Rate: ", self.learning_rate if self.learning_rate else self.cfg.opt.learning_rate, f"instead of {self.cfg.opt.learning_rate}")
        optimizer = torch.optim.AdamW(
            self.model.propagation_module.parameters(),
            lr= self.learning_rate if self.learning_rate else self.cfg.opt.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=self.cfg.opt.weight_decay,
            amsgrad=False,
        )
        if self.ckpt:
            optimizer.load_state_dict(self.ckpt['optimizer_state_dict'])
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.cfg.opt.steps, gamma=self.cfg.opt.decay_factor)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        output_0, output_1, prop_pos_embed = self(batch, False)
        bs = len(output_0)
        loss_focal = 0
        loss_dice = 0
        loss_iou = 0

        # output: 
        # "masks": masks, # (P=3, H, W)
        # "iou_predictions": iou_predictions, # (P=3, 1)
        # "low_res_logits": low_res_masks, # (P=3, 1, 256, 256)
        
        # batch:
        # "gt_mask": gt_mask, # (B, P, H, W) P = 3
        # "selector": selector, # (B, P) P = 3
        # "cropped_img": cropped_img, # (B, 3, H, W)

        pred_masks_list_0 = []
        pred_masks_list_1 = []
        total_num_objects = 0

        for each_output, gt_mask, selector in zip(output_0, batch['gt_mask'], batch['selector']): # selector = [True, True, False]
            total_num_objects += selector.sum()
            pred_masks = each_output["masks"] # [P=3, H, W]
            # pred_masks_list.append(pred_masks.detach())
            pred_masks = pred_masks[selector] # [num_true_obj, H, W]
            gt_mask = gt_mask[0][selector] # [num_true_obj, H, W] # gt_mask[0] to get the 0th mask from the prediction
                        
            pred_masks_sigmoid = torch.sigmoid(pred_masks)
            pred_masks_list_0.append(pred_masks_sigmoid.detach())
            
            loss_focal += self.sigmoid_focal_loss(pred_masks, gt_mask, pred_masks_sigmoid)
            loss_dice += self.dice_loss(pred_masks_sigmoid, gt_mask)
            loss_iou += F.mse_loss(
                each_output["iou_predictions"][selector].reshape(-1), self.iou(pred_masks_sigmoid, gt_mask), reduction="sum"
            )

        if self.cfg.dataset.stage1:
            for each_output, gt_mask, selector in zip(output_1, batch['gt_mask'], batch['selector']): # selector = [True, True, False]
                total_num_objects += selector.sum()
                pred_masks = each_output["masks"] # [P=3, H, W]
                # pred_masks_list.append(pred_masks.detach())
                pred_masks = pred_masks[selector] # [num_true_obj, H, W]
                gt_mask = gt_mask[1][selector] # [num_true_obj, H, W] # gt_mask[1] to get the 1st mask from the prediction
                            
                pred_masks_sigmoid = torch.sigmoid(pred_masks)
                pred_masks_list_1.append(pred_masks_sigmoid.detach())
                
                loss_focal += self.sigmoid_focal_loss(pred_masks, gt_mask, pred_masks_sigmoid)
                loss_dice += self.dice_loss(pred_masks_sigmoid, gt_mask)
                loss_iou += F.mse_loss(
                    each_output["iou_predictions"][selector].reshape(-1), self.iou(pred_masks_sigmoid, gt_mask), reduction="sum"
                )

        # Ex: focal - tensor(0.5012, device='cuda:0') dice - tensor(1.9991, device='cuda:0') iou - tensor(1.7245e-05, device='cuda:0')
        loss_total = (self.cfg.focal_wt * loss_focal + loss_dice + loss_iou) / (bs * total_num_objects)
        avg_focal = (self.cfg.focal_wt * loss_focal) / (bs * total_num_objects)
        avg_dice = loss_dice / (bs * total_num_objects)
        avg_iou = loss_iou / (bs * total_num_objects)
        
        self.train_benchmark.append(loss_total.item())
        pos_embed_image_wt = prop_pos_embed
        log_dict = {
            "Loss/train/total_loss" : loss_total, "Loss/train/focal_loss" : avg_focal, "Loss/train/dice_loss" : avg_dice, "Loss/train/iou_loss" : avg_iou,
            "pos_embed_image_wt/min": pos_embed_image_wt.min(), "pos_embed_image_wt/max": pos_embed_image_wt.max(), "pos_embed_image_wt/mean": pos_embed_image_wt.mean(), "pos_embed_image_wt/std": pos_embed_image_wt.std(),
        }
        self.log_dict(log_dict, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        return {'loss': loss_total, 'masks_0': pred_masks_list_0, 'masks_1': pred_masks_list_1} # List([num_true_obj, H, W])
    
    def validation_step(self, batch, batch_idx):
        output_0, output_1, _ = self(batch, False)
        bs = len(output_0)
        loss_focal = 0
        loss_dice = 0
        loss_iou = 0

        # output: 
        # "masks": masks, # (P=3, H, W)
        # "iou_predictions": iou_predictions, # (P=3, 1)
        # "low_res_logits": low_res_masks, # (P=3, 1, 256, 256)
        
        # batch:
        # "gt_mask": gt_mask, # (B, P, H, W) P = 3
        # "selector": selector, # (B, P) P = 3
        # "cropped_img": cropped_img, # (B, 3, H, W)

        pred_masks_list_0 = []
        pred_masks_list_1 = []
        total_num_objects = 0

        for each_output, gt_mask, selector in zip(output_0, batch['gt_mask'], batch['selector']): # selector = [True, True, False]
            total_num_objects += selector.sum()
            pred_masks = each_output["masks"] # [P=3, H, W]
            # pred_masks_list.append(pred_masks.detach())
            pred_masks = pred_masks[selector] # [num_true_obj, H, W]
            gt_mask = gt_mask[0][selector] # [num_true_obj, H, W] # gt_mask[0] to get the 0th mask from the prediction
                        
            pred_masks_sigmoid = torch.sigmoid(pred_masks)
            pred_masks_list_0.append(pred_masks_sigmoid.detach())
            
            loss_focal += self.sigmoid_focal_loss(pred_masks, gt_mask, pred_masks_sigmoid)
            loss_dice += self.dice_loss(pred_masks_sigmoid, gt_mask)
            loss_iou += F.mse_loss(
                each_output["iou_predictions"][selector].reshape(-1), self.iou(pred_masks_sigmoid, gt_mask), reduction="sum"
            )

        if self.cfg.dataset.stage1:
            for each_output, gt_mask, selector in zip(output_1, batch['gt_mask'], batch['selector']): # selector = [True, True, False]
                total_num_objects += selector.sum()
                pred_masks = each_output["masks"] # [P=3, H, W]
                # pred_masks_list.append(pred_masks.detach())
                pred_masks = pred_masks[selector] #[num_true_obj, H, W]
                gt_mask = gt_mask[1][selector] #[num_true_obj, H, W] # gt_mask[1] to get the 1st mask from the prediction
                            
                pred_masks_sigmoid = torch.sigmoid(pred_masks)
                pred_masks_list_1.append(pred_masks_sigmoid.detach())
                
                loss_focal += self.sigmoid_focal_loss(pred_masks, gt_mask, pred_masks_sigmoid)
                loss_dice += self.dice_loss(pred_masks_sigmoid, gt_mask)
                loss_iou += F.mse_loss(
                    each_output["iou_predictions"][selector].reshape(-1), self.iou(pred_masks_sigmoid, gt_mask), reduction="sum"
                )

        # Ex: focal - tensor(0.5012, device='cuda:0') dice - tensor(1.9991, device='cuda:0') iou - tensor(1.7245e-05, device='cuda:0')
        loss_total = (self.cfg.focal_wt * loss_focal + loss_dice + loss_iou) / (bs * total_num_objects)
        avg_focal = (self.cfg.focal_wt * loss_focal) / (bs * total_num_objects)
        avg_dice = loss_dice / (bs * total_num_objects)
        avg_iou = loss_iou / (bs * total_num_objects)

        self.val_benchmark.append(loss_total.item())
        self.log_dict({"Loss/val/total_loss" : loss_total, "Loss/val/focal_loss" : avg_focal, "Loss/val/dice_loss" : avg_dice, "Loss/val/iou_loss" : avg_iou}, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        return {'loss': loss_total, 'masks_0': pred_masks_list_0, 'masks_1': pred_masks_list_1} # List([num_true_obj, H, W])
    
    def on_train_batch_end(self, output, batch, batch_idx):
        if self.check_frequency(self.current_epoch):
            img_list_0 = []
            img_list_1 = []
            sep_mask_list_0 = []
            sep_mask_list_1 = []
            gt_mask_list_0 = []
            gt_mask_list_1 = []
            sep_gt_mask_list_0 = []
            sep_gt_mask_list_1 = []

            for each_output, gt_mask, cropped_img, selector in zip(output["masks_0"], batch['gt_mask'], batch['cropped_img'][0], batch['selector']):
                gt_mask = gt_mask[0][selector]
                sep_mask_list_0.append(each_output>0.5)
                sep_gt_mask_list_0.append(gt_mask.type(torch.int8))

                max_, max_pos = torch.max(gt_mask, dim=0)
                gt_mask = ((max_pos+1) * (max_)).type(torch.int8)
                gt_mask_list_0.append(gt_mask)
                
                img_list_0.append(cropped_img)

            if self.cfg.dataset.stage1:
                for each_output, gt_mask, cropped_img, selector in zip(output["masks_1"], batch['gt_mask'], batch['cropped_img'][1], batch['selector']):
                    gt_mask = gt_mask[1][selector]
                    sep_mask_list_1.append(each_output>0.5) # (num_true_obj, H, W)
                    sep_gt_mask_list_1.append(gt_mask.type(torch.int8)) # (num_true_obj, H, W)

                    max_, max_pos = torch.max(gt_mask, dim=0)
                    gt_mask = ((max_pos+1) * (max_)).type(torch.int8) # (H, W)
                    gt_mask_list_1.append(gt_mask)
                    
                    img_list_1.append(cropped_img)

            metrics_all = {'Metrics/train/jaccard_single_obj_0': self.train_jaccard_sep_0, 'Metrics/train/dice_single_obj_0': self.train_dice_sep_0}
            sep_mask_tensor_0 = torch.cat(sep_mask_list_0)
            sep_gt_mask_tensor_0 = torch.cat(sep_gt_mask_list_0)

            if self.cfg.dataset.stage1:
                sep_mask_tensor_1 = torch.cat(sep_mask_list_1)
                sep_gt_mask_tensor_1 = torch.cat(sep_gt_mask_list_1)
                self.train_jaccard_sep_1(sep_mask_tensor_1, sep_gt_mask_tensor_1)
                self.train_dice_sep_1(sep_mask_tensor_1, sep_gt_mask_tensor_1)
                metrics_all.update({'Metrics/train/jaccard_single_obj_1': self.train_jaccard_sep_1, 'Metrics/train/dice_single_obj_1': self.train_dice_sep_1})

            self.train_jaccard_sep_0(sep_mask_tensor_0, sep_gt_mask_tensor_0)
            self.train_dice_sep_0(sep_mask_tensor_0, sep_gt_mask_tensor_0)
            self.log_dict(metrics_all, on_step=True, on_epoch=True, sync_dist=True)
            
            if batch_idx < 5:
                if self.cfg.dataset.stage1:
                    self.log_images_1(batch['name'], img_list_0, img_list_1, sep_mask_list_0, gt_mask_list_0, sep_mask_list_1, gt_mask_list_1, batch_idx=batch_idx, train=True)
                else:
                    self.log_images_0(batch['name'], img_list_0, sep_mask_list_0, gt_mask_list_0, batch_idx=batch_idx, train=True)
    
    def on_validation_batch_end(self, output, batch, batch_idx):
        img_list_0 = []
        img_list_1 = []
        sep_mask_list_0 = []
        sep_mask_list_1 = []
        gt_mask_list_0 = []
        gt_mask_list_1 = []
        sep_gt_mask_list_0 = []
        sep_gt_mask_list_1 = []

        for each_output, gt_mask, cropped_img, selector in zip(output["masks_0"], batch['gt_mask'], batch['cropped_img'][0], batch['selector']):
            gt_mask = gt_mask[0][selector]
            sep_mask_list_0.append(each_output>0.5)
            sep_gt_mask_list_0.append(gt_mask.type(torch.int8))

            max_, max_pos = torch.max(gt_mask, dim=0)
            gt_mask = ((max_pos+1) * (max_)).type(torch.int8)
            gt_mask_list_0.append(gt_mask)
            
            img_list_0.append(cropped_img)

        if self.cfg.dataset.stage1:
            for each_output, gt_mask, cropped_img, selector in zip(output["masks_1"], batch['gt_mask'], batch['cropped_img'][1], batch['selector']):
                gt_mask = gt_mask[1][selector]
                sep_mask_list_1.append(each_output>0.5) # (num_true_obj, H, W)
                sep_gt_mask_list_1.append(gt_mask.type(torch.int8)) # (num_true_obj, H, W)

                max_, max_pos = torch.max(gt_mask, dim=0)
                gt_mask = ((max_pos+1) * (max_)).type(torch.int8) # (H, W)
                gt_mask_list_1.append(gt_mask)
                
                img_list_1.append(cropped_img)

        metrics_all = {'Metrics/val/jaccard_single_obj_0': self.val_jaccard_sep_0, 'Metrics/val/dice_single_obj_0': self.val_dice_sep_0}
        sep_mask_tensor_0 = torch.cat(sep_mask_list_0)
        sep_gt_mask_tensor_0 = torch.cat(sep_gt_mask_list_0)

        if self.cfg.dataset.stage1:
            sep_mask_tensor_1 = torch.cat(sep_mask_list_1)
            sep_gt_mask_tensor_1 = torch.cat(sep_gt_mask_list_1)
            self.val_jaccard_sep_1(sep_mask_tensor_1, sep_gt_mask_tensor_1)
            self.val_dice_sep_1(sep_mask_tensor_1, sep_gt_mask_tensor_1)
            metrics_all.update({'Metrics/val/jaccard_single_obj_1': self.val_jaccard_sep_1, 'Metrics/val/dice_single_obj_1': self.val_dice_sep_1})

        self.val_jaccard_sep_0(sep_mask_tensor_0, sep_gt_mask_tensor_0)
        self.val_dice_sep_0(sep_mask_tensor_0, sep_gt_mask_tensor_0)
        self.log_dict(metrics_all, on_step=True, on_epoch=True, sync_dist=True)

        if self.cfg.dataset.stage1:
            self.log_images_1(batch['name'], img_list_0, img_list_1, sep_mask_list_0, gt_mask_list_0, sep_mask_list_1, gt_mask_list_1, batch_idx=batch_idx, train=False)
        else:
            self.log_images_0(batch['name'], img_list_0, sep_mask_list_0, gt_mask_list_0, batch_idx=batch_idx, train=False)