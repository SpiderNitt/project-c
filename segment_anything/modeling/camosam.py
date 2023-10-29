# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
import numpy as np
from metrics import jaccard_dice
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
        ckpt = None
    ) -> None:
        """
        SAM predicts object masks from an image and input prompts.
        """
        super().__init__()
        self.ckpt = ckpt
        self.model = model
        self.cfg = config
        if ckpt is not None:
            self.model.propagation_module.load_state_dict(ckpt['model_state_dict'])
            print("!!! Loaded checkpoint for propagation module !!!")
            if ckpt['decoder_state_dict']:
                self.model.mask_decoder.load_state_dict(ckpt['decoder_state_dict'])
                print("!!! Loaded checkpoint for mask decoder !!!")

        self.set_requires_grad()

        self.epoch_freq = self.cfg.train_metric_interval
        
        self.train_benchmark = []
        self.val_benchmark = []
        
    def set_requires_grad(self):
        model_grad = self.cfg.model.requires_grad

        for param in self.model.image_encoder.parameters():
            param.requires_grad = model_grad.image_encoder

        for param in self.model.prompt_encoder.parameters():
            param.requires_grad = model_grad.prompt_encoder

        for param in self.model.mask_decoder.parameters():
            param.requires_grad = model_grad.mask_decoder
        
        for param in self.model.propagation_module.parameters():
            param.requires_grad = model_grad.propagation_module

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr= self.cfg.opt.learning_rate if self.cfg.opt.learning_rate else 0,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=self.cfg.opt.weight_decay,
            amsgrad=False,
        )

        if self.ckpt and self.cfg.opt.learning_rate is None:
            try:
                optimizer.load_state_dict(self.ckpt['optimizer_state_dict']) # Try-except to handle the case when only propagation ckpt is to be loaded
            except:
                pass
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.cfg.opt.steps, gamma=self.cfg.opt.decay_factor)
        return [optimizer], [scheduler]

    def forward(
        self,
        batched_input: List[Dict[str, Any]],
        multimask_output: bool,
    ) -> List[Dict[str, torch.Tensor]]:
        
        return self.model(batched_input, multimask_output)
    
    def check_frequency(self, check_idx):
        return check_idx % self.epoch_freq == 0

    @torch.no_grad()
    def log_images_0(self, information, img_list_0, sep_mask_list_0, mask_list_0, gt_mask_list_0, iou_list_0, batch_idx, train=True):
        for info, img, sep_mask, mask, gt, iou in zip(information, img_list_0, sep_mask_list_0, mask_list_0, gt_mask_list_0, iou_list_0):
            mask_dict = {}
            mask_dict["ground_truth"] = {"mask_data" : gt.cpu().numpy()}
            mask_dict["prediction"] = {"mask_data" : mask.cpu().numpy()}
            for obj_index, (pred_obj, iou_obj) in enumerate(zip(sep_mask, iou)):
                mask_dict[f"prediction_{obj_index + 1}"] = {"mask_data" : pred_obj.cpu().numpy() * (obj_index + 1)}
            self.logger.log_image(f"Images/{'train' if train else 'val'}/{info['name']}", [img], step=self.global_step, masks=[mask_dict], 
                                  caption=[f"Epoch_{self.current_epoch}_IoU_{iou_obj.cpu().item(): 0.3f}_Frames_{info['frames']}"])

    @torch.no_grad()
    def log_images_1(self, information, img_list_0, img_list_1, sep_mask_list_0, sep_mask_list_1, mask_list_0, mask_list_1, gt_mask_list_0, gt_mask_list_1, iou_list_0, iou_list_1, batch_idx, train=True):
        for info, img_0, img_1, gt_0, gt_1, sep_mask_0, sep_mask_1, mask_0, mask_1, iou_0, iou_1 in zip(information, img_list_0, img_list_1, gt_mask_list_0, gt_mask_list_1, sep_mask_list_0, sep_mask_list_1, mask_list_0, mask_list_1, iou_list_0, iou_list_1):
            mask_dict_0 = {}
            mask_dict_0["ground_truth"] = {"mask_data" : gt_0.cpu().numpy()}
            mask_dict_0["prediction"] = {"mask_data" : mask_0.cpu().numpy()}
            mask_dict_1 = {}
            mask_dict_1["ground_truth"] = {"mask_data" : gt_1.cpu().numpy()}
            mask_dict_1["prediction"] = {"mask_data" : mask_1.cpu().numpy()}
            for obj_index, (sep_obj_0, sep_obj_1, iou_obj_0, iou_obj_1) in enumerate(zip(sep_mask_0, sep_mask_1, iou_0, iou_1)):
                mask_dict_0[f"prediction_{obj_index + 1}"] = {"mask_data" : sep_obj_0.cpu().numpy() * (obj_index + 1)}
                mask_dict_1[f"prediction_{obj_index + 1}"] = {"mask_data" : sep_obj_1.cpu().numpy() * (obj_index + 1)}
            
            self.logger.log_image(f"Images/{'train' if train else 'val'}/{info['name']}", [img_0, img_1], step=self.global_step, masks=[mask_dict_0, mask_dict_1],
                                  caption=[f"Epoch_{self.current_epoch}_IoU_{iou_obj_0.cpu().item(): 0.3f}_Frame_{info['frames'][0]}_{info['frames'][1]}", f"Epoch_{self.current_epoch}_IoU_{iou_obj_1.cpu().item(): 0.3f}_Frame_{info['frames'][1]}_{info['frames'][2]}"])
   
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
        inputs = inputs.flatten(-2)	 # [num_true_obj, C, HxW]
        targets = targets.flatten(-2) # [num_true_obj, C, HxW]
        p = p.flatten(-2) # [num_true_obj, C, HxW]
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** gamma) # [C, H, W]
        if alpha >= 0:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            loss = alpha_t * loss

        return loss.mean(dim=-1) # [num_true_obj, C]

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
        inputs = inputs.flatten(-2)	 # [num_true_obj, C, HxW]
        targets = targets.flatten(-2) # [num_true_obj, C, HxW]
        numerator = 2 * (inputs * targets).sum(-1)
        denominator = inputs.sum(-1) + targets.sum(-1)
        loss = 1 - (numerator + 1) / (denominator + 1)
        return loss # [num_true_obj, C]
    
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
        inputs = inputs.flatten(-2)	 # [num_true_obj, C, HxW]
        targets = targets.flatten(-2) # [num_true_obj, C, HxW]
        numerator = (inputs * targets).sum(-1)
        denominator = inputs.sum(-1) + targets.sum(-1) - numerator
        iou = (numerator + 1) / (denominator + 1)
        return iou # [num_true_obj, C]

    def training_step(self, batch, batch_idx):
        img_embeddings = self.model.getImageEmbeddings(batch['image']) # (B, F=3, 256, 64, 64)
        output_0, low_res_pred, prop_pos_embed, prop_dense_embed = self.model.getPropEmbeddings0(img_embeddings, batch, multimask_output=self.cfg.model.multimask_output)

        pred_masks_list_0 = []
        iou_pred_list_0 = []
        pred_masks_list_1 = []
        iou_pred_list_1 = []
        low_res_pred_list_0 = []
        total_num_objects = 0

        # output_0, output_1, prop_pos_embed = self(batch, self.cfg.model.multimask_output)
        bs = len(output_0)
        loss_focal = 0
        loss_dice = 0
        loss_iou = 0
        loss_total = 0

        for each_output, gt_mask, selector, low_pred in zip(output_0, batch['gt_mask'], batch['selector'], low_res_pred): # selector = [True, True, False]
            total_num_objects += selector.sum()
            pred_masks = each_output["masks"] # [P=3, C, H, W]
            # pred_masks_list.append(pred_masks.detach())
            gt_mask = gt_mask[0].unsqueeze(1) # [P, 1, H, W] # gt_mask[0] to get the 0th mask from the prediction
            gt_mask = gt_mask.repeat((1, pred_masks.shape[1], 1, 1)) # [P, C, H, W] 
            
            pred_masks_sigmoid = torch.sigmoid(pred_masks)
            loss_tmp = (
                self.cfg.focal_wt * self.sigmoid_focal_loss(pred_masks, gt_mask, pred_masks_sigmoid)
                + self.dice_loss(pred_masks_sigmoid, gt_mask)
                + F.mse_loss(
                    each_output["iou_predictions"],
                    self.iou(pred_masks_sigmoid, gt_mask),
                    reduction="none",
                )
            ) # [P, C]

            loss_tmp, min_idx = torch.min(loss_tmp, -1) # [P]
            loss_total += loss_tmp[selector].sum() # (num_true_obj)
            batch_indexing = torch.arange(len(min_idx), device=min_idx.device) # [P]
            loss_focal += self.sigmoid_focal_loss(pred_masks, gt_mask, pred_masks_sigmoid)[batch_indexing, min_idx][selector].sum()
            loss_dice += self.dice_loss(pred_masks_sigmoid, gt_mask)[batch_indexing, min_idx][selector].sum()
            loss_iou += F.mse_loss(
                each_output["iou_predictions"], self.iou(pred_masks_sigmoid, gt_mask), reduction="none"
            )[batch_indexing, min_idx][selector].sum() # [num_true_obj]

            low_res_pred_list_0.append(low_pred[batch_indexing, min_idx]) # (P=3, C, 256, 256) -> (P=3, 256, 256)
            pred_masks_list_0.append(pred_masks_sigmoid[batch_indexing, min_idx][selector].detach())
            iou_pred_list_0.append(each_output["iou_predictions"][batch_indexing, min_idx][selector].detach())
        
        low_res_pred_list_0 = torch.stack(low_res_pred_list_0).unsqueeze(1) # (B, 1, P=3, 256, 256)

        if self.cfg.dataset.stage1:
            for idx in range(2, self.cfg.dataset.num_frames):
                output_1, low_res_pred = self.model.getPropEmbeddings1(img_embeddings, batch, low_res_pred_list_0, self.cfg.model.multimask_output, idx)
                low_res_pred_list_1 = []

                for each_output, gt_mask, selector, low_pred in zip(output_1, batch['gt_mask'], batch['selector'], low_res_pred): # selector = [True, True, False]
                    total_num_objects += selector.sum()
                    pred_masks = each_output["masks"] # [P=3, C, H, W]
                    # pred_masks_list.append(pred_masks.detach())
                    gt_mask = gt_mask[idx-1].unsqueeze(1) # [P, 1, H, W] # gt_mask[1] to get the 1st mask from the prediction
                    gt_mask = gt_mask.repeat((1, pred_masks.shape[1], 1, 1)) # [P, C, H, W] 

                    pred_masks_sigmoid = torch.sigmoid(pred_masks)
                    loss_tmp = (
                        self.cfg.focal_wt * self.sigmoid_focal_loss(pred_masks, gt_mask, pred_masks_sigmoid)
                        + self.dice_loss(pred_masks_sigmoid, gt_mask)
                        + F.mse_loss(
                            each_output["iou_predictions"],
                            self.iou(pred_masks_sigmoid, gt_mask),
                            reduction="none",
                        )
                    ) # [P, C]

                    loss_tmp, min_idx = torch.min(loss_tmp, -1) # [P]
                    loss_total += loss_tmp[selector].sum() # (num_true_obj)
                    batch_indexing = torch.arange(len(min_idx), device=min_idx.device) # [P]
                    loss_focal += self.sigmoid_focal_loss(pred_masks, gt_mask, pred_masks_sigmoid)[batch_indexing, min_idx][selector].sum()
                    loss_dice += self.dice_loss(pred_masks_sigmoid, gt_mask)[batch_indexing, min_idx][selector].sum()
                    loss_iou += F.mse_loss(
                        each_output["iou_predictions"], self.iou(pred_masks_sigmoid, gt_mask), reduction="none"
                    )[batch_indexing, min_idx][selector].sum() # [num_true_obj]

                    low_res_pred_list_1.append(low_pred[batch_indexing, min_idx]) # (P=3, C, 256, 256) -> (P=3, 256, 256)
                    pred_masks_list_1.append(pred_masks_sigmoid[batch_indexing, min_idx][selector].detach())
                    iou_pred_list_1.append(each_output["iou_predictions"][batch_indexing, min_idx][selector].detach())
                low_res_pred_list_1 = torch.stack(low_res_pred_list_1).unsqueeze(1) # (B, 1, P=3, 256, 256)
                low_res_pred_list_0 = torch.cat([low_res_pred_list_0, low_res_pred_list_1], dim=1)
            
        # Ex: focal - tensor(0.5012, device='cuda:0') dice - tensor(1.9991, device='cuda:0') iou - tensor(1.7245e-05, device='cuda:0')
        loss_total = (loss_total) / (total_num_objects)
        avg_focal = (self.cfg.focal_wt * loss_focal) / (total_num_objects)
        avg_dice = loss_dice / (total_num_objects)
        avg_iou = loss_iou / (total_num_objects)
        
        self.train_benchmark.append(loss_total.item())
        pos_embed_attn_wt, pos_embed_affinity_wt = prop_pos_embed
        log_dict = {
            "Loss/train/total_loss" : loss_total, "Loss/train/focal_loss" : avg_focal, "Loss/train/dice_loss" : avg_dice, "Loss/train/iou_loss" : avg_iou,
            "pos_embed_attn_wt/min": pos_embed_attn_wt.min(), "pos_embed_attn_wt/max": pos_embed_attn_wt.max(), "pos_embed_attn_wt/mean": pos_embed_attn_wt.mean(), "pos_embed_attn_wt/std": pos_embed_attn_wt.std(),
            "pos_embed_affinity_wt/min": pos_embed_affinity_wt.min(), "pos_embed_affinity_wt/max": pos_embed_affinity_wt.max(), "pos_embed_affinity_wt/mean": pos_embed_affinity_wt.mean(), "pos_embed_affinity_wt/std": pos_embed_affinity_wt.std(),
            "prop_dense_embed/min": prop_dense_embed.min(), "prop_dense_embed/max": prop_dense_embed.max(), "prop_dense_embed/mean": prop_dense_embed.mean(), "prop_dense_embed/std": prop_dense_embed.std()
        }

        self.log_dict(log_dict, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=bs)

        return {'loss': loss_total, 'masks_0': pred_masks_list_0, 'masks_1': pred_masks_list_1, 'iou_0': iou_pred_list_0, 'iou_1': iou_pred_list_1} # List([num_true_obj, H, W])
    
    def validation_step(self, batch, batch_idx):
        img_embeddings = self.model.getImageEmbeddings(batch['image']) # (B, F=3, 256, 64, 64)
        output_0, low_res_pred, prop_pos_embed, prop_dense_embed = self.model.getPropEmbeddings0(img_embeddings, batch, multimask_output=self.cfg.model.multimask_output)

        pred_masks_list_0 = []
        iou_pred_list_0 = []
        pred_masks_list_1 = []
        iou_pred_list_1 = []
        low_res_pred_list_0 = []
        total_num_objects = 0

        # output_0, output_1, prop_pos_embed = self(batch, self.cfg.model.multimask_output)
        bs = len(output_0)
        loss_focal = 0
        loss_dice = 0
        loss_iou = 0
        loss_total = 0

        for each_output, gt_mask, selector, low_pred in zip(output_0, batch['gt_mask'], batch['selector'], low_res_pred): # selector = [True, True, False]
            total_num_objects += selector.sum()
            pred_masks = each_output["masks"] # [P=3, C, H, W]
            # pred_masks_list.append(pred_masks.detach())
            gt_mask = gt_mask[0].unsqueeze(1) # [P, 1, H, W] # gt_mask[0] to get the 0th mask from the prediction
            gt_mask = gt_mask.repeat((1, pred_masks.shape[1], 1, 1)) # [P, C, H, W] 
            
            pred_masks_sigmoid = torch.sigmoid(pred_masks)
            loss_tmp = (
                self.cfg.focal_wt * self.sigmoid_focal_loss(pred_masks, gt_mask, pred_masks_sigmoid)
                + self.dice_loss(pred_masks_sigmoid, gt_mask)
                + F.mse_loss(
                    each_output["iou_predictions"],
                    self.iou(pred_masks_sigmoid, gt_mask),
                    reduction="none",
                )
            ) # [P, C]

            loss_tmp, min_idx = torch.min(loss_tmp, -1) # [P]
            loss_total += loss_tmp[selector].sum() # (num_true_obj)
            batch_indexing = torch.arange(len(min_idx), device=min_idx.device) # [P]
            loss_focal += self.sigmoid_focal_loss(pred_masks, gt_mask, pred_masks_sigmoid)[batch_indexing, min_idx][selector].sum()
            loss_dice += self.dice_loss(pred_masks_sigmoid, gt_mask)[batch_indexing, min_idx][selector].sum()
            loss_iou += F.mse_loss(
                each_output["iou_predictions"], self.iou(pred_masks_sigmoid, gt_mask), reduction="none"
            )[batch_indexing, min_idx][selector].sum() # [num_true_obj]

            low_res_pred_list_0.append(low_pred[batch_indexing, min_idx]) # (P=3, C, 256, 256) -> (P=3, 256, 256)
            pred_masks_list_0.append(pred_masks_sigmoid[batch_indexing, min_idx][selector].detach())
            iou_pred_list_0.append(each_output["iou_predictions"][batch_indexing, min_idx][selector].detach())
        
        low_res_pred_list_0 = torch.stack(low_res_pred_list_0).unsqueeze(1) # (B, 1, P=3, 256, 256)

        if self.cfg.dataset.stage1:
            for idx in range(2, self.cfg.dataset.num_frames):
                output_1, low_res_pred = self.model.getPropEmbeddings1(img_embeddings, batch, low_res_pred_list_0, self.cfg.model.multimask_output, idx)
                low_res_pred_list_1 = []

                for each_output, gt_mask, selector, low_pred in zip(output_1, batch['gt_mask'], batch['selector'], low_res_pred): # selector = [True, True, False]
                    total_num_objects += selector.sum()
                    pred_masks = each_output["masks"] # [P=3, C, H, W]
                    # pred_masks_list.append(pred_masks.detach())
                    gt_mask = gt_mask[idx-1].unsqueeze(1) # [P, 1, H, W] # gt_mask[1] to get the 1st mask from the prediction
                    gt_mask = gt_mask.repeat((1, pred_masks.shape[1], 1, 1)) # [P, C, H, W] 

                    pred_masks_sigmoid = torch.sigmoid(pred_masks)
                    loss_tmp = (
                        self.cfg.focal_wt * self.sigmoid_focal_loss(pred_masks, gt_mask, pred_masks_sigmoid)
                        + self.dice_loss(pred_masks_sigmoid, gt_mask)
                        + F.mse_loss(
                            each_output["iou_predictions"],
                            self.iou(pred_masks_sigmoid, gt_mask),
                            reduction="none",
                        )
                    ) # [P, C]

                    loss_tmp, min_idx = torch.min(loss_tmp, -1) # [P]
                    loss_total += loss_tmp[selector].sum() # (num_true_obj)
                    batch_indexing = torch.arange(len(min_idx), device=min_idx.device) # [P]
                    loss_focal += self.sigmoid_focal_loss(pred_masks, gt_mask, pred_masks_sigmoid)[batch_indexing, min_idx][selector].sum()
                    loss_dice += self.dice_loss(pred_masks_sigmoid, gt_mask)[batch_indexing, min_idx][selector].sum()
                    loss_iou += F.mse_loss(
                        each_output["iou_predictions"], self.iou(pred_masks_sigmoid, gt_mask), reduction="none"
                    )[batch_indexing, min_idx][selector].sum() # [num_true_obj]

                    low_res_pred_list_1.append(low_pred[batch_indexing, min_idx]) # (P=3, C, 256, 256) -> (P=3, 256, 256)
                    pred_masks_list_1.append(pred_masks_sigmoid[batch_indexing, min_idx][selector].detach())
                    iou_pred_list_1.append(each_output["iou_predictions"][batch_indexing, min_idx][selector].detach())
                low_res_pred_list_1 = torch.stack(low_res_pred_list_1).unsqueeze(1) # (B, 1, P=3, 256, 256)
                low_res_pred_list_0 = torch.cat([low_res_pred_list_0, low_res_pred_list_1], dim=1)
        # Ex: focal - tensor(0.5012, device='cuda:0') dice - tensor(1.9991, device='cuda:0') iou - tensor(1.7245e-05, device='cuda:0')
        loss_total = (loss_total) / (total_num_objects)
        avg_focal = (self.cfg.focal_wt * loss_focal) / (total_num_objects)
        avg_dice = loss_dice / (total_num_objects)
        avg_iou = loss_iou / (total_num_objects)

        self.val_benchmark.append(loss_total.item())
        self.log_dict({"Loss/val/total_loss" : loss_total, "Loss/val/focal_loss" : avg_focal, "Loss/val/dice_loss" : avg_dice, "Loss/val/iou_loss" : avg_iou}, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=bs)

        return {'loss': loss_total, 'masks_0': pred_masks_list_0, 'masks_1': pred_masks_list_1, 'iou_0': iou_pred_list_0, 'iou_1': iou_pred_list_1} # List([num_true_obj, H, W])
    
    # def on_train_batch_end(self, output, batch, batch_idx):
    #     if self.check_frequency(self.current_epoch):
    #         img_list_0 = []
    #         img_list_1 = []
    #         sep_mask_list_0 = []
    #         sep_mask_list_1 = []
    #         mask_list_0 = []
    #         mask_list_1 = []
    #         gt_mask_list_0 = []
    #         gt_mask_list_1 = []
    #         sep_gt_mask_list_0 = []
    #         sep_gt_mask_list_1 = []

    #         jaccard_0 = 0
    #         dice_0 = 0
    #         jaccard_1 = 0
    #         dice_1 = 0
    #         total_objects = 0

    #         for each_output, gt_mask, cropped_img, selector in zip(output["masks_0"], batch['gt_mask'], batch['cropped_img'], batch['selector']):
    #             total_objects += selector.sum()
    #             gt_mask = gt_mask[0][selector]
    #             sep_mask_list_0.append(each_output>0.5)
    #             sep_gt_mask_list_0.append(gt_mask.type(torch.int8))

    #             j, d = jaccard_dice(each_output>0.5, gt_mask.type(torch.bool))
    #             jaccard_0 += j
    #             dice_0 += d

    #             max_, max_pos = torch.max(gt_mask, dim=0)
    #             gt_mask = ((max_pos+1) * (max_)).type(torch.int8)
    #             gt_mask_list_0.append(gt_mask)

    #             max_, max_pos = torch.max(each_output, dim=0)
    #             mask = ((max_pos+1) * (max_ > 0.5)).type(torch.int8)
    #             mask_list_0.append(mask)
                
    #             img_list_0.append(cropped_img[0])
    #         metrics_all = {'Metrics/train/jaccard_single_obj_0': jaccard_0 / total_objects, 'Metrics/train/dice_single_obj_0': dice_0 / total_objects}

    #         if self.cfg.dataset.stage1:
    #             for each_output, gt_mask, cropped_img, selector in zip(output["masks_1"], batch['gt_mask'], batch['cropped_img'], batch['selector']):
    #                 gt_mask = gt_mask[1][selector]
    #                 sep_mask_list_1.append(each_output>0.5) # (num_true_obj, H, W)
    #                 sep_gt_mask_list_1.append(gt_mask.type(torch.int8)) # (num_true_obj, H, W)

    #                 j, d = jaccard_dice(each_output>0.5, gt_mask.type(torch.bool))
    #                 jaccard_1 += j
    #                 dice_1 += d

    #                 max_, max_pos = torch.max(gt_mask, dim=0)
    #                 gt_mask = ((max_pos+1) * (max_)).type(torch.int8) # (H, W)
    #                 gt_mask_list_1.append(gt_mask)
                    
    #                 max_, max_pos = torch.max(each_output, dim=0)
    #                 mask = ((max_pos+1) * (max_ > 0.5)).type(torch.int8)
    #                 mask_list_1.append(mask)

    #                 img_list_1.append(cropped_img[1])

    #             metrics_all.update({'Metrics/train/jaccard_single_obj_1': jaccard_1 / total_objects, 'Metrics/train/dice_single_obj_1': dice_1 / total_objects})

    #         self.log_dict(metrics_all, on_step=True, on_epoch=True, sync_dist=True)
            
    #         # if batch_idx < 5:
    #         #     if self.cfg.dataset.stage1:
    #         #         self.log_images_1(batch['info'], img_list_0, img_list_1, sep_mask_list_0, sep_mask_list_1, mask_list_0, mask_list_1, gt_mask_list_0, gt_mask_list_1, output['iou_0'], output['iou_1'], batch_idx=batch_idx, train=True)
    #         #     else:
    #         #         self.log_images_0(batch['info'], img_list_0, sep_mask_list_0, mask_list_0, gt_mask_list_0, output['iou_0'], batch_idx=batch_idx, train=True)
    
    # def on_validation_batch_end(self, output, batch, batch_idx):
    #     img_list_0 = []
    #     img_list_1 = []
    #     sep_mask_list_0 = []
    #     sep_mask_list_1 = []
    #     mask_list_0 = []
    #     mask_list_1 = []
    #     gt_mask_list_0 = []
    #     gt_mask_list_1 = []
    #     sep_gt_mask_list_0 = []
    #     sep_gt_mask_list_1 = []

    #     jaccard_0 = 0
    #     dice_0 = 0
    #     jaccard_1 = 0
    #     dice_1 = 0
    #     total_objects = 0

    #     for each_output, gt_mask, cropped_img, selector in zip(output["masks_0"], batch['gt_mask'], batch['cropped_img'], batch['selector']):
    #         total_objects += selector.sum()
    #         gt_mask = gt_mask[0][selector]
    #         sep_mask_list_0.append(each_output>0.5)
    #         sep_gt_mask_list_0.append(gt_mask.type(torch.int8))

    #         j, d = jaccard_dice(each_output>0.5, gt_mask.type(torch.bool))
    #         jaccard_0 += j
    #         dice_0 += d

    #         max_, max_pos = torch.max(gt_mask, dim=0)
    #         gt_mask = ((max_pos+1) * (max_)).type(torch.int8)
    #         gt_mask_list_0.append(gt_mask)
            
    #         max_, max_pos = torch.max(each_output, dim=0)
    #         mask = ((max_pos+1) * (max_ > 0.5)).type(torch.int8)
    #         mask_list_0.append(mask)

    #         img_list_0.append(cropped_img[0])
    #     metrics_all = {'Metrics/val/jaccard_single_obj_0': jaccard_0 / total_objects, 'Metrics/val/dice_single_obj_0': dice_0 / total_objects}

    #     if self.cfg.dataset.stage1:
    #         for each_output, gt_mask, cropped_img, selector in zip(output["masks_1"], batch['gt_mask'], batch['cropped_img'], batch['selector']):
    #             gt_mask = gt_mask[1][selector]
    #             sep_mask_list_1.append(each_output>0.5) # (num_true_obj, H, W)
    #             sep_gt_mask_list_1.append(gt_mask.type(torch.int8)) # (num_true_obj, H, W)

    #             j, d = jaccard_dice(each_output>0.5, gt_mask.type(torch.bool))
    #             jaccard_1 += j
    #             dice_1 += d

    #             max_, max_pos = torch.max(gt_mask, dim=0)
    #             gt_mask = ((max_pos+1) * (max_)).type(torch.int8) # (H, W)
    #             gt_mask_list_1.append(gt_mask)
                
    #             max_, max_pos = torch.max(each_output, dim=0)
    #             mask = ((max_pos+1) * (max_ > 0.5)).type(torch.int8)
    #             mask_list_1.append(mask)

    #             img_list_1.append(cropped_img[1])

    #         metrics_all.update({'Metrics/val/jaccard_single_obj_1': jaccard_1 / total_objects, 'Metrics/val/dice_single_obj_1': dice_1 / total_objects})

    #     self.log_dict(metrics_all, on_step=True, on_epoch=True, sync_dist=True)

    #     # if self.cfg.dataset.stage1:
    #     #     self.log_images_1(batch['info'], img_list_0, img_list_1, sep_mask_list_0, sep_mask_list_1, mask_list_0, mask_list_1, gt_mask_list_0, gt_mask_list_1, output['iou_0'], output['iou_1'], batch_idx=batch_idx, train=False)
    #     # else:
    #     #     self.log_images_0(batch['info'], img_list_0, sep_mask_list_0, mask_list_0, gt_mask_list_0, output['iou_0'], batch_idx=batch_idx, train=False)