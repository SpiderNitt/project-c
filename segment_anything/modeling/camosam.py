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
import cv2
import matplotlib.pyplot as plt
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
        self.cfg = config
        if ckpt is not None:
            self.model.propagation_module.load_state_dict(ckpt['model_state_dict'])
            print("!!! Loaded checkpoint for propagation module !!!")
            if ckpt['decoder_state_dict']:
                self.model.mask_decoder.load_state_dict(ckpt['decoder_state_dict'])
                print("!!! Loaded checkpoint for mask decoder !!!")

        self.set_requires_grad()

        # for n,p in self.model.named_parameters():
        #     if p.requires_grad:
        #         print(n)
                
        self.epoch_freq = 1#self.cfg.train_metric_interval
        
        self.train_benchmark = []
        self.val_benchmark = []
        
        self.learning_rate = learning_rate #auto_lr_find = True

    def set_requires_grad(self):
        model_grad = self.cfg.model.requires_grad

        for name,param in self.model.named_parameters():
            if 'adapter' in name or 'feature_extractor' in name :
                param.requires_grad = True
            
            else:
                param.requires_grad = False

        # for param in self.model. image_encoder.parameters():
        #     param.requires_grad = False

        # for param in self.model.prompt_encoder.parameters():
        #     param.requires_grad = model_grad.prompt_encoder

        # for param in self.model.mask_decoder.parameters():
        #     param.requires_grad = model_grad.mask_decoder
        
        # for param in self.model.propagation_module.parameters():
        #     param.requires_grad = model_grad.propagation_module

    def configure_optimizers(self) -> Any:
        print("Using Learning Rate: ", self.learning_rate if self.learning_rate else self.cfg.opt.learning_rate, f"instead of {self.cfg.opt.learning_rate}")
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr= self.learning_rate if self.learning_rate else self.cfg.opt.learning_rate,
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
    def log_images(self, img_list, sep_mask_list, gt_mask_list, iou_list,prompt_point,prompt_label,prompt_mask, img_shapes, org_point_prompt, batch_idx, train=True):
        
        table = wandb.Table(columns=['ID', 'Image', 'IoU'])
        # print(len(img_list), len(sep_mask_list), len(gt_mask_list), len(prompt_point), len(prompt_mask))
        for id, (img, gt, pred, point, mask, iou, img_shapes_, org_point_prompt_) in enumerate(zip(img_list, gt_mask_list, sep_mask_list, prompt_point, prompt_mask, iou_list, img_shapes, org_point_prompt)):
            # print(img_shapes_)
            pred = pred.int()
            gt = gt.int()
            
            gt[gt!=0] = 255
            for i in range(len(pred)):
                pred[i,pred[i]!=0] = 200-i*10 #contrast
            
            # print(pred.shape) #torch.Size([2, 1, 650, 1024])
            pred = pred.squeeze(1).cpu().detach().numpy()
            # print(pred.shape)
            gt = gt.cpu().numpy()
            # img = img.permute(1,2,0)
            # print(img.shape, gt.shape, pred.shape, point.shape, mask.shape, iou.shape)
            # print(gt.unique(), pred.unique(), mask.unique())
            
            log_masks = {}
            for i in range(len([pred])):
                # print(f"pred {pred[i].shape}, gt {gt[i].shape}")
                log_masks[f'prediction_{i}'] = { "mask_data" : pred[i]}
                log_masks[f'ground truth_{i}'] = {"mask_data" : gt[i]}


            if mask is not None:
                mask = mask.cpu().squeeze(1)
                # print(mask.shape)
                for i in range(len(mask)): # each instance
                    # print(mask[i].shape, mask[i].unique())
                    # plt.subplot(1,3,1)
                    # plt.imshow(img)
                    # plt.subplot(1,3,2)
                    # plt.imshow(mask[i].numpy())
                    # plt.subplot(1,3,3)
                    # plt.imshow(gt[i])
                    # plt.show()
                    # print(mask[i].shape, mask[i].unique())
                    
                    mask_ = cv2.resize(mask[i].numpy()*255, (1024,1024), interpolation=cv2.INTER_NEAREST)
                    mask_ = mask_[:img_shapes_[1], :img_shapes_[2]]
                    # print(mask_.shape)
                    mask_ = cv2.resize(mask_, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
                    # mask_ = self.model.postprocess_masks(mask[i].unsqueeze(0).unsqueeze(0)*255, input_size=(256,256), original_size=img.shape[:-1]).squeeze().squeeze()
                    ## do overlay
                    # plt.subplot(1,3,1)
                    # plt.imshow(img)
                    # plt.subplot(1,3,2)
                    # plt.imshow(mask_)
                    # plt.subplot(1,3,3)
                    # plt.imshow(gt[i])
                    # plt.show()
                    # plt.imshow(img)
                    # plt.imshow(mask_, alpha=0.5)
                    # plt.show()
                    # print(mask_.shape, np.unique(mask_))
                    # mask_ = cv2.resize(mask[i].numpy(), (1024,1024), interpolation=cv2.INTER_NEAREST)
                    # mask_ = mask_[:img.shape[0], :img.shape[1]]

                    mask_[mask_!=0] = 150-i*10
                    log_masks[f"mask prompt_{i}"] = {"mask_data" : mask_}
                    # print(mask_.shape)
                 
            if point is not None:
                # print(point, img_shapes_, org_point_prompt_, org_point_prompt_.shape)
                # print(org_point_prompt_.shape, point.shape)
                # point[..., 0] = point[..., 0] * (img_shapes_[2] / 1024)
                # point[..., 1] = point[..., 1] * (img_shapes_[1] / 1024)
                
                # print(point)
                # print(point)
                org_point_prompt_ = org_point_prompt_.int().cpu().detach().numpy()
                for i in range(len(point)): # each instance
                    point_as_mask = np.zeros((img.shape[:-1]))
                    for coords in org_point_prompt_[i]:
                        # print("COORDS",coords, coords.shape)
                        
                        # coords = coords[::-1]
                        point_as_mask = cv2.circle(point_as_mask, coords, radius=10, color=100, thickness=-1)   
                    point_as_mask[point_as_mask!=0] = 100+i*10
                    # print(f"point_as_mask {point_as_mask.shape}")
                    # plt.imshow(gt[i])
                    # plt.imshow(point_as_mask, alpha=0.5)
                    # plt.show()
                    
                    log_masks[f"point prompt_{i}"] = {"mask_data": point_as_mask}
                          
            # for i in log_masks:
            #     print(f"{i} {log_masks[i]['mask_data'].shape}")
            mask_img = wandb.Image(img, masks=log_masks)
            table.add_data(id, mask_img, iou)

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

    def training_step(self, batch, batch_idx):
    
                 
        output = self(batch, False)
        bs = len(output)
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

        pred_masks_list = []
        iou_pred_list = []
        
        total_num_objects = 0
        # print(f'batch_idx{batch_idx}')
        for each_output, each_input in zip(output, batch):
            
            # print(each_input['point_coords'])
            # print(batch['gt_mask'].shape)
            # total_num_objects += selector.sum()
            # print(pred_masks.shape)
            pred_masks = each_output["masks"] # [1, H, W]
            # print(f'pred_masks {pred_masks.shape}')
            # plt.subplot(1,2,1)
            # plt.imshow(pred_masks[0][0].detach().cpu())
            # plt.show()
            # pred_masks_list.append(pred_masks.detach())
            # pred_masks = pred_masks[selector] # [num_true_obj, H, W]
            gt_mask = each_input['gt_mask'] # [num_obj, H, W] 
            # print(f'gt_mask {gt_mask.shape}')

            
            pred_masks_sigmoid = torch.sigmoid(pred_masks)
            pred_masks_list.append(pred_masks_sigmoid)
            iou_pred_list.append(each_output["iou_predictions"].reshape(-1))
            total_num_objects += each_input['gt_mask'].shape[0]

            loss_focal = loss_focal + self.sigmoid_focal_loss(pred_masks, gt_mask, pred_masks_sigmoid)
            # print(f'loss_focal {loss_focal.requires_grad}')
            loss_dice = loss_dice + self.dice_loss(pred_masks_sigmoid, gt_mask)
            loss_iou = loss_iou + F.mse_loss(
                each_output["iou_predictions"].reshape(-1), self.iou(pred_masks_sigmoid, gt_mask), reduction="sum"
            )

        # Ex: focal - tensor(0.5012, device='cuda:0') dice - tensor(1.9991, device='cuda:0') iou - tensor(1.7245e-05, device='cuda:0')
        loss_total = (self.cfg.focal_wt * loss_focal + loss_dice + loss_iou) / (total_num_objects)
        avg_focal = (self.cfg.focal_wt * loss_focal) / (total_num_objects)
        avg_dice = loss_dice / (total_num_objects)
        avg_iou = loss_iou / (total_num_objects)
        
        self.train_benchmark.append(loss_total.item())
        # pos_embed_attn_wt, pos_embed_affinity_wt = prop_pos_embed
        log_dict = {
            "Loss/train/total_loss" : loss_total, "Loss/train/focal_loss" : avg_focal, "Loss/train/dice_loss" : avg_dice, "Loss/train/iou_loss" : avg_iou,
        }

        self.log_dict(log_dict, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=bs)

        return {'loss': loss_total, 'masks': pred_masks_list,  'iou': iou_pred_list} # List([num_true_obj, H, W])
    
    def validation_step(self, batch, batch_idx):
        output_val = self(batch, False)
        bs = len(output_val)
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

        pred_masks_list = []
        iou_pred_list = []
        total_num_objects = 0

        for each_output, each_input in zip(output_val, batch ):
            # print(batch['gt_mask'].shape)
            # total_num_objects += selector.sum()
            # print(pred_masks.shape)
            
            pred_masks = each_output["masks"] # [1, H, W]
            # print(f'pre_masks{pred_masks.shape}')
            # pred_masks_list.append(pred_masks.detach())
            # pred_masks = pred_masks[selector] # [num_true_obj, H, W]
            gt_mask = each_input['gt_mask'] # [num_obj, H, W] # gt_mask[0] to get the 0th mask from the prediction
            total_num_objects += each_input['gt_mask'].shape[0]
            pred_masks_sigmoid = torch.sigmoid(pred_masks)
            pred_masks_list.append(pred_masks_sigmoid.detach())
            iou_pred_list.append(each_output["iou_predictions"].reshape(-1).detach())
            
            loss_focal += self.sigmoid_focal_loss(pred_masks, gt_mask, pred_masks_sigmoid)
            loss_dice += self.dice_loss(pred_masks_sigmoid, gt_mask)
            loss_iou += F.mse_loss(
                each_output["iou_predictions"].reshape(-1), self.iou(pred_masks_sigmoid, gt_mask), reduction="sum"
            )

        # Ex: focal - tensor(0.5012, device='cuda:0') dice - tensor(1.9991, device='cuda:0') iou - tensor(1.7245e-05, device='cuda:0')
        loss_total = (self.cfg.focal_wt * loss_focal + loss_dice + loss_iou) / (total_num_objects)
        avg_focal = (self.cfg.focal_wt * loss_focal) / (total_num_objects)
        avg_dice = loss_dice / (total_num_objects)
        avg_iou = loss_iou / (total_num_objects)

        # Ex: focal - tensor(0.5012, device='cuda:0') dice - tensor(1.9991, device='cuda:0') iou - tensor(1.7245e-05, device='cuda:0')
        loss_total = (self.cfg.focal_wt * loss_focal + loss_dice + loss_iou) / (total_num_objects)
        avg_focal = (self.cfg.focal_wt * loss_focal) / (total_num_objects)
        avg_dice = loss_dice / (total_num_objects)
        avg_iou = loss_iou / (total_num_objects)

        self.val_benchmark.append(loss_total.item())
        self.log_dict({"Loss/val/total_loss" : loss_total, "Loss/val/focal_loss" : avg_focal, "Loss/val/dice_loss" : avg_dice, "Loss/val/iou_loss" : avg_iou}, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=bs)

        return {'loss': loss_total, 'masks': pred_masks_list, 'iou': iou_pred_list} # List([num_true_obj, H, W])
    
    def on_train_batch_end(self, output, batch, batch_idx):
        if self.check_frequency(self.current_epoch):
            img_list_0 = []
            # img_list_1 = []
            sep_mask_list_0 = []
            # sep_mask_list_1 = []
            mask_list_0 = []
            # mask_list_1 = []
            gt_mask_list_0 = []
            # gt_mask_list_1 = []
            sep_gt_mask_list_0 = []
            prompt_point = []
            prompt_label = []
            prompt_mask = []
            img_shapes = []
            org_point_prompt = []
            # sep_gt_mask_list_1 = []

            jaccard_0 = 0
            dice_0 = 0
            # jaccard_1 = 0
            # dice_1 = 0
            total_objects = 0

            for each_output, each_input in zip(output["masks"], batch):
                # total_objects += selector.sum()
                prompt_point.append(each_input.get('point_coords',None))
                prompt_label.append(each_input.get('point_labels',None))
                prompt_mask.append(each_input.get('mask_inputs',None))
                img_shapes.append(each_input.get('image',None).shape)
                org_point_prompt.append(each_input.get('point_coords_original',None))
            
                gt_mask = each_input['gt_mask']
                sep_mask_list_0.append(each_output>0.5)
                sep_gt_mask_list_0.append(gt_mask.type(torch.int8))

                j, d = jaccard_dice(each_output>0.5, gt_mask.type(torch.bool))
                jaccard_0 += j
                dice_0 += d

                max_, max_pos = torch.max(gt_mask, dim=0)
                gt_mask = ((max_pos+1) * (max_)).type(torch.int8)
                gt_mask_list_0.append(gt_mask)

                max_, max_pos = torch.max(each_output, dim=0)
                mask = ((max_pos+1) * (max_ > 0.5)).type(torch.int8)
                mask_list_0.append(mask)
                total_objects += each_input['gt_mask'].shape[0]
                img_list_0.append(each_input["org_img"])
                metrics_all = {'Metrics/train/jaccard_single_obj_0': jaccard_0 / total_objects, 'Metrics/train/dice_single_obj_0': dice_0 / total_objects}

                metrics_all.update({'Metrics/train/jaccard_single': jaccard_0 / total_objects, 'Metrics/train/dice_single': dice_0 / total_objects})

            self.log_dict(metrics_all, on_step=True, on_epoch=True, sync_dist=True)
            
            self.log_images(img_list_0, sep_mask_list_0, sep_gt_mask_list_0, output['iou'], prompt_point, prompt_label, prompt_mask, img_shapes, org_point_prompt,  batch_idx=batch_idx, train=True)
    
    def on_validation_batch_end(self, output, batch, batch_idx):
        img_list_0 = []
        # img_list_1 = []
        sep_mask_list_0 = []
        # sep_mask_list_1 = []
        mask_list_0 = []
        # mask_list_1 = []
        gt_mask_list_0 = []
        # gt_mask_list_1 = []
        sep_gt_mask_list_0 = []
        
        prompt_point = []
        prompt_label = []
        prompt_mask = []
        img_shapes = []
        org_point_prompt = []
        # sep_gt_mask_list_1 = []

        jaccard_0 = 0
        dice_0 = 0
        # jaccard_1 = 0
        # dice_1 = 0
        total_objects = 0

        for each_output, each_input in zip(output["masks"], batch):
            # total_objects += selector.sum()
            prompt_point.append(each_input.get('point_coords',None))
            prompt_label.append(each_input.get('point_labels',None))
            prompt_mask.append(each_input.get('mask_inputs',None))
            img_shapes.append(each_input.get('image',None).shape)
            org_point_prompt.append(each_input.get('point_coords_original',None))
            
            gt_mask = each_input['gt_mask']
            sep_mask_list_0.append(each_output>0.5)
            sep_gt_mask_list_0.append(gt_mask.type(torch.int8))

            j, d = jaccard_dice(each_output>0.5, gt_mask.type(torch.bool))
            jaccard_0 += j
            dice_0 += d

            max_, max_pos = torch.max(gt_mask, dim=0)
            gt_mask = ((max_pos+1) * (max_)).type(torch.int8)
            gt_mask_list_0.append(gt_mask)
            
            max_, max_pos = torch.max(each_output, dim=0)
            mask = ((max_pos+1) * (max_ > 0.5)).type(torch.int8)
            mask_list_0.append(mask)
            total_objects += each_input['gt_mask'].shape[0]
            img_list_0.append(each_input["org_img"])
        metrics_all = {'Metrics/val/jaccard_single_obj_0': jaccard_0 / total_objects, 'Metrics/val/dice_single_obj_0': dice_0 / total_objects}

        # if self.cfg.dataset.stage1:
        #     for each_output, gt_mask, cropped_img, selector in zip(output["masks_1"], batch['gt_mask'], batch['cropped_img'], batch['selector']):
        #         gt_mask = gt_mask[1][selector]
        #         sep_mask_list_1.append(each_output>0.5) # (num_true_obj, H, W)
        #         sep_gt_mask_list_1.append(gt_mask.type(torch.int8)) # (num_true_obj, H, W)

        #         j, d = jaccard_dice(each_output>0.5, gt_mask.type(torch.bool))
        #         jaccard_1 += j
        #         dice_1 += d

        #         max_, max_pos = torch.max(gt_mask, dim=0)
        #         gt_mask = ((max_pos+1) * (max_)).type(torch.int8) # (H, W)
        #         gt_mask_list_1.append(gt_mask)
                
        #         max_, max_pos = torch.max(each_output, dim=0)
        #         mask = ((max_pos+1) * (max_ > 0.5)).type(torch.int8)
        #         mask_list_1.append(mask)

        #         img_list_1.append(cropped_img[1])

        #     metrics_all.update({'Metrics/val/jaccard_single_obj_1': jaccard_1 / total_objects, 'Metrics/val/dice_single_obj_1': dice_1 / total_objects})

        self.log_dict(metrics_all, on_step=True, on_epoch=True, sync_dist=True)


        self.log_images(img_list_0, sep_mask_list_0, sep_gt_mask_list_0, output['iou'], prompt_point, prompt_label, prompt_mask, img_shapes, org_point_prompt,  batch_idx=batch_idx, train=False)