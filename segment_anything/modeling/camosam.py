# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch
from torch import nn
from torch.nn import functional as F

from typing import Any, Dict, List, Optional, Tuple

import lightning as L
from torchvision.utils import make_grid
import wandb

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
		
		for param in self.model.mask_decoder.parameters():
			param.requires_grad = True
	
		self.cfg = config
		self.batch_freq = self.cfg.eval_interval

	def forward(
		self,
		batched_input: List[Dict[str, Any]],
		multimask_output: bool,
	) -> List[Dict[str, torch.Tensor]]:
		"""
		Predicts masks end-to-end from provided images and prompts.
		If prompts are not known in advance, using SamPredictor is
		recommended over calling the model directly.

		Arguments:
		  batched_input (list(dict)): A list over input images, each a
			dictionary with the following keys. A prompt key can be
			excluded if it is not present.
			  'image': The image as a torch tensor in 3xHxW format,
				already transformed for input to the model.
			  'original_size': (tuple(int, int)) The original size of
				the image before transformation, as (H, W).
			  'point_coords': (torch.Tensor) Batched point prompts for
				this image, with shape BxNx2. Already transformed to the
				input frame of the model.
			  'point_labels': (torch.Tensor) Batched labels for point prompts,
				with shape BxN.
			  'boxes': (torch.Tensor) Batched box inputs, with shape Bx4.
				Already transformed to the input frame of the model.
			  'mask_inputs': (torch.Tensor) Batched mask inputs to the model,
				in the form Bx1xHxW.
		  multimask_output (bool): Whether the model should predict multiple
			disambiguating masks, or return a single mask.

		Returns:
		  (list(dict)): A list over input images, where each element is
			as dictionary with the following keys.
			  'masks': (torch.Tensor) Batched binary mask predictions,
				with shape BxCxHxW, where B is the number of input prompts,
				C is determined by multimask_output, and (H, W) is the
				original size of the image.
			  'iou_predictions': (torch.Tensor) The model's predictions
				of mask quality, in shape BxC.
			  'low_res_logits': (torch.Tensor) Low resolution logits with
				shape BxCxHxW, where H=W=256. Can be passed as mask input
				to subsequent iterations of prediction.
		"""
		return self.model(batched_input, multimask_output)
	
	def check_frequency(self, check_idx):
		return check_idx % self.batch_freq == 0
	
	@torch.no_grad()
	def log_images(self, batch, output, train=True):
		bs = len(batch)
		img_list = []
		gt_list = []
		mask_list = []
		for i in range(bs):
			batch_i = batch[i]
			output_i = output[i]
			img_list.append(batch_i['image'].permute(1, 2, 0))
			gt_list.append(batch_i['gt_mask'].squeeze())
			mask_list.append(output_i['masks'][0].squeeze())
		
		# num_maks = gt_list[0].shape[0]
		# gt_grid = make_grid(torch.cat(gt_list, dim=0), n_row=len(gt_list))
		# mask_grid = make_grid(torch.cat(mask_list, dim=0), nrow=len(mask_list))
		# self.log_dict({"images" : wandb.Image(make_grid(img_list)), "gt_mask" : wandb.Image(gt_grid), "masks" : wandb.Image(mask_grid)})

		table = wandb.Table(columns=['ID', 'Image'])

		for id, (img, gt, pred) in enumerate(zip(img_list, gt_list, mask_list)):
			mask_img = wandb.Image(img.cpu().numpy(), masks = {
				"prediction" : {
					"mask_data" : pred.cpu().numpy(),
				},
				"ground truth" : {"mask_data" : gt.cpu().numpy()}
			})
			
			table.add_data(id, mask_img)
		if train:
			wandb.log({"Train" : table})
		else:
			wandb.log({"Validation" : table})

	def sigmoid_focal_loss(
		self,
		inputs: torch.Tensor,
		targets: torch.Tensor,
		alpha: float = 0.25,
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

		p = torch.sigmoid(inputs)
		ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
		p_t = p * targets + (1 - p) * (1 - targets)
		loss = ce_loss * ((1 - p_t) ** gamma)

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
		inputs = inputs.sigmoid()
		inputs = inputs.flatten(1)
		targets = targets.flatten(1)
		numerator = 2 * (inputs * targets).sum(1)
		denominator = inputs.sum(-1) + targets.sum(-1)
		loss = 1 - (numerator + 1) / (denominator + 1)
		return loss.mean()

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
		scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, self.lr_lambda)
		return [optimizer], [scheduler]

	def training_step(self, batch, batch_idx):
		bs = len(batch)
		output = self(batch, False)

		loss_focal = 0
		loss_dice = 0
		loss_iou = 0

		for each_output, image_record in zip(output, batch):
			# compute batch_iou of pred_mask and gt_mask
			pred_masks = each_output["masks"].reshape(-1, each_output["masks"].shape[-2], each_output["masks"].shape[-1])
			gt_mask = image_record['gt_mask'].reshape(-1, image_record['gt_mask'].shape[-2], image_record['gt_mask'].shape[-1])
			num_masks = pred_masks.size(0)
			
			intersection = torch.sum(torch.mul(pred_masks, gt_mask), dim=(-1, -2))
			union = torch.sum(pred_masks, dim=(-1, -2))
			epsilon = 1e-7

			batch_iou = (intersection / (union + epsilon))
			loss_focal += self.sigmoid_focal_loss(pred_masks, gt_mask, reduction="mean")
			loss_dice += self.dice_loss(pred_masks, gt_mask)
			loss_iou += F.mse_loss(
				each_output["iou_predictions"].reshape(-1), batch_iou, reduction="mean"
			)

		loss_total = (20.0 * loss_focal + loss_dice + loss_iou) / bs
		avg_focal = loss_focal.item() / bs
		avg_dice = loss_dice.item() / bs
		avg_iou = loss_iou.item() / bs

		if self.check_frequency(batch_idx):
			self.log_images(batch, output)
		self.log_dict({"train_total_loss" : loss_total, "train_focal_loss" : avg_focal, "train_dice_loss" : avg_dice, "train_iou_loss" : avg_iou}, on_step=True, on_epoch=True, prog_bar=True)

		return loss_total
	
	def validation_step(self, batch, batch_idx):
		bs = len(batch)
		output = self(batch, False)

		loss_focal = 0
		loss_dice = 0
		loss_iou = 0

		for each_output, image_record in zip(output, batch):
			# compute batch_iou of pred_mask and gt_mask
			pred_masks = each_output["masks"].reshape(-1, each_output["masks"].shape[-2], each_output["masks"].shape[-1])
			gt_mask = image_record['gt_mask'].reshape(-1, image_record['gt_mask'].shape[-2], image_record['gt_mask'].shape[-1])
			num_masks = pred_masks.size(0)

			intersection = torch.sum(torch.mul(pred_masks, gt_mask), dim=(-1, -2))
			union = torch.sum(pred_masks, dim=(-1, -2))
			epsilon = 1e-7

			batch_iou = (intersection / (union + epsilon))
			loss_focal += self.sigmoid_focal_loss(pred_masks, gt_mask, reduction="mean")
			loss_dice += self.dice_loss(pred_masks, gt_mask)
			loss_iou += F.mse_loss(
				each_output["iou_predictions"].reshape(-1), batch_iou, reduction="mean"
			)

		loss_total = (20.0 * loss_focal + loss_dice + loss_iou) / bs
		avg_focal = loss_focal.item() / bs
		avg_dice = loss_dice.item() / bs
		avg_iou = loss_iou.item() / bs

		self.log_dict({"val_total_loss" : loss_total, "val_focal_loss" : avg_focal, "val_dice_loss" : avg_dice, "val_iou_loss" : avg_iou})

		return loss_total