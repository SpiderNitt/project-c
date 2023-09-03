# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch
from torch import nn
from torch.nn import functional as F

from typing import Any, Dict, List, Optional, Tuple

from .image_encoder import ImageEncoderViT
from .mask_decoder import MaskDecoder
from .prompt_encoder import PromptEncoder

import lightning as L
from torchvision.utils import make_grid
import wandb

class Sam(L.LightningModule):
	mask_threshold: float = 0.0
	image_format: str = "RGB"

	def __init__(
		self,
		config,
		image_encoder: ImageEncoderViT,
		prompt_encoder: PromptEncoder,
		mask_decoder: MaskDecoder,
		pixel_mean: List[float] = [123.675, 116.28, 103.53],
		pixel_std: List[float] = [58.395, 57.12, 57.375],
	) -> None:
		"""
		SAM predicts object masks from an image and input prompts.

		Arguments:
		  image_encoder (ImageEncoderViT): The backbone used to encode the
			image into image embeddings that allow for efficient mask prediction.
		  prompt_encoder (PromptEncoder): Encodes various types of input prompts.
		  mask_decoder (MaskDecoder): Predicts masks from the image embeddings
			and encoded prompts.
		  pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
		  pixel_std (list(float)): Std values for normalizing pixels in the input image.
		"""
		super().__init__()
		self.image_encoder = image_encoder
		self.prompt_encoder = prompt_encoder
		self.mask_decoder = mask_decoder
		self.register_buffer(
			"pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False
		)
		self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
		self.cfg=config

	@property
	def device(self) -> Any:
		return self.pixel_mean.device

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
		input_images = torch.stack(
			[self.preprocess(x["image"]) for x in batched_input], dim=0
		)
		image_embeddings = self.image_encoder(input_images)

		outputs = []
		for image_record, curr_embedding in zip(batched_input, image_embeddings):
			sparse_embeddings, dense_embeddings = self.prompt_encoder(
				points=None,
				boxes=None,
				masks=None,
			)
			low_res_masks, iou_predictions = self.mask_decoder(
				image_embeddings=curr_embedding.unsqueeze(0),
				image_pe=self.prompt_encoder.get_dense_pe(),
				sparse_prompt_embeddings=sparse_embeddings,
				dense_prompt_embeddings=dense_embeddings,
				multimask_output=multimask_output,
			)
			masks = self.postprocess_masks(
				low_res_masks,
				input_size=image_record["image"].shape[-2:],
				original_size=image_record["original_size"],
			)
			masks = masks > self.mask_threshold
			outputs.append(
				{
					"masks": masks, # (num_prompts, num_masks_per_prompt, H, W) -> (1, 1, H, W)
					"iou_predictions": iou_predictions, # (num_prompts, num_masks_per_prompt) -> (1, 1)
					"low_res_logits": low_res_masks,
				}
			)
		return outputs

	def postprocess_masks(
		self,
		masks: torch.Tensor,
		input_size: Tuple[int, ...],
		original_size: Tuple[int, ...],
	) -> torch.Tensor:
		"""
		Remove padding and upscale masks to the original image size.

		Arguments:
		  masks (torch.Tensor): Batched masks from the mask_decoder,
			in BxCxHxW format.
		  input_size (tuple(int, int)): The size of the image input to the
			model, in (H, W) format. Used to remove padding.
		  original_size (tuple(int, int)): The original size of the image
			before resizing for input to the model, in (H, W) format.

		Returns:
		  (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
			is given by original_size.
		"""
		masks = F.interpolate(
			masks,
			(self.image_encoder.img_size, self.image_encoder.img_size),
			mode="bilinear",
			align_corners=False,
		)
		masks = masks[..., : input_size[0], : input_size[1]]
		masks = F.interpolate(
			masks, original_size, mode="bilinear", align_corners=False
		)
		return masks

	def preprocess(self, x: torch.Tensor) -> torch.Tensor:
		"""Normalize pixel values and pad to a square input."""
		# Normalize colors
		x = (x - self.pixel_mean) / self.pixel_std

		# Pad
		h, w = x.shape[-2:]
		padh = self.image_encoder.img_size - h
		padw = self.image_encoder.img_size - w
		x = F.pad(x, (0, padw, 0, padh))
		return x
	
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
			gt_list.append(batch_i['gt_masks'].squeeze())
			mask_list.append(output_i['masks'][0].squeeze())
		
		# num_maks = gt_list[0].shape[0]
		# gt_grid = make_grid(torch.cat(gt_list, dim=0), n_row=len(gt_list))
		# mask_grid = make_grid(torch.cat(mask_list, dim=0), nrow=len(mask_list))
		# self.log_dict({"images" : wandb.Image(make_grid(img_list)), "gt_masks" : wandb.Image(gt_grid), "masks" : wandb.Image(mask_grid)})

		table = wandb.Table(columns=['ID', 'Image'])

		for id, (img, gt, pred) in enumerate(zip(img_list, gt_list, mask_list)):
			mask_img = wandb.Image(img, masks = {
				"prediction" : {
					"mask_data" : pred,
				},
				"ground truth" : {"mask_data" : gt}
			})
			
			table.add_data(id, mask_img)
		if train:
			self.log({"Train" : table})
		else:
			self.log({"Validation" : table})

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
		return optimizer, scheduler

	def training_step(self, batch, batch_idx):
		bs = len(batch)
		output = self(batch, False)

		loss_focal = 0
		loss_dice = 0
		loss_iou = 0

		for each_output, image_record in zip(output, batch):
			# compute batch_iou of pred_mask and gt_mask
			pred_masks = each_output["masks"].reshape(-1, each_output["masks"].shape[-2], each_output["masks"].shape[-1])
			gt_mask = image_record['gt_masks'].reshape(-1, image_record['gt_masks'].shape[-2], image_record['gt_masks'].shape[-1])
			num_masks = pred_masks.shape(0)
			
			intersection = torch.sum(torch.mul(pred_masks, gt_mask), dim=(-1, -2))
			union = torch.sum(pred_masks, dim=(-1, -2))
			epsilon = 1e-7

			batch_iou = (intersection / (union + epsilon))
			loss_focal += self.sigmoid_focal_loss(pred_masks, gt_mask, reduction="mean")
			loss_dice += self.dice_loss(pred_masks, gt_mask)
			loss_iou += F.mse_loss(
				each_output["iou_predictions"], batch_iou, reduction="mean"
			)

		loss_total = (20.0 * loss_focal + loss_dice + loss_iou) / bs
		avg_focal = loss_focal.item() / bs # compute average loss of a batch
		avg_dice = loss_dice.item() / bs
		avg_iou = loss_iou.item() / bs
		avg_total = loss_total.item() / bs
		self.log_dict({"train_total_loss" : avg_total, "train_focal_loss" : avg_focal, "train_dice_loss" : avg_dice, "train_iou_loss" : avg_iou}, on_step=True, on_epoch=True, prog_bar=True)

		return avg_total
	
	def validation_step(self, batch, batch_idx):
		bs = len(batch)
		output = self(batch, False)

		loss_focal = 0
		loss_dice = 0
		loss_iou = 0

		for each_output, image_record in zip(output, batch):
			# compute batch_iou of pred_mask and gt_mask
			pred_masks = each_output["masks"].reshape(-1, each_output["masks"].shape[-2], each_output["masks"].shape[-1])
			gt_mask = image_record['gt_masks'].reshape(-1, image_record['gt_masks'].shape[-2], image_record['gt_masks'].shape[-1])
			num_masks = pred_masks.shape(0)

			intersection = torch.sum(torch.mul(pred_masks, gt_mask), dim=(-1, -2))
			union = torch.sum(pred_masks, dim=(-1, -2))
			epsilon = 1e-7

			batch_iou = (intersection / (union + epsilon))
			loss_focal += self.sigmoid_focal_loss(pred_masks, gt_mask, reduction="mean")
			loss_dice += self.dice_loss(pred_masks, gt_mask)
			loss_iou += F.mse_loss(
				each_output["iou_predictions"], batch_iou, reduction="mean"
			)

		loss_total = (20.0 * loss_focal + loss_dice + loss_iou) / bs
		avg_focal = loss_focal.item() / bs # compute average loss of a batch
		avg_dice = loss_dice.item() / bs
		avg_iou = loss_iou.item() / bs
		avg_total = loss_total.item() / bs
		self.log_dict({"val_total_loss" : avg_total, "val_focal_loss" : avg_focal, "val_dice_loss" : avg_dice, "val_iou_loss" : avg_iou})

		return avg_total