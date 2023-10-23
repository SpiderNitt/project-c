# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F

from typing import Any, Dict, List, Tuple

from .image_encoder import ImageEncoderViT
from .mask_decoder import MaskDecoder
from .prompt_encoder import PromptEncoder
from .modules import PropagationModule


class Sam(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        image_encoder: ImageEncoderViT,
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
        cfg = None
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
        self.cfg = cfg
        self.max_num_obj = cfg.dataset.max_num_obj
        self.num_frames = cfg.dataset.num_frames
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder

        self.propagation_module = PropagationModule(cfg)

        self.register_buffer(
            "pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False
        )
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
    
    @property
    def device(self) -> Any:
        return self.pixel_mean.device

    def forward(
        self,
        batched_input: List[Dict[str, Any]],
        multimask_output: bool = False,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Predicts masks end-to-end from provided images and prompts.
        If prompts are not known in advance, using SamPredictor is
        recommended over calling the model directly.
        
        B: Video Batch Size
        F: Number of frames
        P: Number of prompts/objects
        
        Arguments:
        batched_input (list(dict)): A list over input image, each a
        dictionary with the following keys. A prompt key can be
        excluded if it is not present.
            image': The frames as a torch tensor in Fx3xHxW format,
                already transformed for input to the model.
            'gt_mask': (F, H, W) The ground truth mask for the last frame
            'prev_masks': (F-1, P, 256, 256) The ground truth masks for the previous frames preprocessed for mask encoder
            'selector': (P) The selector for the current video [True, True, False] indicates 2 objects
            'info': 
            'original_size': (tuple(int, int)) The original size of
                the image before transformation, as (H, W).
        Returns:
        (list(dict)): A list over input images, where each element is
        as dictionary with the following keys.
            'masks': (torch.Tensor) Batched binary mask predictions,
            with shape PxCxHxW, where P is the number of input prompts/objects,
            C is determined by multimask_output, and (H, W) is the
            original size of the image.
            'iou_predictions': (torch.Tensor) The model's predictions
            of mask quality, in shape PxC.
            'low_res_logits': (torch.Tensor) Low resolution logits with
            shape PxCxHxW, where H=W=256. Can be passed as mask input
            to subsequent iterations of prediction.   
        """
        input_images = batched_input["image"] # Output -> (B, num_frames=3, 3, 1024, 1024)
        
        '''
        To disbale grad for previous frames alone
        prev_frames, current_frame = input_images[:, :-1], input_images[:, -1]
        current_frame_embeddings = self.image_encoder(current_frame)# Output -> (B, 256, 64, 64)
        with torch.no_grad():
            prev_frames_embeddings = self.image_encoder(input_images.reshape(-1, 3, 1024, 1024)).reshape(input_images.shape[0], self.num_frames - 1, 256, 64, 64)
        torch.cuda.empty_cache()
        '''
        # We disable grad since we are not updating the image encoder weights
        self.image_encoder.eval()
        with torch.no_grad():
            image_embeddings = self.image_encoder(input_images.reshape(-1, 3, 1024, 1024)).reshape(len(batched_input["selector"]), self.num_frames, 256, 64, 64)  # Output -> (B, F=3, 256, 64, 64)
        # torch.cuda.empty_cache()
        # image_embeddings = torch.randn(len(batched_input["selector"]), self.num_frames, 256, 64, 64).to(self.device)

        pos_embed = self.prompt_encoder.get_dense_pe() # (256, 64, 64)
        
        prev_masks_0 = prev_masks_1 = batched_input["prev_masks"] # (B, F=2/1, P=3, 256, 256)
        prev_masks_0 = prev_masks_0.view(-1, 1, *prev_masks_0.shape[-2:])
        _, mask_embeddings_0 = self.prompt_encoder(points=None, boxes=None, masks=prev_masks_0)
        mask_embeddings_0 = mask_embeddings_0.view(len(batched_input["selector"]), -1, self.max_num_obj, 256, 64, 64) # (B, F=2/1, P=3, 256, 64, 64)
        
        # embeddings = {"current_frame_embeddings": current_frame_embeddings, "prev_frames_embeddings": prev_frames_embeddings, "mask_embeddings": mask_embeddings}
        embeddings_0 = {"image_embeddings": image_embeddings[:, :(2 if self.cfg.dataset.stage1 else 3)], "mask_embeddings": mask_embeddings_0}
        
        all_sparse_embeddings_0, all_dense_embeddings_0, prop_pos_embed = self.propagation_module(
            embeddings_0, pos_embed
        )  # (B, P=3, 64, 64, 256)
        all_dense_embeddings_0 = all_dense_embeddings_0.permute(0, 1, 4, 2, 3) # (B, P=3, 256, 64, 64)

        low_res_pred = []
        outputs_0 = []
        for i, (curr_embedding, prop_sparse_embeddings, prop_dense_embeddings) in enumerate(zip(image_embeddings[:, (-2 if self.cfg.dataset.stage1 else -1)], all_sparse_embeddings_0, all_dense_embeddings_0)):
            # curr_embedding: (256, 64, 64) -> current target frame embedding
            # prop_dense_embeddings: (3, 256, 64, 64) -> basically we have 3 prompts
            # prop_sparse_embeddings: (3, 8, 256) -> basically we have 3 prompts, each prompt has 8 points
            
            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=curr_embedding.unsqueeze(0),
                image_pe=pos_embed,
                sparse_prompt_embeddings=prop_sparse_embeddings,
                dense_prompt_embeddings=prop_dense_embeddings,
                multimask_output=multimask_output,
            )
            low_res_pred.append(low_res_masks.squeeze(1).unsqueeze(0)) # (P=3, 1, 256, 256) -> (1, P=3, 256, 256)
            masks = self.postprocess_masks(
                low_res_masks,
                input_size=list(batched_input["resize_longest_size"][i]),
                original_size=list(batched_input["original_size"][i])
            )
            outputs_0.append(
                {
                    "masks": masks.squeeze(1), # (P=3, 1, H, W) -> (P=3, H, W)
                    "iou_predictions": iou_predictions, # (P=3, 1)
                    "low_res_logits": low_res_masks, # (P=3, 1, 256, 256)
                }
            )
        
        if not self.cfg.dataset.stage1:
            return outputs_0, None, prop_pos_embed

        low_res_pred = (torch.stack(low_res_pred, dim=0) > self.mask_threshold).float() # (B, 1, 3, 256, 256)
        prev_masks_1 = torch.cat([prev_masks_1, low_res_pred], dim=1) # (B, [F-1]=2, P=3, 256, 256)

        prev_masks_1 = prev_masks_1.view(-1, 1, *prev_masks_1.shape[-2:])
        _, mask_embeddings_1 = self.prompt_encoder(points=None, boxes=None, masks=prev_masks_1)
        mask_embeddings_1 = mask_embeddings_1.view(len(batched_input["selector"]), -1, self.max_num_obj, 256, 64, 64) # (B, [F-1]=2, P=3, 256, 64, 64)
        
        # embeddings = {"current_frame_embeddings": current_frame_embeddings, "prev_frames_embeddings": prev_frames_embeddings, "mask_embeddings": mask_embeddings}
        embeddings_1 = {"image_embeddings": image_embeddings, "mask_embeddings": mask_embeddings_1}
        
        all_sparse_embeddings_1, all_dense_embeddings_1, _ = self.propagation_module(
            embeddings_1, pos_embed
        )  # (B, P=3, 64, 64, 256)
        all_dense_embeddings_1 = all_dense_embeddings_1.permute(0, 1, 4, 2, 3) # (B, P=3, 256, 64, 64)

        outputs_1 = []
        for i, (curr_embedding, prop_sparse_embeddings, prop_dense_embeddings) in enumerate(zip(image_embeddings[:, -1], all_sparse_embeddings_1, all_dense_embeddings_1)):
            # curr_embedding: (256, 64, 64) -> current target frame embedding
            # prop_dense_embeddings: (3, 256, 64, 64) -> basically we have 3 prompts
            # prop_sparse_embeddings: (3, 8, 256) -> basically we have 3 prompts, each prompt has 8 points
            
            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=curr_embedding.unsqueeze(0),
                image_pe=pos_embed,
                sparse_prompt_embeddings=prop_sparse_embeddings,
                dense_prompt_embeddings=prop_dense_embeddings,
                multimask_output=multimask_output,
            )
            masks = self.postprocess_masks(
                low_res_masks,
                input_size=list(batched_input["resize_longest_size"][i]),
                original_size=list(batched_input["original_size"][i])
            )
            outputs_1.append(
                {
                    "masks": masks.squeeze(1), # (P=3, 1, H, W) -> (P=3, H, W)
                    "iou_predictions": iou_predictions, # (P=3, 1)
                    "low_res_logits": low_res_masks, # (P=3, 1, 256, 256)
                }
            )
            
        return outputs_0, outputs_1, prop_pos_embed

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