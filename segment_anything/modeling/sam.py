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
        input_images = torch.stack(
            [self.preprocess(x["image"]) for x in batched_input], dim=0
        )  # Output -> (B, num_frames=3, 3, 1024, 1024)
        
        '''
        To disbale grad for previous frames alone
        prev_frames, current_frame = input_images[:, :-1], input_images[:, -1]
        current_frame_embeddings = self.image_encoder(current_frame)# Output -> (B, 256, 64, 64)
        with torch.no_grad():
            prev_frames_embeddings = self.image_encoder(input_images.reshape(-1, 3, 1024, 1024)).reshape(input_images.shape[0], self.num_frames - 1, 256, 64, 64)
        torch.cuda.empty_cache()
        '''
        # We disable grad since we are not updating the image encoder weights
        with torch.no_grad():
            image_embeddings = self.image_encoder(input_images.reshape(-1, 3, 1024, 1024)).reshape(len(batched_input), self.num_frames, 256, 64, 64)  # Output -> (B, F=3, 256, 64, 64)
        torch.cuda.empty_cache()
        
        prev_masks = torch.stack([x["prev_masks"] for x in batched_input], dim=0) # (B, [F-1]=2, P=3, 256, 256)
        prev_masks = prev_masks.reshape(-1, 1, *prev_masks.shape[-2:])
        _, mask_embeddings= self.prompt_encoder(points=None, boxes=None, masks=prev_masks)
        mask_embeddings = mask_embeddings.reshape(len(batched_input), self.num_frames - 1, self.max_num_obj, 256, 64, 64) # (B, [F-1]=2, P=3, 256, 64, 64)
        
        # embeddings = {"current_frame_embeddings": current_frame_embeddings, "prev_frames_embeddings": prev_frames_embeddings, "mask_embeddings": mask_embeddings}
        embeddings = {"image_embeddings": image_embeddings, "mask_embeddings": mask_embeddings}
        
        all_sparse_embeddings, all_dense_embeddings = self.propagation_module(
            embeddings, self.cfg
        )  # (B, P=3, 64, 64, 256)
        all_dense_embeddings = all_dense_embeddings.permute(0, 1, 4, 2, 3) # (B, P=3, 256, 64, 64)

        outputs = []
        for image_record, curr_embedding, prop_sparse_embeddings, prop_dense_embeddings in zip(batched_input, image_embeddings[:, -1], all_sparse_embeddings, all_dense_embeddings):
            # curr_embedding: (256, 64, 64)
            # prop_dense_embeddings: (3, 256, 64, 64) -> basically we have 3 prompts
            if "point_coords" in image_record:
                points = (image_record["point_coords"], image_record["point_labels"])
            else:
                points = None
            
            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=curr_embedding.unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=prop_sparse_embeddings,
                dense_prompt_embeddings=prop_dense_embeddings,
                multimask_output=multimask_output,
            )
            masks = self.postprocess_masks(
                low_res_masks,
                input_size=image_record["image"].shape[-2:],
                original_size=image_record["original_size"],
            )
            
            outputs.append(
                {
                    "masks": masks, # (P=3, 1, H, W)
                    "iou_predictions": iou_predictions, # (P=3, 1)
                    "low_res_logits": low_res_masks, # (P=3, 1, 256, 256)
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