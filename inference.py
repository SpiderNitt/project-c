from PIL import Image
import numpy as np
import os
from pathlib import Path
import torch
import torch.nn.functional as F
from segment_anything.utils.transforms import ResizeLongestSide
import shutil
import glob

class Memory():
    def __init__(self, length) -> None:
        self.embed = []
        self.mask = []
        self.score = []
        self.total_size = length
        self.frames_n = []

    def add(self, image_embed, mask, iou):
        # mask: (P, 256, 256)
        # image_embed: (256, 64, 64)
        if iou > 0.75:
            if len(self.embed) < self.total_size:
                self.embed.append(image_embed)
                self.mask.append(mask)
                self.score.append(iou)
                
            elif min(self.score) < iou:
                idx = self.score.index(min(self.score))
                self.score.pop(idx)
                self.embed.pop(idx)
                self.mask.pop(idx)

                self.embed.append(image_embed)
                self.mask.append(mask)
                self.score.append(iou)
        
    def get_embed(self):
        # image_embed: (F, 256, 64, 64)
        return torch.stack(self.embed, dim=0)
    
    def get_prev_mask(self):
        # (F, P, 256, 256)
        return torch.stack(self.mask, dim=0)

def infer(model, device, video_name, cfg, folder_name, multimask_output=True):
    memory = Memory(length=3)

    pred_dir = Path(os.path.join(cfg.root_dir, 'DAVIS/Predictions', folder_name))
    Path.mkdir(pred_dir / video_name, exist_ok=True, parents=True)

    root_dir = os.path.join(cfg.root_dir, 'DAVIS/Annotations/480p/', video_name)
    vid_len = len(glob.glob1(root_dir, "*.png"))

    first_gt_path = os.path.join(root_dir, '00000.png')
    first_gt, palette, num_obj, original_size, input_size = load_mask(first_gt_path)
    first_embed = torch.load(os.path.join(cfg.root_dir, 'embeddings', video_name, '00000.pth'))

    memory.add(first_embed, first_gt, 1) # Add first frame and mask to memory

    pos_embed = model.prompt_encoder.get_dense_pe()

    for i in range(1, vid_len):
        current_frame_embeddings = torch.load(os.path.join(cfg.root_dir, 'embeddings', video_name, f'{i:05}.pth')).to(device)

        prev_masks = memory.get_prev_mask().to(device) # (F, P, 256, 256)
        prev_masks = prev_masks.view(-1, 1, *prev_masks.shape[-2:])        
        _, mask_embeddings = model.prompt_encoder(points=None, boxes=None, masks=prev_masks)
        mask_embeddings = mask_embeddings.view(1, -1, num_obj, 256, 64, 64) # (1, F, P, 256, 64, 64)

        prev_frames_embeddings = memory.get_embed().to(device) # (F, 256, 64, 64)

        # embeddings = {"current_frame_embeddings": current_frame_embeddings, "prev_frames_embeddings": prev_frames_embeddings, "mask_embeddings": mask_embeddings}
        embeddings = {"current_frame_embeddings": current_frame_embeddings.unsqueeze(0), "prev_frames_embeddings": prev_frames_embeddings.unsqueeze(0), "mask_embeddings": mask_embeddings} # (1, F, 256, 64, 64), (1, F, P, 256, 64, 64)

        all_sparse_embeddings, all_dense_embeddings, _ = model.propagation_module(
            embeddings, pos_embed
        )  # (1, P, 64, 64, 256)
        all_dense_embeddings = all_dense_embeddings.permute(0, 1, 4, 2, 3) # (1, P, 256, 64, 64)
            
        low_res_masks, iou_predictions = model.mask_decoder(
            image_embeddings=current_frame_embeddings.unsqueeze(0),
            image_pe=pos_embed,
            sparse_prompt_embeddings=all_sparse_embeddings[0],
            dense_prompt_embeddings=all_dense_embeddings[0],
            multimask_output=multimask_output,
        ) # (P, C, 256, 256)
        max_iou, max_index = torch.max(iou_predictions, -1)
        batch_indexing = torch.arange(len(max_index), device=max_index.device)

        masks = model.postprocess_masks(
            low_res_masks,
            input_size=(input_size[0]*4, input_size[1]*4), # Since mask was longest resized to 256 and not 1024
            original_size=original_size
        )
        low_res_masks = low_res_masks[batch_indexing, max_index] # (P, 256, 256)
        masks = masks[batch_indexing, max_index] # (P, 256, 256)

        masks = masks.detach().cpu()
        max_, max_pos = torch.max(masks, dim=0)
        masks = ((max_pos+1) * (max_ > 0)).type(torch.int8)

        masks = masks.cpu().numpy().astype(np.uint8)

        masks = Image.fromarray(masks).convert("P")
        masks.putpalette(palette)
        masks.save(pred_dir / video_name / f'{i:05}.png')
        
        memory.add(current_frame_embeddings.cpu(), (low_res_masks > 0).cpu().float(), max_iou.mean().cpu().item())

    shutil.copy(first_gt_path, pred_dir / video_name)

def load_mask(gt_path):
    gt = Image.open(gt_path).convert('P')
    palette = gt.getpalette()
    gt = np.array(gt)
    original_size = gt.shape[-2:]

    resize_longest = ResizeLongestSide(256)

    sep_mask = []

    labels = np.unique(gt)
    labels = labels[labels != 0]

    num_obj = len(labels)
    
    for label in labels:
        if label == 255:
            continue
        mask = np.zeros(gt.shape, dtype=np.uint8)
        mask[gt == label] = 1
        mask = torch.as_tensor(resize_longest.apply_image(mask))
        input_size = mask.shape[-2:]
        sep_mask.append(mask)

    sep_mask = torch.stack(sep_mask).float()

    h, w = sep_mask.shape[-2:]
    padh = 256 - h
    padw = 256 - w
    sep_mask = F.pad(sep_mask, (0, padw, 0, padh))
    return sep_mask, palette, num_obj, original_size, input_size