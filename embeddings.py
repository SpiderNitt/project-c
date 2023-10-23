import torch
from torch.nn import functional as F
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide

from PIL import Image
import numpy as np

import os
import gc
import glob
from pathlib import Path
from omegaconf import OmegaConf
from tqdm.auto import tqdm
from config import cfg

pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)

def generate_embeddings(cfg):

    with open(cfg.dataset.root_dir + 'DAVIS/ImageSets/2017/val.txt', 'r') as f:
        val_list = [line.strip() for line in f]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = sam_model_registry[cfg.model.type](checkpoint=cfg.model.checkpoint, cfg=cfg)

    model.eval()        
    model.to(device)

    root_dir = cfg.dataset.root_dir
    gt_path = root_dir + "DAVIS/Annotations/480p"
    path = root_dir + "DAVIS/JPEGImages/480p"
    dataset = {}

    Path(root_dir + "embeddings").mkdir(exist_ok=True, parents=True)

    for target in sorted(os.listdir(path)):
        dataset[target] = {"image":[], "mask":[]}
        path1 = os.path.join(path, target)
        path2 = os.path.join(gt_path, target)
    
        for frame, gt_frame in zip(sorted(glob.glob1(path1, "*.jpg")), sorted(glob.glob1(path2, "*.png"))):
            dataset[target]["image"].append(os.path.join(path1, frame))
            dataset[target]["mask"].append(os.path.join(path2, gt_frame))

        assert len(dataset[target]["image"]) == len(dataset[target]["mask"])
    
    resize_longest = ResizeLongestSide(1024)

    for video in tqdm(val_list, total=len(val_list)):
        info = {}
        info['name'] = video
        
        frames = dataset[video]
        num_frames = len(frames['image'])
        print("Number of frames: ", num_frames)
        images = []
        
        for image_input in frames['image']:
            this_im = Image.open(image_input).convert('RGB')
            this_im = torch.as_tensor(resize_longest.apply_image(np.array(this_im, dtype=np.uint8))).permute(2, 0, 1)
            this_im = preprocess(this_im)
        
            images.append(this_im)
        
        if video not in os.listdir(root_dir + "embeddings"):
            os.mkdir(root_dir + 'embeddings/' + video)
        
        for idx, img in tqdm(enumerate(images), total=len(images)):
            image_embeddings = None
            each_img = img.to(device)
            with torch.inference_mode():
                image_embeddings = model.image_encoder(each_img.unsqueeze(0))  # Output -> (1, 256, 64, 64)
                image_embeddings = image_embeddings.reshape(256, 64, 64)
                torch.save(image_embeddings.cpu(), os.path.join(root_dir, "embeddings", video, f"{idx:05}.pth"))
            
        del img, image_embeddings
        gc.collect()
        torch.cuda.empty_cache()

def preprocess(x: torch.Tensor) -> torch.Tensor:
    """Normalize pixel values and pad to a square input."""
    # Normalize colors
    x = (x - pixel_mean) / pixel_std
    # Pad
    h, w = x.shape[-2:]
    padh = 1024 - h
    padw = 1024 - w
    x = F.pad(x, (0, padw, 0, padh))
    return x

generate_embeddings(cfg)