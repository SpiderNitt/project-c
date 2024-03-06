import os
import torch
from torch.utils.data.dataset import Dataset
import torch.utils.data as data
from torch.nn import functional as F
from segment_anything.utils.transforms import ResizeLongestSide
from PIL import Image
import numpy as np

def preprocess(x: torch.Tensor) -> torch.Tensor:
    device = 'cpu'
    pixel_mean = torch.as_tensor([123.675, 116.28, 103.53],device=device).view(-1,1,1)
    pixel_std = torch.as_tensor([58.395, 57.12, 57.375],device=device).view(-1,1,1)

    # Normalize colors
    x = (x - pixel_mean) / pixel_std

    # Pad
    h, w = x.shape[-2:]
    padh = 1024 - h
    padw = 1024 - w
    x = F.pad(x, (0, padw, 0, padh))
    return x

def load_mask(gt_path):
    gt = Image.open(gt_path).convert('P')
    palette = gt.getpalette()
    gt = np.array(gt)//255
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

class MOCA(Dataset):
    def __init__(self):
        directory = 'raw/MoCA-Mask/MoCA_Video/TestDataset_per_sq'
        self.dataset = {}

        for i in sorted(os.listdir(directory)):
            path = os.path.join(directory,i+'/Imgs')
            gt_path =os.path.join(directory,i+'/GT')
            self.dataset[i] = {"image":[], "mask":[], "frame_num": []}
            for target in sorted(os.listdir(path)):
                path1 = os.path.join(path, target)
                path2 = os.path.join(gt_path, target[:-4]+".png")
                self.dataset[i]["image"].append(path1)
                self.dataset[i]["mask"].append(path2)
                self.dataset[i]["frame_num"].append(target[:-4])
        self.keys = list(self.dataset.keys())
        self.resize_longest = ResizeLongestSide(1024)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image = self.dataset[self.keys[idx]]["image"]
        mask = self.dataset[self.keys[idx]]["mask"]
        frame_num = self.dataset[self.keys[idx]]["frame_num"]
        img_list = []

        for i in range(len(image)):
            this_im = Image.open(image[i]).convert('RGB')
            this_im = torch.as_tensor(self.resize_longest.apply_image(np.array(this_im, dtype=np.uint8))).permute(2, 0, 1)
            this_im = preprocess(this_im)
            img_list.append(this_im)
        first_gt, palette, num_obj, original_size, input_size = load_mask(mask[0])
        
        return {
            'image': img_list, # (num_frames, 3, 1024, 1024) 
            'first_gt': first_gt, # (1, 256, 256)
            'frame_num': frame_num,
            'original_size': list(original_size),
            'resize_longest_size': [input_size[0]*4, input_size[1]*4],
            'palette': palette,
            'num_obj': num_obj,
            'info': self.keys[idx],
            'first_gt_path': mask[0]
        }

def get_test_loader():
    test_dataset = MOCA()
    test_loader = data.DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=False,
        persistent_workers=True,
        pin_memory=True,
    )
    return test_loader
# print(MOCA()[0])
MOCA()[0]
