import os

import cv2
import numpy as np
import torch
from torch.nn import functional as F
import torchvision.transforms as transforms
from segment_anything.utils.transforms import ResizeLongestSide
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

class MoCA(Dataset):
    def __init__(self, cfg, train=True):
        if train:
            self.path = cfg.dataset.train.root_dir
        else:
            self.path = cfg.dataset.val.root_dir

        self.obj_path = {}
        self.obj_arr = []
        self.input_seq = cfg.seq_len
        self.total_imgs = 0
        self.obj_len = {}
        self.img_size = cfg.img_size
        self.transform = ResizeLongestSide(cfg.img_size)

        for vid in os.listdir(self.path):
            self.obj_path[vid] = {'Imgs':[], 'GT':[]}
            self.obj_arr.append(vid)
            self.obj_len[vid] = 0
            temp_path = os.path.join(self.path, vid)

            img_list = sorted(os.listdir(os.path.join(temp_path, "GT")))
            self.obj_len[vid] = len(img_list) - self.input_seq  
            self.total_imgs += self.obj_len[vid]

            for type_ in ['Imgs', 'GT']:
                img_list = sorted(os.listdir(os.path.join(temp_path, type_)))
                for frame in img_list:
                    self.obj_path[vid][type_].append(os.path.join(temp_path, type_, frame))


    def get_object_from_id(self, id_):
        n = len(self.obj_arr)
        if (id_ ==0): return self.obj_arr[n-1], 0
        while id_>0:
            n -= 1
            assert n>=0 ,"Out of Index in DataSet"
            id_ -= self.obj_len[self.obj_arr[n]]

        return self.obj_arr[n], self.obj_len[self.obj_arr[n]]+id_

    def __len__(self):
        return self.total_imgs

    def __getitem__(self, idx):
        obj, nth_frame = self.get_object_from_id(idx)
        img_seq = []
        mask_frame_original_size = None

        for next_frame in range(nth_frame, nth_frame+self.input_seq):
            input_image = cv2.imread(self.obj_path[obj]['Imgs'][next_frame])
            input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
            input_image = self.transform.apply_image(input_image)
            input_image = torch.as_tensor(input_image)
            input_image = input_image.permute(2, 0, 1).contiguous()[None, :, :, :]

            if next_frame == nth_frame+self.input_seq-1:
                mask = cv2.imread(self.obj_path[obj]['GT'][next_frame])
                mask_frame_original_size = mask.shape[:2]
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                mask = torch.as_tensor(mask)

            img_seq.append(input_image)

        # image -> (3, H, W), gt_masks -> (1, 1, H, W), original_size -> (H, W)
        return {"image": img_seq[0][0], "gt_masks": mask.unsqueeze(0).unsqueeze(0).float(), "original_size": mask_frame_original_size}

def collate_fn(batch):
    return batch