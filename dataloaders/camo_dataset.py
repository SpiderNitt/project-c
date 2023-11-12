import os
from PIL import Image, ImageEnhance
import torch
import cv2
import skimage.exposure
import numpy as np
from numpy.random import default_rng
import torchvision
import torch.utils.data as data
import torchvision.transforms as transforms
import random
import numpy as np
from glob import glob
import os.path as osp
from pathlib import Path
from segment_anything.utils.transforms import ResizeLongestSide
from torchvision.transforms.functional import resize
from torchvision.transforms import InterpolationMode
from dataloaders.tps import random_tps_warp
from dataloaders.reseed import reseed
from torchvision.transforms.functional import resize, pil_to_tensor
from dataloaders.range_transform import im_normalization, im_mean
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time

def iou(mask1, mask2):
    # print("___", np.unique(mask1), np.unique(mask2))
    intersection = (mask1 * mask2).sum()
    if intersection == 0:
        return 0.0
    
    union = (mask1 + mask2 > 0).sum()
    
    return float(intersection) / union

# several data augumentation strategies
def cv_random_flip(imgs, label):
    # left right flip
    flip_flag = random.randint(0, 1)
    if flip_flag == 1:
        imgs = imgs.transpose(Image.FLIP_LEFT_RIGHT)
        for i in range(len(label)):
            label[i] = label[i].transpose(Image.FLIP_LEFT_RIGHT)
    return imgs, label


def randomCrop(imgs, label):
    border = 30
    image_width = imgs.size[0]
    image_height = imgs.size[1]
    crop_win_width = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)
    random_region = (
        (image_width - crop_win_width) >> 1,
        (image_height - crop_win_height) >> 1,
        (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1,
    )

    imgs = imgs.crop(random_region)
    for i in range(len(label)):
        
        label[i] = label[i].crop(random_region)
    return imgs, label


def randomRotation(imgs, label):
    mode = Image.BICUBIC
    if random.random() > 0.8:
        random_angle = np.random.randint(-10, 10)
        imgs = imgs.rotate(random_angle, mode)
        for i in range(len(label)):
            label[i] = label[i].rotate(random_angle, mode)
    return imgs, label


def colorEnhance(imgs):
    # for i in range(len(imgs)):
    bright_intensity = random.randint(75, 125) / 100 # +- 25%
    imgs = ImageEnhance.Brightness(imgs).enhance(bright_intensity)
    contrast_intensity = random.randint(75, 125) / 100 # +- 25%
    imgs = ImageEnhance.Contrast(imgs).enhance(contrast_intensity)
    color_intensity = random.randint(5, 15) / 10.0 
    imgs = ImageEnhance.Color(imgs).enhance(color_intensity)
    sharp_intensity = random.randint(0, 30) / 10.0
    imgs = ImageEnhance.Sharpness(imgs).enhance(sharp_intensity)
        
        # print(bright_intensity, contrast_intensity, color_intensity, sharp_intensity)
    return imgs

def randomPeper(img_list):
    output_list = []
    for img in img_list:
        img = np.array(img)
        noiseNum = int(0.0015 * img.shape[0] * img.shape[1])

        for i in range(noiseNum):
            randX = random.randint(0, img.shape[0] - 1)
            randY = random.randint(0, img.shape[1] - 1)

            if random.randint(0, 1) == 0:
                img[randX, randY] = 0
            else:
                img[randX, randY] = 255
        output_list.append(Image.fromarray(img))
    return output_list




class VideoDataset(data.Dataset):
    def __init__(
        self, cfg, val
    ):
        self.dataset_root = cfg.root_dir
        self.val = val
        self.cfg = cfg
        self.pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
        self.pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
        self.image_list = []
        self.gt_list = []
        self.num_frames = cfg.num_frames
        self.max_num_obj = cfg.max_num_obj

        
        ####### CAAD10K-V3
        
        path = os.path.join(self.dataset_root, "COD10K-v3") 
        
        split_path = os.path.join(path, "Test")
        info = open(split_path+"/CAM-NonCAM_Instance_Test.txt", 'r').readlines()
        
        if not self.val:
            split_path = os.path.join(path, "Train")
            info = open(split_path+"/CAM-NonCAM_Instance_Train.txt", 'r').readlines()

        img_path = os.path.join(split_path, "Image")
        gt_path = os.path.join(split_path, "GT_Instance")

        
        for i in info:
            if "[INFO]" in i:
                i = i.lstrip().rstrip()
                if int(i[-1]) >= 1:
                    name = i.split(" ")[-2].split(".")[0]
                    self.image_list.append(os.path.join(img_path, name+'.jpg'))
                    self.gt_list.append(os.path.join(gt_path, name+'.png'))
                  
        count = len(self.image_list)
        print(f'{"Train" if not self.val else "Test"} Loaded COD10K', len(self.image_list))
        count = len(self.image_list)
        assert len(self.image_list) == len(self.gt_list)
          

        ####CAMO
        path = os.path.join(self.dataset_root, "CAMO/CAMO-COCO-V.1.0/CAMO-COCO-V.1.0-CVIU2019/Camouflage")
        img_path = os.path.join(path, "Images/Test")
        
        if not self.val:
            img_path = os.path.join(path, "Images/Train")

        gt_path = path+"/GT"

        for i in sorted(os.listdir(img_path)):
            self.image_list.append(os.path.join(img_path, i))
            self.gt_list.append(os.path.join(gt_path, i.split(".")[0]+".png"))
            
        print(f'{"Train" if not self.val else "Test"} Loaded CAMO', len(self.image_list) - count)
        count = len(self.image_list)
        assert len(self.image_list) == len(self.gt_list)
        
        print(f'===={"Train" if not self.val else "Test"} Loaded {count} images')

        # self.pair_im_lone_transform = transforms.Compose([
        #     # transforms.ColorJitter(0.1, 0.05, 0.05, 0), # No hue change here as that's not realistic
        # ])


        self.pair_im_dual_transform = transforms.Compose([
            transforms.RandomAffine(degrees=None if val else 20, shear=None if val else 20, translate=None if val else (0.2,0.4), scale=None if val else (0.85, 1.15), interpolation=InterpolationMode.BICUBIC, fill=im_mean),
            # transforms.Resize(384, InterpolationMode.BICUBIC),
            # transforms.RandomCrop((384, 384), pad_if_needed=True, fill=im_mean),
        ])

        self.pair_gt_dual_transform = transforms.Compose([
            transforms.RandomAffine(degrees=None if val else 20, shear=None if val else 20, translate=None if val else (0.2,0.4), scale=None if val else (0.85, 1.15), interpolation=InterpolationMode.BICUBIC, fill=0),
            # transforms.Resize(384, InterpolationMode.NEAREST),
            # transforms.RandomCrop((384, 384), pad_if_needed=True, fill=0),
        ])


        # These transform are the same for all pairs in the sampled sequence
        self.all_im_lone_transform = transforms.Compose([
            transforms.ColorJitter((0.8,1.2), (0.8,1.2), (0.8,1.2), 0.15),
            transforms.RandomGrayscale(0.5),
        ])

        self.all_im_dual_transform = transforms.Compose([
            transforms.RandomAffine(degrees=0, scale=(0.8, 1.5), fill=im_mean),
            transforms.RandomHorizontalFlip(0.5),
        ])

        self.all_gt_dual_transform = transforms.Compose([
            transforms.RandomAffine(degrees=0, scale=(0.8, 1.5), fill=0),
            transforms.RandomHorizontalFlip(0.5),
        ])

        self.resize_longest = ResizeLongestSide(1024)
        self.resize_longest_mask = ResizeLongestSide(256)


    def preprocess_prev_masks(self, x):
        h, w = x.shape[-2:]
        padh = 256 - h
        padw = 256 - w
        x = F.pad(x, (0, padw, 0, padh))
        return x
    
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = 1024 - h
        padw = 1024 - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

 
    def __getitem__(self, idx):
        info = {}
        info['name'] = self.image_list[idx].split('/')[-1]
        info['frames'] = [0]*self.num_frames
        im = Image.open(self.image_list[idx]).convert('RGB')
        gt = Image.open(self.gt_list[idx]).convert('L')
        # plt.imshow(np.asarray(gt))
        # plt.show()
        sequence_seed = np.random.randint(2147483647)

        cropped_img = []
        images = []
        masks = []
        ids = []
        reseed(sequence_seed)
        this_im = self.all_im_dual_transform(im)
        # print("===")
        # print(np.mean(np.asarray(this_im)))
        # plt.imshow(np.asarray(this_im))
        # plt.show()
        this_im_ = self.all_im_lone_transform(this_im)
        # print("???")
        # print(np.mean(np.asarray(this_im_)))
        # plt.imshow(np.asarray(this_im_))
        # plt.show()    
        
        reseed(sequence_seed)
        this_gt_ = self.all_gt_dual_transform(gt)
        
        for _ in range(self.num_frames):
            
            pairwise_seed = np.random.randint(2147483647)
            reseed(pairwise_seed)
            this_im = self.pair_im_dual_transform(this_im_)
            # this_im = self.pair_im_lone_transform(this_im)
            
            this_gt = np.asarray(this_gt_)
            ids = np.unique(this_gt)
            if 0 in ids:
                ids.remove(0)
            
            each_gt = None
            for each_sep_obj in ids:
                # print(each_sep_obj)
                this = Image.fromarray((this_gt==each_sep_obj).astype(np.uint8))
                reseed(pairwise_seed)
                this = self.pair_gt_dual_transform(this)
                
                this = np.asarray(this).copy()
                this[this!=0] = each_sep_obj
                
                if each_gt is None:
                    each_gt = this
                each_gt += this
            
            
            cropped_img.append(pil_to_tensor(this_im))
            this_im = torch.as_tensor(self.resize_longest.apply_image(np.array(this_im, dtype=np.uint8))).permute(2, 0, 1)
            resize_longest_size = this_im.shape[-2:]
            this_im = self.preprocess(this_im)
            images.append(this_im)
            
            if each_gt is None:
                return self.__getitem__(np.random.randint(self.__len__()))
            masks.append(each_gt)

        images = torch.stack(images, 0)
        masks = np.stack(masks, 0)

        target_objects = np.unique(masks[0]).tolist()
        if 0 in target_objects:
            target_objects.remove(0)
        if len(target_objects) > self.max_num_obj:
            target_objects = np.random.choice(target_objects, size=self.max_num_obj, replace=False)
        
        info['num_objects'] = max(1, len(target_objects))

        H, W = tuple(masks.shape[1:])
        # Generate one-hot ground-truth
        cls_gt = np.zeros((self.num_frames, H, W), dtype=np.int64)
        all_frame_gt = np.zeros((self.num_frames, self.max_num_obj, H, W), dtype=np.int64) # Shape explains itself
        for t in range(self.num_frames):
            for i, l in enumerate(target_objects):
                this_mask = (masks==l)
                cls_gt[this_mask] = i+1
                all_frame_gt[t,i] = (this_mask[t])

        cls_gt = np.expand_dims(cls_gt, 1)
        all_frame_gt_256 = all_frame_gt.reshape(-1, H, W)
        
        new_all_frame_gt = []
        for t in range(len(all_frame_gt_256)):
            new_all_frame_gt.append(torch.as_tensor(self.resize_longest_mask.apply_image(all_frame_gt_256[t].astype(dtype=np.uint8))))

        new_all_frame_gt = torch.stack(new_all_frame_gt, 0).reshape(-1, self.max_num_obj, *new_all_frame_gt[0].shape[-2:])
        new_all_frame_gt = self.preprocess_prev_masks(new_all_frame_gt).float()

        new_prev_frame_gt = new_all_frame_gt[:-1]

        all_frame_gt = torch.as_tensor(all_frame_gt).float()

        # 1 if object exist, 0 otherwise
        selector = [1 if i < info['num_objects'] else 0 for i in range(self.max_num_obj)]
        selector = torch.BoolTensor(selector)

        data = {
            'image': images, # (num_frames, 3, 1024, 1024) 
            'gt_mask': all_frame_gt, # (num_frames, num_obj=3, H, W)
            'gt_mask_256': new_all_frame_gt, # (num_frames, num_obj=3, 256, 256)
            'prev_masks': new_prev_frame_gt, # (num_frames, num_obj=3, 256, 256)
            'selector': selector, # (num_obj=3) Indicates if ith object exists
            'cropped_img': cropped_img, # (num_frames, 3, H, W)
            'original_size': list(all_frame_gt.shape[-2:]),
            'resize_longest_size': list(resize_longest_size),
            'info': info
        }
        return data

 

    def rgb_loader(self, path):
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")

    def binary_loader(self, path):
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("L")

    def __len__(self):
        return len(self.image_list)

def collate_fn(batch):
    output = {}
    for key in batch[0].keys():
        output[key] = [d[key] for d in batch]
        if key in ["image", "prev_masks", "selector"]:
            output[key] = torch.stack(output[key], 0)
    
    return output


# dataloader for training
def get_loader(
    cfg,
):
    train_dataset = VideoDataset(cfg, val=False)
    train_data_loader = data.DataLoader(
            dataset=train_dataset,
            batch_size=cfg.train_batch_size,
            shuffle=True,
            persistent_workers=cfg.persistent_workers,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            collate_fn=collate_fn
    )
    
    val_dataset = VideoDataset(cfg, val=True)
    val_data_loader = data.DataLoader(
            dataset=val_dataset,
            batch_size=cfg.val_batch_size,
            shuffle=False,
            persistent_workers=cfg.persistent_workers,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            collate_fn=collate_fn
    )
    

    return train_data_loader, val_data_loader