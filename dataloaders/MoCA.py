import os
from PIL import Image, ImageEnhance
import torch
import cv2
import skimage.exposure
import numpy as np
from numpy.random import default_rng

import torch.utils.data as data
import torchvision.transforms as transforms
import random
import numpy as np
from glob import glob
import os.path as osp
from pathlib import Path
from segment_anything.utils.transforms import ResizeLongestSide
from torchvision.transforms.functional import resize
import torch.nn.functional as F
import matplotlib.pyplot as plt

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
        for i in range(len(imgs)):
            imgs[i] = imgs[i].transpose(Image.FLIP_LEFT_RIGHT)
            label[i] = label[i].transpose(Image.FLIP_LEFT_RIGHT)
    return imgs, label


def randomCrop(imgs, label):
    border = 30
    image_width = imgs[0].size[0]
    image_height = imgs[0].size[1]
    crop_win_width = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)
    random_region = (
        (image_width - crop_win_width) >> 1,
        (image_height - crop_win_height) >> 1,
        (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1,
    )

    for i in range(len(imgs)):
        imgs[i] = imgs[i].crop(random_region)
        label[i] = label[i].crop(random_region)
    return imgs, label


def randomRotation(imgs, label):
    mode = Image.BICUBIC
    if random.random() > 0.8:
        random_angle = np.random.randint(-10, 10)
        for i in range(len(imgs)):
            imgs[i] = imgs[i].rotate(random_angle, mode)
            label[i] = label[i].rotate(random_angle, mode)
    return imgs, label


def colorEnhance(imgs):
    for i in range(len(imgs)):
        bright_intensity = random.randint(75, 125) / 100 # +- 25%
        imgs[i] = ImageEnhance.Brightness(imgs[i]).enhance(bright_intensity)
        contrast_intensity = random.randint(75, 125) / 100 # +- 25%
        imgs[i] = ImageEnhance.Contrast(imgs[i]).enhance(contrast_intensity)
        color_intensity = random.randint(5, 15) / 10.0 
        imgs[i] = ImageEnhance.Color(imgs[i]).enhance(color_intensity)
        sharp_intensity = random.randint(0, 30) / 10.0
        imgs[i] = ImageEnhance.Sharpness(imgs[i]).enhance(sharp_intensity)
        
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
        self,
        dataset="MoCA-Mask/MoCA_Video",
        seq_len=4,
        trainsize=1024,
        split="TrainDataset_per_sq",
        iou_range = [0.05, 0.4], # 5 to 40% overlap
        max_prompt_points = 3,
        finetune = False
    ):
        self.seq_len = seq_len
        self.trainsize = trainsize
        self.transform = ResizeLongestSide(trainsize)
        self.image_list = []
        self.gt_list = []
        self.extra_info = []
        self.iou_range = iou_range
        self.max_prompt_points = max_prompt_points
        self.finetune = finetune
        
        seedval = 75
        self.rng = default_rng(seed=seedval)
        
        root = Path(dataset)
        img_format = "*.jpg"
        data_root = osp.join(root, split)

        for scene in os.listdir(osp.join(data_root)):
            if split == "MoCA-Video-Train":
                images = sorted(glob(osp.join(data_root, scene, "Frame", img_format)))
            elif split == "TrainDataset_per_sq":
                images = sorted(glob(osp.join(data_root, scene, "Imgs", img_format)))
            elif split == "TestDataset_per_sq":
                images = sorted(glob(osp.join(data_root, scene, "Imgs", img_format)))
            gt_list = sorted(glob(osp.join(data_root, scene, "GT", "*.png")))
            # pdb.set_trace()

            self.gt_list+=gt_list
            self.image_list+=images

        self.gt_transform = transforms.Compose([transforms.ToTensor()])
        self.uniform_point_prompt = np.random.uniform(0, 1, (self.__len__(),)).tolist()
        random.shuffle(self.uniform_point_prompt)
        self.n_points = 0

    def sample_point_uniform(self):
        if len(self.uniform_point_prompt) == 0:
            self.uniform_point_prompt = np.random.uniform(0, 1, (self.__len__(),)).tolist()
            random.shuffle(self.uniform_point_prompt)
            
        return self.uniform_point_prompt.pop()
        

    def __getitem__(self, index):
        imgs = []
        names = []
        all_gt = []
        index = index % len(self.image_list)

        imgs = [self.rgb_loader(self.image_list[index])]
        all_gt = [self.binary_loader(self.gt_list[index])]

        if self.finetune:
            imgs, all_gt = cv_random_flip(imgs, all_gt)
            # imgs, all_gt = randomCrop(imgs, all_gt) # DO WE NEED RANDOM CROP ?????????
            imgs, all_gt = randomRotation(imgs, all_gt)
            imgs = colorEnhance(imgs)
            # all_gt = randomPeper(all_gt) # SERIOUSLY WHY DO WE NEED THIS ?????????????
        
        imgs = torch.as_tensor(
                self.transform.apply_image(np.array(imgs[0], dtype=np.uint8))
            ).permute(2, 0, 1)
            
        gt_mask = self.gt_transform(all_gt[0])[0]

        ### SAMPLING PROMPTS
        
        point_prompt = None
        label_prompt = None
        mask_prompt = None
        


        if random.choice([True, False]): # mask prompt
            matching_iou = False
            img = (gt_mask.numpy()*255).astype(np.uint8)
            
            retrying = 0
            while not matching_iou:
                retrying+=1
                if retrying>100:
                    print(f"Skipped mask prompt for {self.image_list[index]} after 100 attempts")
                    matching_iou = True
            
                height, width = img.shape[:2]

                # create random noise image
                noise = self.rng.integers(0, 255, (height,width), np.uint8, True)

                # blur the noise image to control the size
                blur = cv2.GaussianBlur(noise, (0,0), sigmaX=30, sigmaY=30, borderType = cv2.BORDER_DEFAULT)

                # stretch the blurred image to full dynamic range
                stretch = skimage.exposure.rescale_intensity(blur, in_range='image', out_range=(0,255)).astype(np.uint8)

                # threshold stretched image to control the size
                thresh = cv2.threshold(stretch, 130, 255, cv2.THRESH_BINARY)[1]

                # apply morphology open and close to smooth out and make 3 channels
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
                mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                # add mask to input
                result1 = cv2.bitwise_and(img, mask)

                _, labels, stats, _ = cv2.connectedComponentsWithStats(result1)
                
                patch_ids = np.unique(labels).tolist()
                patch_ids.remove(0)
                random.shuffle(patch_ids)
                
                for diff_patch in patch_ids:
                    prompt = labels.copy()
                    prompt[prompt!=diff_patch] = 0
                    prompt[prompt==diff_patch] = 1
                    
                    iou_score = iou((img/255).astype(np.uint8), prompt)
                    # print("IOU SCORE:", iou_score)
                    if (iou_score >= self.iou_range[0]) and (iou_score <= self.iou_range[1]):
                        matching_iou = True
                        mask_prompt = prompt.copy()
                        break

        if random.choice([True, False]) or (mask_prompt is None): #point prompt
            possible_points = gt_mask.numpy().nonzero()
            if len(possible_points) != 0:
                self.n_points+=1
                self.n_points = (self.n_points%self.max_prompt_points) + 1 # uniform distribution
                n_prompt_points = self.n_points

                point_prompt = []
                for _ in range(n_prompt_points):
                    point_prompt_idx = int(self.sample_point_uniform()*len(possible_points[0])) % len(possible_points[0])
                    point_prompt.append([possible_points[1][point_prompt_idx], possible_points[0][point_prompt_idx]])

                point_prompt = np.asarray(point_prompt)
                label_prompt = np.asarray([1]*n_prompt_points)
            
        if (point_prompt is None) and (mask_prompt is None):
            return self.__getitem__(index+1)
        
        return {
            "image": imgs, # Tensor(3, H (transformed), 1024)
            "gt_mask": gt_mask/255.0, # Tensor(H, W)
            "original_size": gt_mask.shape, # List (H,W) - [720, 1280]
            "point_coords": point_prompt, # numpy array (N,2) - 
            "point_labels": label_prompt, # numpy array (N,)
            "mask_input": mask_prompt # numpy array (H, W) -> ideally requires 256,256 which is done inside forward pass for logging purposes
        }  # image -> list((3, H', W')), gt_mask -> Tensor(H, W), original_size: (H, W)

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
    return batch


# dataloader for training
def get_loader(
    dataset="MoCA-Mask/MoCA_Video",
    batchsize=1,
    seq_len=4,
    iou_range = [0.05, 0.4],
    max_prompt_points = 3,
    trainsize=1024,
    train_split="TrainDataset_per_sq",
    validation_split="TestDataset_per_sq",
    num_workers=0,
    pin_memory=True,
    collate_fn=collate_fn,
):
    train_dataset = VideoDataset(dataset, seq_len, trainsize, split=train_split, iou_range=iou_range, max_prompt_points=max_prompt_points, finetune=True)
    train_data_loader = data.DataLoader(
        dataset=train_dataset,
        batch_size=batchsize,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
    
    val_dataset = VideoDataset(dataset, seq_len, trainsize, split=validation_split, iou_range=iou_range, max_prompt_points=max_prompt_points, finetune=True)
    val_data_loader = data.DataLoader(
        dataset=val_dataset,
        batch_size=batchsize,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
    

    return train_data_loader, val_data_loader
