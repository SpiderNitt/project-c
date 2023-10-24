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
        self,
        dataset_root="../raw",
        seq_len=4,
        trainsize=1024,
        iou_range = [0.05, 0.4], # 5 to 40% overlap
        max_prompt_points = 3,
        is_validation_prompt_mid = True,
        max_num_objects = 3,
        finetune = False
    ):
        self.seq_len = seq_len
        self.trainsize = trainsize
        self.transform = ResizeLongestSide(trainsize)
        self.resize_gt = ResizeLongestSide(256)
        self.image_list = []
        self.gt_list = []
        self.extra_info = []
        self.iou_range = iou_range
        self.max_prompt_points = max_prompt_points
        self.finetune = finetune
        self.is_validation_prompt_mid = is_validation_prompt_mid
        self.max_num_objects = max_num_objects
        
        seedval = 75
        self.rng = default_rng(seed=seedval)
        self.uniform_point_prompt = np.random.uniform(0, 1, (self.__len__(),)).tolist()
        random.shuffle(self.uniform_point_prompt)
        self.n_points = 0
        
        self.dataset_root = dataset_root
        self.gt_transform = transforms.Compose([transforms.ToTensor()])
        
        # ####### MoCA
        # dataset = osp.join(dataset_root, "MoCA-Mask/MoCA_Video") 
        # root = Path(dataset)
        # img_format = "*.jpg"
        
        # split = "TestDataset_per_sq"
        # if finetune:
        #     split = "TrainDataset_per_sq"
            
        # data_root = osp.join(root, split)
        
        
        # for scene in os.listdir(osp.join(data_root)):
        #     if split == "MoCA-Video-Train":
        #         images = sorted(glob(osp.join(data_root, scene, "Frame", img_format)))
        #     elif split == "TrainDataset_per_sq":
        #         images = sorted(glob(osp.join(data_root, scene, "Imgs", img_format)))
        #     elif split == "TestDataset_per_sq":
        #         images = sorted(glob(osp.join(data_root, scene, "Imgs", img_format)))
        #     gt_list = sorted(glob(osp.join(data_root, scene, "GT", "*.png")))
        #     # pdb.set_trace()

        #     self.gt_list+=gt_list
        #     self.image_list+=images

        # count = len(self.image_list)
        # assert len(self.image_list) == len(self.gt_list)
        # print(f'{"Train" if finetune else "Test"} Loaded MoCA', count)
        
        
        ####### CAAD10K-V3
        
        path = os.path.join(self.dataset_root, "COD10K-v3") 
        
        split_path = os.path.join(path, "Test")
        info = open(split_path+"/CAM-NonCAM_Instance_Test.txt", 'r').readlines()
        
        if finetune:
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
        print(f'{"Train" if finetune else "Test"} Loaded COD10K', len(self.image_list))
        count = len(self.image_list)
        assert len(self.image_list) == len(self.gt_list)
          
          
        ###NC4K
        # path = os.path.join(self.dataset_root, "nc4k")
        # split_path = os.path.join(path, "fix_rank_test_dataset")
        
        # if finetune:
        #     split_path = os.path.join(path, "fix_rank_dataset2000")
        
        # img_path = os.path.join(split_path, "img")
        # gt_path = os.path.join(split_path, "instance")


        # for i in sorted(os.listdir(img_path)):
        #     self.image_list.append(os.path.join(img_path, i))
        #     self.gt_list.append(os.path.join(gt_path, i.split(".")[0]+".png"))        

        # print(f'{"Train" if finetune else "Test"} Loaded NC4K', len(self.image_list) - count)
        # count = len(self.image_list)
        # assert len(self.image_list) == len(self.gt_list)
        
        
        ####CAMO
        path = os.path.join(self.dataset_root, "CAMO/CAMO-COCO-V.1.0/CAMO-COCO-V.1.0-CVIU2019/Camouflage")
        img_path = os.path.join(path, "Images/Test")
        
        if finetune:
            img_path = os.path.join(path, "Images/Train")

        gt_path = path+"/GT"

        for i in sorted(os.listdir(img_path)):
            self.image_list.append(os.path.join(img_path, i))
            self.gt_list.append(os.path.join(gt_path, i.split(".")[0]+".png"))
            
        print(f'{"Train" if finetune else "Test"} Loaded CAMO', len(self.image_list) - count)
        count = len(self.image_list)
        assert len(self.image_list) == len(self.gt_list)
        
        print(f'===={"Train" if finetune else "Test"} Loaded {count} images')

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Pad
        h, w = x.shape[-2:]
        padh = 256 - h
        padw = 256 - w
        x = F.pad(x, (0, padw, 0, padh))
        # x = torchvision.transforms.functional.resize(x.unsqueeze(0), (256, 256), interpolation=torchvision.transforms.InterpolationMode.BILINEAR, antialias=True) 
        return x
    
    
    def sample_point_uniform(self):
        if len(self.uniform_point_prompt) == 0:
            self.uniform_point_prompt = np.random.uniform(0, 1, (self.__len__(),)).tolist()
            random.shuffle(self.uniform_point_prompt)
            
        return self.uniform_point_prompt.pop()
        
    def get_midpoint(self,a_mask):
        ''' Get the bounding box of a given mask '''
        pos = np.where(a_mask)
        xmin = np.min(pos[1])
        xmax = np.max(pos[1])
        ymin = np.min(pos[0])
        ymax = np.max(pos[0])
        return [(ymax+ymin)//2, (xmin+xmax)//2, ]
    
    def NClosest(self, all_points, target, N):
    
        all_points.sort(key = lambda K: (K[0]-target[0])**2 + (K[1]-target[1])**2)
    
        return all_points[:N]
 
 
    def __getitem__(self, index):

        imgs = []
        names = []
        all_gt = []
        info = {}
        index = index % len(self.image_list)
        info['name'] = self.image_list[index]
        imgs = self.rgb_loader(self.image_list[index])
        gt = np.asarray(self.binary_loader(self.gt_list[index]))
    
        for i in list(np.unique(gt)):
            if i != 0:
                all_gt.append(Image.fromarray((gt==i).astype(np.uint8)))

        all_gt = all_gt[:self.max_num_objects]     ##### max_num_objects    
        
        if self.finetune:
            imgs, all_gt = cv_random_flip(imgs, all_gt)
            # imgs, all_gt = randomCrop(imgs, all_gt) # DO WE NEED RANDOM CROP ?????????
            imgs, all_gt = randomRotation(imgs, all_gt)
            imgs = colorEnhance(imgs)
            # all_gt = randomPeper(all_gt) # SERIOUSLY WHY DO WE NEED THIS ?????????????

        img_copy = np.asarray(imgs).copy() 
        imgs = torch.as_tensor(
                self.transform.apply_image(np.array(imgs, dtype=np.uint8))
            ).permute(2, 0, 1)
            
        
        all_gt_copy = all_gt.copy()
        
        for i in range(len(all_gt_copy)):
            # print(all_gt_copy)
            all_gt_copy[i] = self.gt_transform(all_gt_copy[i])[0]*255
            # print(np.unique(all_gt_copy[i]))
            # print(f'all_gt__shape {self.preprocess(torch.as_tensor(self.transform.apply_image(np.asarray(all_gt[i], dtype=np.uint8)))).shape}') #1,256,256
            # all_gt[i] = Image.fromarray(self.preprocess(torch.as_tensor(self.transform.apply_image(np.asarray(all_gt[i], dtype=np.uint8)))).numpy().astype(np.uint8)*255).resize((256,256))
            # print(all_gt[i].shape)
            # print(f'all_gt_{np.unique(all_gt_[i])}')
            # all_gt[i] = self.gt_transform(all_gt[i])[0]        

            all_gt[i] = self.preprocess(torch.as_tensor(self.resize_gt.apply_image(np.asarray(all_gt[i],dtype=np.uint8))))
            # print(all_gt[i].shape)
            # print(np.unique(all_gt[i]))
        all_point_prompt = None
        all_label_prompt = None
        all_mask_prompt = None
        all_point_prompt_original = None


        is_mask_prompt = random.choice([True, False])
        is_point_prompt = random.choice([True, False]) if is_mask_prompt else True #atleast one of them
        
        if is_point_prompt:
            self.n_points+=1
            self.n_points = (self.n_points%self.max_prompt_points) + 1 # uniform distribution
            n_prompt_points = self.n_points
            all_point_prompt = []
            all_label_prompt = []
            all_point_prompt_original = []
            
        if is_mask_prompt:
            all_mask_prompt = []
        
        for gt_mask, gt_mask_ in zip(all_gt, all_gt_copy):     #### for each object/instance in the image   

            ### SAMPLING PROMPTS

            point_prompt = None
            label_prompt = None
            mask_prompt = None

            
            if is_mask_prompt: # mask prompt
                matching_iou = False
                img = (gt_mask.numpy()*255).astype(np.uint8)

                retrying = 0
                while not matching_iou:
                    retrying+=1
                    if retrying>500:
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
                            all_mask_prompt.append(torch.as_tensor(mask_prompt, dtype=torch.float))
                            break

            if is_point_prompt: #point prompt
                possible_points = list(gt_mask_.numpy().nonzero())
                if len(possible_points[0]) != 0:
                    point_prompt = []
                    if not self.is_validation_prompt_mid:
                        for _ in range(n_prompt_points):
                            point_prompt_idx = int(self.sample_point_uniform()*len(possible_points[0])) % len(possible_points[0])
                            point_prompt.append([possible_points[1][point_prompt_idx], possible_points[0][point_prompt_idx]])
                        point_prompt = np.asarray(point_prompt)  
                        label_prompt = np.asarray([1]*n_prompt_points)
                    else:
                        mid = self.get_midpoint(gt_mask_.numpy())
                        # print(mid)
                        possible_points = [list(x) for x in zip(possible_points[0], possible_points[1])]
                        # print(possible_points)
                        point_prompt = self.NClosest(possible_points, mid, 1)
                        point_prompt = [point_prompt[0][::-1]]
                        point_prompt = np.asarray(point_prompt)  
                        label_prompt = np.asarray([1])
                        # print(point_prompt)
                    
                    # print(point_prompt)
                    all_point_prompt.append(torch.as_tensor(self.transform.apply_coords(point_prompt,  all_gt_copy[0].shape), dtype=torch.float))
                    all_label_prompt.append(torch.as_tensor(label_prompt, dtype=torch.int))
                    all_point_prompt_original.append(torch.as_tensor(point_prompt , dtype = torch.float))

            if ((point_prompt is None) and is_point_prompt) or ((mask_prompt is None) and is_mask_prompt):
                return self.__getitem__(index+1)
        
        # print(all_point_prompt, all_point_prompt[i].shape)
        # print(all_label_prompt, all_label_prompt[i].shape)
        # print(all_mask_prompt, all_mask_prompt[i].shape)
        # if all_point_prompt is not None:
        #     print(all_point_prompt, all_point_prompt[0], all_point_prompt[0].shape, sep="\n")
            
        # if all_label_prompt is not None:
        #     print(all_label_prompt, all_label_prompt[0], all_label_prompt[0].shape, sep="\n")
            
        # if all_mask_prompt is not None:
        #     print(all_mask_prompt, all_mask_prompt[0], all_mask_prompt[0].shape, sep="\n")
       
       
        return {
            "image": imgs, # Tensor(3, H (transformed), 1024)
            "org_img":img_copy,
            "gt_mask":  torch.stack(all_gt_copy), # Tensor(n, H, W)
            "original_size": all_gt_copy[0].shape, # List (H,W) - [720, 1280]
            "point_coords": torch.stack(all_point_prompt, dim=0) if all_point_prompt is not None else None, # torch Tensor (B, N,2),
            "point_coords_original": torch.stack(all_point_prompt_original,dim=0) if all_point_prompt_original is not None else None,
            "point_labels": torch.stack(all_label_prompt, dim=0) if all_label_prompt is not None else None, # torch Tensor (B, N,)
            "mask_inputs": torch.stack(all_mask_prompt, dim=0).unsqueeze(1) if all_mask_prompt is not None else None, # torch Tensor (H, W) -> ideally requires 256,256 which is done inside forward pass for logging purposes
            "info":info,
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
    cfg,
    root_dir="raw",
    batchsize=1,
    seq_len=4,
    iou_range = [0.15, 0.5],
    max_prompt_points = 3,
    trainsize=1024,
    num_workers=0,
    pin_memory=True,
    collate_fn=collate_fn,
    
):
    train_dataset = VideoDataset(root_dir, seq_len, trainsize, iou_range=iou_range, max_prompt_points=max_prompt_points, finetune=True, is_validation_prompt_mid = False)
    train_data_loader = data.DataLoader(
        dataset=train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        collate_fn=collate_fn,
    )
    
    val_dataset = VideoDataset(root_dir, seq_len, trainsize, iou_range=iou_range, max_prompt_points=max_prompt_points, finetune=False, is_validation_prompt_mid = True)
    val_data_loader = data.DataLoader(
        dataset=val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        collate_fn=collate_fn,
    )
    

    return train_data_loader, val_data_loader