import os
from os import path, replace
import random
import torch
from torch.utils.data.dataset import Dataset
import torch.utils.data as data
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torch.nn import functional as F

from PIL import Image
import numpy as np
from torchvision.transforms.functional import resize
from segment_anything.utils.transforms import ResizeLongestSide
from .range_transform import im_normalization, im_mean
from .reseed import reseed

class VOSDataset(Dataset):
    """
    Works for DAVIS/YouTubeVOS/BL30K training
    For each sequence:
    - Pick three frames
    - Pick two objects
    - Apply some random transforms that are the same for all frames
    - Apply random transform to each of the frame
    - The distance between frames is controlled
    """
    def __init__(self, im_root, gt_root, vid_list, max_jump, subset=None, num_frames=3, max_num_obj=3, val=False):
        self.im_root = im_root
        self.gt_root = gt_root
        self.max_jump = max_jump
        self.num_frames = num_frames
        self.max_num_obj = max_num_obj

        self.videos = []
        self.frames = {}

        vid_list = sorted(os.listdir(self.im_root))
        # Pre-filtering
        for vid in vid_list:
            if subset is not None:
                if vid not in subset:
                    continue
            frames = sorted(os.listdir(os.path.join(self.im_root, vid)))
            if len(frames) < num_frames:
                continue
            self.frames[vid] = frames
            self.videos.append(vid)

        print('%d out of %d videos accepted in %s.' % (len(self.videos), len(vid_list), im_root))

        # These set of transform is the same for im/gt pairs, but different among the 3 sampled frames
        self.pair_im_lone_transform = transforms.Compose([
            transforms.ColorJitter(0.01, 0.01, 0.01, 0),
        ])

        self.pair_im_dual_transform = transforms.Compose([
            transforms.RandomAffine(degrees=0 if val else 15, shear=0 if val else 10, interpolation=InterpolationMode.BILINEAR, fill=im_mean),
        ])

        self.pair_gt_dual_transform = transforms.Compose([
            transforms.RandomAffine(degrees=0 if val else 15, shear=0 if val else 10, interpolation=InterpolationMode.NEAREST, fill=0),
        ])

        # These transform are the same for all pairs in the sampled sequence
        self.all_im_lone_transform = transforms.Compose([
            transforms.ColorJitter(0.1, 0.03, 0.03, 0),
            transforms.RandomGrayscale(0.05),
        ])


        self.all_im_dual_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize((1080, 1920), interpolation=InterpolationMode.NEAREST)
        ])
        self.all_gt_dual_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize((1080, 1920), interpolation=InterpolationMode.NEAREST)
        ])

        # Final transform without randomness
        self.final_im_transform = transforms.Compose([
            transforms.ToTensor(),
            im_normalization,
        ])

        self.resize_longest = ResizeLongestSide(1024)
    
    def preprocess_prev_masks(self, x):
        h, w = x.shape[-2:]
        padh = 1024 - h
        padw = 1024 - w
        x = F.pad(x, (0, padw, 0, padh))
        x = resize(x, (256, 256), interpolation=InterpolationMode.BILINEAR, antialias=True)
        return x

    def __getitem__(self, idx):
        video = self.videos[idx]
        info = {}
        info['name'] = video

        vid_im_path = path.join(self.im_root, video)
        vid_gt_path = path.join(self.gt_root, video)
        frames = self.frames[video]

        trials = 0
        while trials < 5:
            info['frames'] = [] # Appended with actual frames

            num_frames = self.num_frames
            length = len(frames)
            this_max_jump = min(len(frames), self.max_jump)

            # iterative sampling
            frames_idx = [np.random.randint(length)]
            acceptable_set = set(range(max(0, frames_idx[-1]-this_max_jump), min(length, frames_idx[-1]+this_max_jump+1))).difference(set(frames_idx))
            while(len(frames_idx) < num_frames):
                idx = np.random.choice(list(acceptable_set))
                frames_idx.append(idx)
                new_set = set(range(max(0, frames_idx[-1]-this_max_jump), min(length, frames_idx[-1]+this_max_jump+1)))
                acceptable_set = acceptable_set.union(new_set).difference(set(frames_idx))

            frames_idx = sorted(frames_idx)
            if np.random.rand() < 0.5:
                # Reverse time
                frames_idx = frames_idx[::-1]

            sequence_seed = np.random.randint(2147483647)
            images = []
            masks = []
            target_objects = []
            for f_idx in frames_idx:
                jpg_name = frames[f_idx][:-4] + '.jpg'
                png_name = frames[f_idx][:-4] + '.png'
                info['frames'].append(jpg_name)

                reseed(sequence_seed)
                this_im = Image.open(path.join(vid_im_path, jpg_name)).convert('RGB')
                this_im = self.all_im_dual_transform(this_im)
                this_im = self.all_im_lone_transform(this_im)
                reseed(sequence_seed)
                this_gt = Image.open(path.join(vid_gt_path, png_name)).convert('P')
                this_gt = self.all_gt_dual_transform(this_gt)

                pairwise_seed = np.random.randint(2147483647)
                reseed(pairwise_seed)
                this_im = self.pair_im_dual_transform(this_im)
                this_im = self.pair_im_lone_transform(this_im)
                reseed(pairwise_seed)
                this_gt = self.pair_gt_dual_transform(this_gt)

                this_im = torch.as_tensor(self.resize_longest.apply_image(np.array(this_im, dtype=np.uint8))).permute(2, 0, 1)
                this_gt = np.array(this_gt)

                images.append(this_im)
                masks.append(this_gt)

            images = torch.stack(images, 0)

            labels = np.unique(masks[0])
            # Remove background
            labels = labels[labels!=0]

            if len(labels) == 0:
                target_objects = []
                trials += 1
            else:
                target_objects = labels.tolist()
                break

        if len(target_objects) > self.max_num_obj:
            target_objects = np.random.choice(target_objects, size=self.max_num_obj, replace=False)

        info['num_objects'] = max(1, len(target_objects))

        masks = np.stack(masks, 0)

        # Generate one-hot ground-truth
        cls_gt = np.zeros((self.num_frames, 1080, 1920), dtype=np.int64)
        first_frame_gt = np.zeros((1, self.max_num_obj, 1080, 1920), dtype=np.int64)
        all_frame_gt = np.zeros((self.num_frames, self.max_num_obj, 1080, 1920), dtype=np.int64) # Shape explains itself
        for t in range(self.num_frames):
            for i, l in enumerate(target_objects):
                this_mask = (masks==l)
                cls_gt[this_mask] = i+1
                first_frame_gt[0,i] = (this_mask[0])
                all_frame_gt[t,i] = (this_mask[t])

        cls_gt = np.expand_dims(cls_gt, 1)
        prev_frame_gt = all_frame_gt[:-1].reshape(-1, 1080, 1920)
        new_prev_frame_gt = []
        for t in range(len(prev_frame_gt)):
            new_prev_frame_gt.append(torch.as_tensor(self.resize_longest.apply_image(prev_frame_gt[t].astype(dtype=np.uint8))))
        new_prev_frame_gt = torch.stack(new_prev_frame_gt, 0).reshape(-1, self.max_num_obj, *new_prev_frame_gt[0].shape[-2:])
        new_prev_frame_gt = self.preprocess_prev_masks(new_prev_frame_gt)

        all_frame_gt = torch.as_tensor(all_frame_gt).float()

        # 1 if object exist, 0 otherwise
        selector = [1 if i < info['num_objects'] else 0 for i in range(self.max_num_obj)]
        selector = torch.BoolTensor(selector)

        # (H', W') -> after LongestSideResize
        data = {
            'image': images, # (num_frames=3, 3, H', W') 
            'gt_mask': all_frame_gt[-1], # (num_obj=3, H, W)
            'prev_masks': new_prev_frame_gt, # (num_frames=2, num_obj=3, 256, 256)
            'selector': selector, # (num_obj=3) Indicates if ith object exists
            'info': info,
            'original_size': all_frame_gt.shape[-2:]
        }

        return data

    def __len__(self):
        return len(self.videos)
   
def collate_fn(batch):
    return batch    

def get_loader(cfg):    
    with open(cfg.root_dir+'ImageSets/2017/train.txt', 'r') as file:
        train_list = [line.strip() for line in file]
    print("Training Samples: ",len(train_list))

    train_dataset = VOSDataset(cfg.root_dir+'JPEGImages/Full-Resolution', 
                               cfg.root_dir+'Annotations/Full-Resolution', 
                               train_list ,max_jump=cfg.max_jump, 
                               num_frames=cfg.num_frames,  
                               max_num_obj=cfg.max_num_obj, 
                               val=False)
    train_data_loader = data.DataLoader(
        dataset=train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        collate_fn=collate_fn,
    )
    
    with open(cfg.root_dir+'ImageSets/2017/val.txt', 'r') as file:
        val_list = [line.strip() for line in file]
    print("Validation Samples: ",len(val_list))

    val_dataset = VOSDataset(cfg.root_dir+'JPEGImages/Full-Resolution', 
                             cfg.root_dir+'Annotations/Full-Resolution', 
                             val_list, max_jump=cfg.max_jump, 
                             num_frames=cfg.num_frames,  
                             max_num_obj=cfg.max_num_obj, 
                             val=True)
    val_data_loader = data.DataLoader(
        dataset=val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        collate_fn=collate_fn,
    )

    return train_data_loader, val_data_loader