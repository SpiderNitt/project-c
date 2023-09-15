import lightning as L
import torch
from torch.utils.data import DataLoader

from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import OmegaConf

from segment_anything import sam_model_registry
from segment_anything.modeling import ImageLogger, CamoSam
from dataloaders import VideoDataset, get_loader

import os

config = {
    "seq_len": 4,
    "num_devices": 1,
    "batch_size": 1,
    "num_workers": 0,
    "num_epochs": 1,
    "eval_interval": 20,
    "img_size": 1024,
    "out_dir": "/",
    "opt": {
        "learning_rate": 8e-4,
        "weight_decay": 1e-4,
        "decay_factor": 10,
        "steps": [60000, 86666],
        "warmup_steps": 250,
    },
    "model": {
        "type": "vit_b",
        "checkpoint": "sam_vit_b_01ec64.pth",
        "freeze": {
            "image_encoder": True,
            "prompt_encoder": True,
            "mask_decoder": True,
        },
    },
    "dataset": {
        "train": {
            "root_dir": "MoCA-Mask/MoCA_Video",
        },
        "val": {
            "root_dir": "MoCA-Mask/MoCA_Video",
        },
    },
}


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
cfg = OmegaConf.create(config)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = sam_model_registry['vit_b'](checkpoint=None, cfg=cfg)
model = CamoSam(cfg, model)
# resume_path = "/content/drive/MyDrive/Group3/sam-finetuning/sam_vit_b_01ec64.pth"
# model.load_state_dict(load_state_dict(resume_path, location='cpu'), strict=False)

train_dataloader = get_loader(cfg.dataset.train.root_dir, cfg.batch_size, cfg.seq_len)

wandblogger = WandbLogger()
checkpoint_callback = ModelCheckpoint(
    dirpath="./checkpoint", every_n_epochs=1, save_top_k=-1
)
trainer = L.Trainer(
    accelerator=device,
    callbacks=[checkpoint_callback],
    precision=32,
    logger=wandblogger,
    max_epochs=cfg.num_epochs,
    # strategy="ddp",
    log_every_n_steps=16,
)

trainer.fit(model, train_dataloader)
