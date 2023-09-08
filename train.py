import lightning as L
from torch.utils.data import DataLoader

from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import OmegaConf

from segment_anything import sam_model_registry
from segment_anything.modeling import ImageLogger, CamoSam
from dataset import MoCA, collate_fn

import os

config = {
    "seq_len": 1,
    "num_devices": 4,
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
            "root_dir": "raw/MoCA-Mask/MoCA_Video/TrainDataset_per_sq",
        },
        "val": {
            "root_dir": "raw/MoCA-Mask/MoCA_Video/TestDataset_per_sq",
        },
    },
}


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
cfg = OmegaConf.create(config)
model = sam_model_registry['vit_b'](checkpoint=cfg.model.checkpoint)
model = CamoSam(cfg, model)
# resume_path = "/content/drive/MyDrive/Group3/sam-finetuning/sam_vit_b_01ec64.pth"
# model.load_state_dict(load_state_dict(resume_path, location='cpu'), strict=False)

train = MoCA(cfg, train=True)
val = MoCA(cfg, train=False)

train_dataloader = DataLoader(train,
                              batch_size=cfg.batch_size,
                              shuffle=True,
                              num_workers=cfg.num_workers,
                              collate_fn=collate_fn
                              )
val_dataloader = DataLoader(val,
                            batch_size=cfg.batch_size,
                            shuffle=True,
                            num_workers=cfg.num_workers,
                            collate_fn=collate_fn)

wandblogger = WandbLogger()
checkpoint_callback = ModelCheckpoint(
    dirpath="./checkpoint", every_n_epochs=1, save_top_k=-1
)
trainer = L.Trainer(
    callbacks=[checkpoint_callback],
    precision=32,
    logger=wandblogger,
    max_epochs=cfg.num_epochs,
    # strategy="ddp",
    log_every_n_steps=16,
)

trainer.fit(model, train_dataloader, val_dataloader)