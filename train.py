import lightning as L
import torch

from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, ModelSummary
from omegaconf import OmegaConf

from segment_anything import sam_model_registry
from segment_anything.modeling import CamoSam

from dataloaders.davis import get_loader

import os

# DAVIS
config = {
    "precision": "32",
    "num_devices": 1,
    "num_epochs": 50,
    "metric_train_eval_interval": 250,
    "img_size": 1024,
    "out_dir": "/",
    "opt": {
        "learning_rate": 3e-4,
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
        "root_dir": "raw/DAVIS/",
        "batch_size": 1,
        "max_num_obj": 3,
        "num_frames": 3,
        "max_jump": 5,
        "num_workers": 4,
        "pin_memory": True,
    },
}
cfg = OmegaConf.create(config)

# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
device = "cuda" if torch.cuda.is_available() else "cpu"
model = sam_model_registry[cfg.model.type](checkpoint=cfg.model.checkpoint, cfg=cfg)
model = CamoSam(cfg, model)
# model = torch.compile(model, mode="reduce-overhead")

train_dataloader, validation_dataloader = get_loader(cfg.dataset)

# torch._dynamo.config.verbose=True # for debugging
wandblogger = WandbLogger(project="DAVIS Propagation")
wandblogger.experiment.config.update(config)
checkpoint_callback = ModelCheckpoint(
    dirpath="./checkpoint", every_n_epochs=1, save_top_k=-1
)
trainer = L.Trainer(
    accelerator=device,
    callbacks=[ModelSummary(max_depth=2), checkpoint_callback],
    precision=cfg.precision,
    logger=wandblogger,
    max_epochs=cfg.num_epochs,
    strategy="ddp",
    # log_every_n_steps=cfg.log_every_n_steps,
    check_val_every_n_epoch=20,
)

trainer.fit(model, train_dataloader, validation_dataloader)