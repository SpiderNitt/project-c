import torch
import lightning as L
from segment_anything_fast import sam_model_fast_registry
from segment_anything_fast.modeling.camosam import CamoSam
from lightning.pytorch.loggers import WandbLogger

import argparse
import wandb
import os

from pathlib import Path

from dataloaders.moca_test import get_test_loader
from config import cfg

L.seed_everything(2023, workers=True)
torch.set_float32_matmul_precision('highest')

path = Path(f'DAVIS-evaluation/Results')
path.mkdir(parents=True, exist_ok=True)

parser = argparse.ArgumentParser(description='Your script description')
parser.add_argument('--version', type=str, default="v0", help='Stage')

args = parser.parse_args()

ckpt = None
wandblogger = WandbLogger(project="Proper", save_code=True, settings=wandb.Settings(code_dir="."))
artifact = wandblogger.use_artifact(f'spider-r-d/Proper/model_8ttdc3tf:{args.version}', type='model')
artifact_dir = artifact.download()
a = os.listdir(artifact_dir)[0]
ckpt = torch.load(artifact_dir + '/' + a)

# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
device = "cuda" if torch.cuda.is_available() else "cpu"
model = sam_model_fast_registry[cfg.model.type](checkpoint=cfg.model.checkpoint, cfg=cfg)
model = CamoSam(cfg, model, ckpt=ckpt)

trainer = L.Trainer(
    accelerator=device,
    devices=cfg.num_devices,
    precision="bf16-mixed",
    max_epochs=cfg.num_epochs,
    logger=wandblogger,
    num_sanity_val_steps=0,
    # strategy="ddp",
    log_every_n_steps=15,
    check_val_every_n_epoch=cfg.val_interval if cfg.dataset.stage1 else 1,
    val_check_interval = None if cfg.dataset.stage1 else cfg.val_interval,
    enable_checkpointing=True,
    profiler='simple',
    # overfit_batches=1
)

test_dataloader = get_test_loader()
trainer.validate(model, test_dataloader)