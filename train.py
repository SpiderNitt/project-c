import torch
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import Callback, ModelSummary, LearningRateMonitor
from segment_anything import sam_model_registry
from segment_anything.modeling import CamoSam

import os
import wandb
from pathlib import Path

from dataloaders.vos_dataset import get_loader
from callbacks import WandB_Logger, InferCallback
from config import cfg

torch.set_float32_matmul_precision('high')

path  = Path(f'DAVIS-evaluation/Results')
path.mkdir(parents=True, exist_ok=True)

ckpt = None

if cfg.model.propagation_ckpt:
    if 'artifacts' in cfg.model.propagation_ckpt:
        api = wandb.Api()
        artifact = api.artifact('spider-r-d/Common Propagation/model_cyzxvd5f:v26', type='model')
        artifact_dir = artifact.download()

        ckpt = torch.load('artifacts/model_cyzxvd5f:v26/499_epoch_7500_global_step.pth')
    elif cfg.model.propagation_ckpt:
        ckpt = torch.load(cfg.model.propagation_ckpt)

# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
device = "cuda" if torch.cuda.is_available() else "cpu"
model = sam_model_registry[cfg.model.type](checkpoint=cfg.model.checkpoint, cfg=cfg)
model = CamoSam(cfg, model, ckpt=ckpt)

wandblogger = WandbLogger(project="Common Propagation", save_code=True, settings=wandb.Settings(code_dir="."))

# torch._dynamo.config.verbose=True # for debugging
lr_monitor = LearningRateMonitor(logging_interval='epoch')
model_weight_callback = WandB_Logger(cfg, wandblogger)
infer_callback = InferCallback(cfg, wandblogger)

callbacks = [ModelSummary(max_depth=3), lr_monitor, model_weight_callback, infer_callback] if "davis" in cfg.dataset.name else [ModelSummary(max_depth=3), lr_monitor, model_weight_callback]

trainer = L.Trainer(
    accelerator=device,
    devices=cfg.num_devices,
    callbacks=callbacks,
    precision=cfg.precision,
    logger=wandblogger,
    max_epochs=cfg.num_epochs,
    num_sanity_val_steps=0,
    # strategy="ddp",
    log_every_n_steps=15,
    check_val_every_n_epoch=cfg.val_interval,
    enable_checkpointing=False,
    profiler='simple',
    # overfit_batches=1
)
if trainer.global_rank == 0:
    wandblogger.experiment.config.update(dict(cfg))

train_dataloader, validation_dataloader = get_loader(cfg.dataset)
trainer.fit(model, train_dataloader, validation_dataloader)