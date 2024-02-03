import torch
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import Callback, ModelSummary, LearningRateMonitor
from segment_anything import sam_model_registry
from segment_anything.modeling import CamoSam

import argparse
import wandb
from pathlib import Path

from dataloaders.camo_dataset import get_loader
from dataloaders.vos_dataset import get_loader as get_loader_moca
from callbacks import WandB_Logger, InferCallback
from config import cfg

parser = argparse.ArgumentParser(description='Your script description')
parser.add_argument('--stage', type=int, default=0, help='Stage')
parser.add_argument('--bsize', type=int, default=4, help='Batch size')
parser.add_argument('--num_frames', type=int, default=3, help='Number of frames')
parser.add_argument('--ckpt', type=str, default=None, help='Path to checkpoint (default: None)')
parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs (default: 10)')
parser.add_argument('--lr', type=float, default=0, help='Learning Rate')

args = parser.parse_args()
cfg.dataset.stage1 = bool(args.stage)
if cfg.dataset.stage1:
    cfg.dataset.num_frames = args.num_frames
    cfg.dataset.train_batch_size = args.bsize
    cfg.model.propagation_ckpt = args.ckpt
    cfg.num_epochs = args.num_epochs
    cfg.opt.learning_rate = args.lr

L.seed_everything(2023)
torch.set_float32_matmul_precision('high')

path = Path(f'DAVIS-evaluation/Results')
path.mkdir(parents=True, exist_ok=True)

ckpt = None

if cfg.model.propagation_ckpt:
    ckpt = torch.load(cfg.model.propagation_ckpt)

# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
device = "cuda" if torch.cuda.is_available() else "cpu"
model = sam_model_registry[cfg.model.type](checkpoint=cfg.model.checkpoint, cfg=cfg)
model = CamoSam(cfg, model, ckpt=ckpt)

wandblogger = WandbLogger(project="ECCV_SAM", save_code=True, settings=wandb.Settings(code_dir="."))

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
    check_val_every_n_epoch=cfg.val_interval if cfg.dataset.stage1 else 1,
    val_check_interval = None if cfg.dataset.stage1 else cfg.val_interval,
    enable_checkpointing=True,
    profiler='simple',
    # overfit_batches=1
)
if trainer.global_rank == 0:
    wandblogger.experiment.config.update(dict(cfg))

if cfg.dataset.stage1:
    train_dataloader, validation_dataloader = get_loader_moca(cfg.dataset)
else:
    train_dataloader, _ = get_loader(cfg.dataset)
    _, validation_dataloader = get_loader_moca(cfg.dataset)
trainer.fit(model, train_dataloader, validation_dataloader)