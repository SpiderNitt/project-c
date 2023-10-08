import lightning as L
import torch
import wandb
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import Callback, ModelSummary
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
    "save_log_weights_interval": 15,
    "metric_train_eval_interval": 90,
    "model_checkpoint_at": "checkpoints",
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

class WandB_Logger(Callback):
    def __init__(self, cfg, wb):
        self.logged_weight_epoch = 0
        self.cfg = cfg
        self.wb = wb
        
        if cfg.model_checkpoint_at not in os.listdir():
            os.mkdir(cfg.model_checkpoint_at)
            

    def on_validation_epoch_end(self, trainer, pl_module):
        if (pl_module.current_epoch) % self.cfg.save_log_weights_interval == 0:
            model_name = f"{os.path.join(cfg.model_checkpoint_at, f'{pl_module.current_epoch}_epoch_{trainer.global_step}_global_step.pth')}"
            
            if pl_module.current_epoch == 0:
                pl_module.train_benchmark = []
                pl_module.val_benchmark = []
            else:
                pl_module.train_benchmark = sum(pl_module.train_benchmark) / len(pl_module.train_benchmark)
                pl_module.val_benchmark = sum(pl_module.val_benchmark) / len(pl_module.val_benchmark)
            
            torch.save({
                        'cfg': self.cfg,
                        'epoch': pl_module.current_epoch,
                        'model_state_dict': pl_module.model.state_dict(),
                        'optimizer_state_dict': pl_module.optimizers().state_dict() if type(pl_module.optimizers())!=list else {},
                        'benchmark': [pl_module.train_benchmark, pl_module.val_benchmark],
            }, model_name)
            my_model = wandb.Artifact(f"model_{self.wb.id}", type="model")
            my_model.add_file(model_name)
            self.wb.log_artifact(my_model)
            # Link the model to the Model Registry
            self.wb.link_artifact(my_model, f"DAVIS Propagation/Model_arch_1")
            
        pl_module.train_benchmark = []
        pl_module.val_benchmark = []

# torch._dynamo.config.verbose=True # for debugging
wandblogger = WandbLogger(project="DAVIS Propagation")
wandblogger.experiment.config.update(config)
model_weight_callback = WandB_Logger(cfg, wandblogger.experiment)

trainer = L.Trainer(
    accelerator=device,
    callbacks=[ModelSummary(max_depth=2), model_weight_callback],
    precision=cfg.precision,
    logger=wandblogger,
    max_epochs=cfg.num_epochs,
    strategy="ddp",
    # log_every_n_steps=cfg.log_every_n_steps,
    check_val_every_n_epoch=20,
)

trainer.validate(model, validation_dataloader)
trainer.fit(model, train_dataloader, validation_dataloader)