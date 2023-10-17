import lightning as L
from lightning.pytorch import LightningDataModule
import torch
import wandb
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import Callback, ModelSummary, LearningRateMonitor
from omegaconf import OmegaConf
from lightning.pytorch.tuner import Tuner
import torch.utils.data as data
from segment_anything import sam_model_registry
from segment_anything.modeling import CamoSam
from lightning.pytorch.utilities.rank_zero import rank_zero_only, rank_zero_warn

from dataloaders.davis import get_loader, VOSDataset

import os
torch.set_float32_matmul_precision('medium')

# DAVIS
config = {
    "description": "Positional embed before cross attention. No sparse embeddings. Pos embed wt has shape (256)",
    "precision": "32",
    "num_devices": 1,
    "num_epochs": 300,
    "save_log_weights_interval": 20,
    "metric_train_eval_interval": 20,
    "model_checkpoint_at": "checkpoints",
    "img_size": 1024,
    "out_dir": "/",
    "focal_wt": 20,
    "num_tokens": 1,
    "opt": {
        "learning_rate": 4e-4, #1e-4
        "auto_lr": False,
        "weight_decay": 1e-4,
        "decay_factor": 1/2,
        "steps": [100, 250],
    },
    "model": {
        "type": "vit_l",
        "checkpoint": "sam_vit_l_0b3195.pth",
        "freeze": {
            "image_encoder": True,
            "prompt_encoder": True,
            "mask_decoder": True,
        },
    },
    "dataset": {
        "root_dir": "raw/DAVIS/",
        "stage1": False,
        "batch_size": 4,
        "max_num_obj": 3,
        "num_frames": 3,
        "max_jump": 5,
        "num_workers": 4,
        "pin_memory": False,
    },
}
cfg = OmegaConf.create(config)

# api = wandb.Api()
# artifact = api.artifact('spider-r-d/DAVIS Propagation/model_cyzxvd5f:v26', type='model')
# artifact_dir = artifact.download()

# ckpt = torch.load('artifacts/model_cyzxvd5f:v26/499_epoch_7500_global_step.pth')
ckpt = None

# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
device = "cuda" if torch.cuda.is_available() else "cpu"
model = sam_model_registry[cfg.model.type](checkpoint=cfg.model.checkpoint, cfg=cfg)
model = CamoSam(cfg, model, ckpt=ckpt)
# model = torch.compile(model, mode="reduce-overhead")

# class LitDataModule(LightningDataModule):
#     def __init__(self, batch_size):
#         super().__init__()
#         self.save_hyperparameters()
#         # or
#         self.batch_size = batch_size
#         self.cfg = cfg

#     def train_dataloader(self):
#         with open(self.cfg.dataset.root_dir+'ImageSets/2017/train.txt', 'r') as file:
#             train_list = [line.strip() for line in file]
#         print("Training Samples: ",len(train_list))

#         train_dataset = VOSDataset(self.cfg.dataset.root_dir+'JPEGImages/Full-Resolution', 
#                                    self.cfg.dataset.root_dir+'Annotations/Full-Resolution', 
#                                    train_list ,max_jump=self.cfg.dataset.max_jump, 
#                                    num_frames=self.cfg.dataset.num_frames,  
#                                    max_num_obj=self.cfg.dataset.max_num_obj, 
#                                    val=False)
#         train_data_loader = data.DataLoader(
#             dataset=train_dataset,
#             batch_size=self.batch_size,
#             shuffle=True,
#             # num_workers=self.cfg.dataset.num_workers,
#             # pin_memory=self.cfg.dataset.pin_memory,
#         )
        
#         return train_data_loader
    
#     def val_dataloader(self):

#         with open(self.cfg.dataset.root_dir+'ImageSets/2017/val.txt', 'r') as file:
#             val_list = [line.strip() for line in file]
#         print("Validation Samples: ",len(val_list))

#         val_dataset = VOSDataset(self.cfg.dataset.root_dir+'JPEGImages/Full-Resolution', 
#                                  self.cfg.dataset.root_dir+'Annotations/Full-Resolution', 
#                                  val_list, max_jump=self.cfg.dataset.max_jump, 
#                                  num_frames=self.cfg.dataset.num_frames,  
#                                  max_num_obj=self.cfg.dataset.max_num_obj, 
#                                  val=True)
#         val_data_loader = data.DataLoader(
#             dataset=val_dataset,
#             batch_size=self.batch_size,
#             shuffle=False,
#             # num_workers=self.cfg.dataset.num_workers,
#             # pin_memory=self.cfg.dataset.pin_memory,
#         )

#         return val_data_loader

class WandB_Logger(Callback):
    def __init__(self, cfg, wb):
        self.logged_weight_epoch = 0
        self.cfg = cfg
        self.wb = wb
        
        if cfg.model_checkpoint_at not in os.listdir():
            os.mkdir(cfg.model_checkpoint_at)
            
    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module):
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
                    'model_state_dict': pl_module.model.propagation_module.state_dict(),
                    'optimizer_state_dict': pl_module.optimizers().state_dict() if type(pl_module.optimizers())!=list else {},
                    'benchmark': [pl_module.train_benchmark, pl_module.val_benchmark],
        }, model_name)
        my_model = wandb.Artifact(f"model_{self.wb.id}", type="model")
        my_model.add_file(model_name)
        self.wb.log_artifact(my_model)
        # Link the model to the Model Registry
        # self.wb.link_artifact(my_model, f"DAVIS Propagation/Model_arch_1")
            
        pl_module.train_benchmark = []
        pl_module.val_benchmark = []

# torch._dynamo.config.verbose=True # for debugging
wandblogger = WandbLogger(project="DAVIS Propagation", save_code=True, settings=wandb.Settings(code_dir="."))
model_weight_callback = WandB_Logger(cfg, wandblogger.experiment)
lr_monitor = LearningRateMonitor(logging_interval='step')

# datamodule = LitDataModule(cfg.dataset.batch_size)
trainer = L.Trainer(
    accelerator=device,
    devices=cfg.num_devices,
    callbacks=[model_weight_callback, ModelSummary(max_depth=2), lr_monitor],
    precision=cfg.precision,
    logger=wandblogger,
    max_epochs=cfg.num_epochs,
    # strategy="ddp",
    log_every_n_steps=15,
    check_val_every_n_epoch=cfg.metric_train_eval_interval,
    enable_checkpointing=False,
    profiler='simple',
    # overfit_batches=1
)
if trainer.global_rank == 0:
    wandblogger.experiment.config.update(config)
# tuner = Tuner(trainer)
# tuner.lr_find(model, datamodule=datamodule)
# #     #TODO: Add scale batch size
# tuner.scale_batch_size(model, datamodule=datamodule)

train_dataloader, validation_dataloader = get_loader(cfg.dataset)
trainer.fit(model, train_dataloader, validation_dataloader)