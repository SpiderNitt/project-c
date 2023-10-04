import lightning as L
import torch
import wandb
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, Callback
from omegaconf import OmegaConf

from segment_anything import sam_model_registry
from segment_anything.modeling import CamoSam
import sys
sys.path.append("dataloaders/")
from dataloaders.MoCA import get_loader

# from dataloaders.vos_dataset import get_loader

import os

# supported measures: 'MAE'       => Mean Squared Error
#                     'E-measure' =>  Enhanced-alignment measure
#                     'S-measure' =>  Structure-measure
#                     'Max-F'     =>  Maximum F-measure
#                     'Adp-F'     =>  Adaptive F-measure
#                     'Wgt-F'     =>  Weighted F-measure

############## MoCA
config = {
    "model_checkpoint_at": "checkpoint/",
    "save_log_weights_interval":2,
    "precision": 32,
    "seq_len": 4,
    "num_devices": 1,
    "batch_size": 2,
    "num_workers": 0,
    "num_epochs": 20,
    "max_prompt_points":2,
    "iou_range": [0.02, 0.2], # mask prompt
    "metric_train_eval_interval": 20,
    "log_every_n_steps": 1,
    "log_metrics": ['MAE', 'E-measure', 'S-measure', 'Max-F', 'Adp-F', 'Wgt-F'],
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
            "root_dir": "raw/MoCA-Mask/MoCA_Video",
        },
        "val": {
            "root_dir": "raw/MoCA-Mask/MoCA_Video",
        },
    },
}
############## DAVIS
# config = {
#     "train_split": 0.93,
#     "precision": 32,
#     "seq_len": 4,
#     "num_devices": 1,
#     "batch_size": 2,
#     "num_workers": 0,
#     "num_epochs": 50,
#     "augmentation_fps_max_frame_skip":5,
#     "metric_train_eval_interval": 20,
#     "log_every_n_steps": 1,
#     "log_metrics": ['MAE', 'E-measure', 'S-measure', 'Max-F', 'Adp-F', 'Wgt-F'],
#     "img_size": 1024,
#     "out_dir": "/",
#     "opt": {
#         "learning_rate": 8e-4,
#         "weight_decay": 1e-4,
#         "decay_factor": 10,
#         "steps": [60000, 86666],
#         "warmup_steps": 250,
#     },
#     "model": {
#         "type": "vit_b",
#         "checkpoint": "sam_vit_b_01ec64.pth",
#         "freeze": {
#             "image_encoder": True,
#             "prompt_encoder": True,
#             "mask_decoder": True,
#         },
#     },
#     "dataset": {
#             "root_dir": "raw/DAVIS/",

#     },
# }


class WandB_Logger(Callback):
    def __init__(self, cfg, wb):
        self.logged_weight_epoch = 0
        self.cfg = cfg
        self.wb = wb

    def on_validation_epoch_end(self, trainer, pl_module):
        if (pl_module.current_epoch) % self.cfg.save_log_weights_interval == 0 and pl_module.current_epoch >0:
            model_name = f"{os.path.join(cfg.model_checkpoint_at, f'{pl_module.current_epoch}_epoch_{trainer.global_step}_global_step.pth')}"
            torch.save({
                        'cfg': self.cfg,
                        'epoch': pl_module.current_epoch,
                        'model_state_dict': pl_module.model.state_dict(),
                        'optimizer_state_dict': pl_module.optimizers().state_dict() if type(pl_module.optimizers())!=list else {},
                        'benchmark': [pl_module.train_benchmark, pl_module.val_benchmark],
            }, model_name)
            print("UPLOADING MODEL TO W&B", model_name)
            my_model = wandb.Artifact(f"model_{self.wb.id}", type="model")
            my_model.add_file(model_name)
            self.wb.log_artifact(my_model)
            # Link the model to the Model Registry
            self.wb.link_artifact(my_model, f"Adapter/TEST")

# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
cfg = OmegaConf.create(config)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = sam_model_registry['vit_b'](checkpoint="sam_vit_b_01ec64.pth", cfg=cfg)
model = CamoSam(cfg, model)
# resume_path = "/content/drive/MyDrive/Group3/sam-finetuning/sam_vit_b_01ec64.pth"
# model.load_state_dict(load_state_dict(resume_path, location='cpu'), strict=False)


# MoCA
train_dataloader, validation_dataloader = get_loader(cfg.dataset.train.root_dir, cfg.batch_size, cfg.seq_len, cfg.iou_range, cfg.max_prompt_points)

# DAVIS
# train_dataloader, validation_dataloader = get_loader(cfg.dataset.root_dir, 
#                                                      train_split=cfg.train_split,
#                                                     batchsize= cfg.batch_size,  
#                                                     max_jump = cfg.augmentation_fps_max_frame_skip,
#                                                     num_frames=cfg.seq_len,  
#                                                     max_num_obj=1, )

wandblogger = WandbLogger(project="Adapter")
wandblogger.experiment.config.update(config)

model_weight_callback = WandB_Logger(cfg, wandblogger.experiment)

trainer = L.Trainer(
    accelerator=device,
    callbacks=[model_weight_callback],
    precision=cfg.precision,
    logger=wandblogger,
    max_epochs=cfg.num_epochs,
    # strategy="ddp",
    log_every_n_steps=cfg.log_every_n_steps,
)

trainer.validate(model, validation_dataloader)
trainer.fit(model, train_dataloader, validation_dataloader)
