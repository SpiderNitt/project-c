import lightning as L
import torch

from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import OmegaConf

from segment_anything import sam_model_registry
from segment_anything.modeling import CamoSam
import sys
sys.path.append("dataloaders/")
# from dataloaders.MoCA import get_loader

from dataloaders.vos_dataset import get_loader

import os

# supported measures: 'MAE'       => Mean Squared Error
#                     'E-measure' =>  Enhanced-alignment measure
#                     'S-measure' =>  Structure-measure
#                     'Max-F'     =>  Maximum F-measure
#                     'Adp-F'     =>  Adaptive F-measure
#                     'Wgt-F'     =>  Weighted F-measure

# config = {
#     "precision": 32,
#     "seq_len": 4,
#     "num_devices": 1,
#     "batch_size": 2,
#     "num_workers": 0,
#     "num_epochs": 50,
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
#         "train": {
#             "root_dir": "raw/MoCA-Mask/MoCA_Video",
#         },
#         "val": {
#             "root_dir": "raw/MoCA-Mask/MoCA_Video",
#         },
#     },
# }

config = {
    "train_split": 0.93,
    "precision": 32,
    "seq_len": 4,
    "num_devices": 1,
    "batch_size": 2,
    "num_workers": 0,
    "num_epochs": 50,
    "augmentation_fps_max_frame_skip":5,
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
            "root_dir": "raw/DAVIS/",

    },
}


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
cfg = OmegaConf.create(config)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = sam_model_registry['vit_b'](checkpoint=None, cfg=cfg)
model = CamoSam(cfg, model)
# resume_path = "/content/drive/MyDrive/Group3/sam-finetuning/sam_vit_b_01ec64.pth"
# model.load_state_dict(load_state_dict(resume_path, location='cpu'), strict=False)

train_dataloader, validation_dataloader = get_loader(cfg.dataset.root_dir, 
                                                     train_split=cfg.train_split,
                                                    batchsize= cfg.batch_size,  
                                                    max_jump = cfg.augmentation_fps_max_frame_skip,
                                                    num_frames=cfg.seq_len,  
                                                    max_num_obj=1, )

wandblogger = WandbLogger(project="Propagation")
wandblogger.experiment.config.update(config)
checkpoint_callback = ModelCheckpoint(
    dirpath="./checkpoint", every_n_epochs=1, save_top_k=-1
)
trainer = L.Trainer(
    accelerator=device,
    callbacks=[checkpoint_callback],
    precision=cfg.precision,
    logger=wandblogger,
    max_epochs=cfg.num_epochs,
    # strategy="ddp",
    log_every_n_steps=cfg.log_every_n_steps,
)

trainer.fit(model, train_dataloader, validation_dataloader)
