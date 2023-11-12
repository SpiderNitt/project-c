from omegaconf import OmegaConf

cfg = {
    "description": "",
    "precision": "32",
    "num_devices": 1,
    "num_epochs": 150,
    "save_log_weights_interval": 25,
    "train_metric_interval": 25,
    'val_interval': 25,
    "inference_epochs": [],
    "model_checkpoint_at": "checkpoints",
    "img_size": 1024,
    "out_dir": "/",
    "focal_wt": 20,
    "num_tokens": 1,
    "opt": {
        "learning_rate": 5e-4,
        "auto_lr": False,
        "weight_decay": 1e-4,
        "decay_factor": 1/2,
        "steps": [50, 100],
    },
    "model": {
        "type": "vit_l",
        "checkpoint": "sam_vit_l_0b3195.pth",
        "requires_grad": {
            "image_encoder": False,
            "prompt_encoder": False,
            "mask_decoder": False,
            "propagation_module": True,
        },
        "multimask_output": True,
        "propagation_ckpt": None,
    },
    "dataset": {
        "name": "davis",
        "root_dir": "DAVIS-evaluation/data/",
        "stage1": False,
        "train_batch_size": 4,
        "val_batch_size": 2,
        "max_num_obj": 3,
        "num_frames": 3,
        "max_jump": 5,
        "num_workers": 4,
        "pin_memory": True,
        "persistent_workers": True,
    },
}
cfg = OmegaConf.create(cfg)