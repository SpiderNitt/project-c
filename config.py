from omegaconf import OmegaConf

cfg = {
    "description": "Positional embed before cross attention and affinity. Added extra affinity module. Each Pos_embed_wt has shape (256)",
    "precision": "32",
    "num_devices": 1,
    "num_epochs": 100,
    "save_log_weights_interval": 25,
    "train_metric_interval": 25,
    'val_interval': 25,
    "inference_epochs": [50, 100],
    "model_checkpoint_at": "checkpoints",
    "img_size": 1024,
    "out_dir": "/",
    "focal_wt": 20,
    "num_tokens": 1,
    "opt": {
        "learning_rate": 4e-4,
        "auto_lr": False,
        "weight_decay": 1e-4,
        "decay_factor": 1/2,
        "steps": [75],
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
        "propagation_ckpt": None,
    },
    "dataset": {
        "name": "davis",
        "root_dir": "raw/",
        "stage1": False,
        "train_batch_size": 4,
        "val_batch_size": 2,
        "max_num_obj": 3,
        "num_frames": 3,
        "max_jump": 5,
        "num_workers": 4,
        "pin_memory": False,
        "persistent_workers": True,
    },
}
cfg = OmegaConf.create(cfg)