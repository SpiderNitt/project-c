import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.rank_zero import rank_zero_only, rank_zero_warn
from inference import infer

import wandb
import os
from pathlib import Path

class WandB_Logger(Callback):
    def __init__(self, cfg, wnb):
        super().__init__()
        self.cfg = cfg
        self.wnb = wnb.experiment
        self.version = wnb.version
        self.switch_flag = False
    
    @rank_zero_only
    def on_train_epoch_end(self, trainer, pl_module):
        if (pl_module.current_epoch + 1) % self.cfg.save_log_weights_interval == 0:
            Path(os.path.join(self.cfg.model_checkpoint_at), self.version).mkdir(parents=True, exist_ok=True)
            model_name = f"{os.path.join(self.cfg.model_checkpoint_at, self.version, f'{pl_module.current_epoch}_epoch_{trainer.global_step}_global_step.pth')}"
            
            if pl_module.current_epoch == 0:
                pl_module.train_benchmark = []
                pl_module.val_benchmark = []
            else:
                pl_module.train_benchmark = sum(pl_module.train_benchmark) / len(pl_module.train_benchmark)
                pl_module.val_benchmark = sum(pl_module.val_benchmark) / len(pl_module.val_benchmark)
            
            model_dict = {
                'cfg': self.cfg,
                'epoch': pl_module.current_epoch,
                'model_state_dict': pl_module.model.propagation_module.state_dict(),
                'optimizer_state_dict': pl_module.optimizers().state_dict() if type(pl_module.optimizers())!=list else {},
                'benchmark': [pl_module.train_benchmark, pl_module.val_benchmark],
                'decoder_state_dict': pl_module.model.mask_decoder.state_dict() if self.cfg.model.requires_grad.mask_decoder else {},
            }
            
            torch.save(model_dict, model_name)
            torch.save(model_dict, f"{os.path.join(self.cfg.model_checkpoint_at, f'{self.cfg.dataset.num_frames * self.cfg.dataset.stage1}.pth')}")

            my_model = wandb.Artifact(f"model_{self.version}", type="model")
            my_model.add_file(model_name)
            self.wnb.log_artifact(my_model)
                
            pl_module.train_benchmark = []
            pl_module.val_benchmark = []


class InferCallback(Callback):
    def __init__(self, cfg, wnb):
        super().__init__()
        self.cfg = cfg
        self.wnb = wnb.experiment
        self.version = wnb.version

        results_path  = Path(f'DAVIS-evaluation/Results/{self.version}')
        results_path.mkdir(parents=True, exist_ok=True)

    @rank_zero_only
    def on_train_epoch_end(self, trainer, pl_module):
        if (pl_module.current_epoch + 1) in self.cfg.inference_epochs:
            with open(self.cfg.dataset.root_dir + 'DAVIS/ImageSets/2017/val.txt', 'r') as f:
                val_list = [line.strip() for line in f]

            print('---Inferencing---')
            pl_module.model.eval()
            with torch.inference_mode():
                for video in val_list:
                    infer(pl_module.model, pl_module.device, video, self.cfg.dataset, f'{self.version}/epoch_{pl_module.current_epoch + 1}', self.cfg.model.multimask_output)
            pl_module.model.train()

            os.system(f'cd DAVIS-evaluation; export PYTHONPATH=$(pwd)/python/lib; python python/tools/eval.py -i data/DAVIS/Predictions/{self.version}/epoch_{pl_module.current_epoch + 1} -o Results/{self.version}/epoch_{pl_module.current_epoch + 1}.yaml --year 2017 --phase val')
            print('---Inferencing Done---')

            artifact = wandb.Artifact(name="Results", type="Results")
            artifact.add_file(local_path=f"./DAVIS-evaluation/Results/{self.version}/epoch_{pl_module.current_epoch + 1}.yaml")
            self.wnb.log_artifact(artifact)