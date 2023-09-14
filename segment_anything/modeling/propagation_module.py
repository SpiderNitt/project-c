import torch
import torch.nn as nn
from torch.nn import functional as F

class PropagationModule(nn.Module):
    def __init__(self, seq_len=4) -> None:
        super().__init__()
        self.conv2d = nn.Conv2d(seq_len, 1, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv2d(x) 