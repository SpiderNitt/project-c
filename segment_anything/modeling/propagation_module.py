import torch
import torch.nn as nn
from torch.nn import functional as F

class PropagationModule(nn.Module):
    def __init__(self, seq_len=4) -> None:
        super().__init__()
        self.conv2d_0 = nn.Conv2d(seq_len, seq_len * 4, kernel_size=3, padding=1)
        self.silu = nn.SiLU()
        self.conv2d_1 = nn.Conv2d(seq_len * 4, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.silu(self.conv2d_0(x))
        return self.conv2d_1(x)