import torch
import torch.nn as nn
from unet import UNet

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.DataParallel(UNet())
        
    def forward(self, x_t, t):
        return self.base(x_t, t)
    