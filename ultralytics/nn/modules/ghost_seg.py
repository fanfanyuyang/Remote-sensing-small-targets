import torch
import torch.nn as nn
import math
from .conv import Conv, DWConv

class GhostModuleSeg(nn.Module):
    def __init__(self, c_in=512, c_mid=256, c_out=512, dummy=None):
        super().__init__()
        # 禁用手动缩放，用YAML原始通道值
        self.primary = Conv(c_in, c_mid, k=3, s=2, p=1)
        self.cheap_op = nn.Sequential(
            DWConv(c_mid, c_mid, k=3, s=1, d=1),
            DWConv(c_mid, c_mid, k=3, s=1, d=2)
        )
        self.out_conv = Conv(c_mid * 2, c_out, k=1, s=1)

    def forward(self, x):
        x_primary = self.primary(x)
        x_ghost = self.cheap_op(x_primary)
        x_out = torch.cat([x_primary, x_ghost], dim=1)
        return self.out_conv(x_out)