import torch
import torch.nn as nn
import torch.nn.functional as Functional
from mamba_ssm import Mamba

from DeepSleep import ConvBlock, DeepSleep


class BiMamba(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        #(batch_size, input_dim, time_steps) -> (batch_size, hidden_dim, time_steps)
        self.proj = nn.Conv1d(in_channels=hidden_dim*2,
                              out_channels=hidden_dim,
                              kernel_size=1)
        self.mamba_f = Mamba(hidden_dim)
        self.mamba_b = Mamba(hidden_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch_size, hidden_dim, time_steps) -> (batch_size, time_steps, hidden_dim)
        x_b = self.mamba_b(x[:, torch.arange(x.size(1) - 1, -1, -1), :])
        x_f = self.mamba_f(x)
        x = torch.cat((x_f, x_b), dim=2)
        x = x.permute(0, 2, 1)
        x = self.proj(x)
        return x


class MambaBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7):
        super().__init__()
        self.bottleneck = ConvBlock(240, 480)
        self.bimamba = BiMamba(hidden_dim=out_channels)


    def forward(self, x):
        x = self.bottleneck(x)
        x_mamba = self.bimamba(x)
        return torch.cat([x, x_mamba], dim=1)


class MambaDeepSleep(DeepSleep):
    def __init__(self, in_channels=13, out_channels=1):
        super().__init__(in_channels, out_channels)
        self.bottleneck = MambaBottleneck(240, 480)
        self.up8 = nn.ConvTranspose1d(920, 240, kernel_size=4, stride=4)
