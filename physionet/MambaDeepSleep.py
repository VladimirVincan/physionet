import torch
import torch.nn as nn
import torch.nn.functional as Functional
from mamba_ssm import Mamba, Mamba2

from DeepSleep import ConvBlock, DeepSleep


class BiMamba(nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()

        #(batch_size, input_dim, time_steps) -> (batch_size, hidden_dim, time_steps)
        self.proj = nn.Conv1d(in_channels=hidden_dim * 2,
                              out_channels=hidden_dim,
                              kernel_size=1)
        self.mamba_f = Mamba(hidden_dim)
        self.mamba_b = Mamba(hidden_dim)

    def forward(self, x):
        x = x.permute(
            0, 2, 1
        )  # (batch_size, hidden_dim, time_steps) -> (batch_size, time_steps, hidden_dim)
        x_b = self.mamba_b(x[:, torch.arange(x.size(1) - 1, -1, -1), :])
        x_f = self.mamba_f(x)
        x = torch.cat((x_f, x_b), dim=2)
        x = x.permute(0, 2, 1)
        x = self.proj(x)
        return x


class BiT_MamSleep(nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()
        self.mamba_f = Mamba(hidden_dim)
        self.mamba_b = Mamba(hidden_dim)
        self.conv_f = nn.Conv1d(in_channels=hidden_dim,
                                out_channels=hidden_dim,
                                kernel_size=1)
        self.conv_b = nn.Conv1d(in_channels=hidden_dim,
                                out_channels=hidden_dim,
                                kernel_size=1)
        self.lin_mamba = nn.Conv1d(in_channels=hidden_dim,
                                   out_channels=hidden_dim,
                                   kernel_size=1)
        self.lin_gate = nn.Conv1d(in_channels=hidden_dim,
                                  out_channels=hidden_dim,
                                  kernel_size=1)
        self.lin_out = nn.Conv1d(in_channels=hidden_dim,
                                 out_channels=hidden_dim,
                                 kernel_size=1)
        self.silu = nn.SiLU()
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # print(x.shape, flush=True)
        # return x
        # x = self.norm(x.permute(0,2,1)).permute(0,2,1)
        x = self.norm(x.permute(0,2,1)).permute(0,2,1)
        x_mamba_f = self.lin_mamba(x)
        x_mamba_b = x_mamba_f[:, torch.arange(x_mamba_f.size(1) - 1, -1, -1), :]

        x_mamba_f = self.silu(self.conv_f(x_mamba_f)).permute(0, 2, 1)
        x_mamba_b = self.silu(self.conv_b(x_mamba_b)).permute(0, 2, 1)

        x_gate = self.silu(self.lin_gate(x))
        x_mamba_f = self.mamba_f(x_mamba_f)
        x_mamba_b = self.mamba_b(x_mamba_b)
        x_mamba_b = x_mamba_b[:, torch.arange(x_mamba_b.size(1) - 1, -1, -1), :]
        x_mamba = x_mamba_f + x_mamba_b
        x_mamba = x_mamba.permute(0, 2, 1)
        x_mamba = x_mamba * x_gate
        x_mamba = self.lin_out(x_mamba)
        x_mamba = self.norm(x_mamba.permute(0,2,1)).permute(0,2,1)
        # x_mamba = self.norm(x_mamba)
        return x_mamba


class MambaBottleneck(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=7):
        super().__init__()
        self.bottleneck = ConvBlock(240, 480)
        self.bimamba1 = BiT_MamSleep(hidden_dim=out_channels)
        self.bimamba2 = BiT_MamSleep(hidden_dim=out_channels)
        self.bimamba3 = BiT_MamSleep(hidden_dim=out_channels)

    def forward(self, x):
        x = self.bottleneck(x)
        x_mamba = self.bimamba1(x)
        x_mamba2 = self.bimamba2(x_mamba+x)
        x_mamba3 = self.bimamba3(x_mamba2+x_mamba)
        return x_mamba3+x_mamba2
        # return torch.cat([x, x_mamba], dim=1)


class MambaDeepSleep(DeepSleep):

    def __init__(self, in_channels=13, out_channels=1):
        super().__init__(in_channels, out_channels)
        self.bottleneck = MambaBottleneck(240, 480)
        # self.up8 = nn.ConvTranspose1d(960, 240, kernel_size=4, stride=4)
