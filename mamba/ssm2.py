import numpy as np
import torch
import torch.nn as nn
from mamba_ssm import Mamba


class StateSpaceModel2(nn.Module):

    def __init__(self, scale=1):
        super().__init__()

        self.scale = scale

        self.encoder_layers = nn.ModuleList([
            DoublePointConv(13, 16, 64),
            nn.MaxPool1d(16, stride=16, return_indices=True),
            DoublePointConv(16, 32, 32),
            nn.MaxPool1d(4, stride=4, return_indices=True),
            DoublePointConv(32, 64, 16),
            nn.MaxPool1d(4, stride=4, return_indices=True),
        ])

        self.middle_layers = nn.ModuleList([
            DoubleConv(64, 64, 8),
            BiMamba(64, 16),
            DoubleConv(128, 64, 8),
        ])

        self.decoder_layers = nn.ModuleList([
            nn.MaxUnpool1d(4, stride=4),
            DoublePointConv(128, 32, 16),
            nn.MaxUnpool1d(4, stride=4),
            DoublePointConv(64, 16, 16),
            nn.MaxUnpool1d(16, stride=16),
            DoublePointConv(32, 32, 16),
        ])

        self.classification_layer = nn.Conv1d(32, 1, kernel_size=1)

    def forward(self, x, sigmoid=False, *args, **kwargs):
        print(x.shape)

        route_connection = []
        indices = []
        x = x.permute(0, 2, 1)
        for layer in self.encoder_layers:
            if isinstance(layer, nn.MaxPool1d):
                x, index = layer(x)
                indices.append(index)
            else:
                x = layer(x)
                route_connection.append(x)
        for layer in self.middle_layers:
            x = layer(x)
        for layer in self.decoder_layers:
            if isinstance(layer, DoublePointConv):
                curr_connection = route_connection.pop(-1)
                x = torch.cat([x, curr_connection], dim=1)
                x = layer(x)
            elif isinstance(layer, nn.MaxUnpool1d):
                index = indices.pop(-1)
                x = layer(x, index)
        x = self.classification_layer(x)

        if not self.training:
            if sigmoid:
                x = torch.nn.functional.sigmoid(x)
            # x = x.permute(0, 2, 1)
            # x = torch.nn.functional.interpolate(x,
            #                                     scale_factor=self.scale,
            #                                     mode='linear')
            # x = x.permute(0, 2, 1)
        return x


class BiMamba(nn.Module):

    def __init__(self, d_model, n_state):
        super(BiMamba, self).__init__()

        self.mamba_forward = Mamba(d_model, n_state)
        self.mamba_backward = Mamba(d_model, n_state)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x_inverse = torch.flip(x, dims=[1])

        x_forward = self.mamba_forward(x)
        x_backward = self.mamba_backward(x_inverse)
        x_backward = torch.flip(x_backward, dims=[1])

        x = torch.cat((x_forward, x_backward), dim=2)
        x = x.permute(0, 2, 1)

        return x


class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv_op = nn.Sequential(
            nn.Conv1d(in_channels,
                      out_channels,
                      kernel_size=kernel_size,
                      padding='same'), nn.GELU(),
            nn.Conv1d(out_channels,
                      out_channels,
                      kernel_size=kernel_size,
                      padding='same'), nn.GELU())

    def forward(self, x):
        return self.conv_op(x)


class DoublePointConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv_op = nn.Sequential(
            nn.Conv1d(in_channels,
                      in_channels,
                      groups=in_channels,
                      kernel_size=kernel_size,
                      padding='same'), nn.GELU(),
            nn.Conv1d(in_channels, out_channels, kernel_size=1,
                      padding='same'), nn.GELU())

    def forward(self, x):
        return self.conv_op(x)
