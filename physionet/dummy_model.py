import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DummyModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        in_channels = 13
        hidden_dim = 64

        # Conv1D expects input of shape (batch, channels, sequence_length)
        self.conv1 = nn.Conv1d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.out_layer = nn.Conv1d(hidden_dim, 1, kernel_size=1)  # Output one value per timestep

    def forward(self, x):
        """
        x: Tensor of shape (batch_size, sequence_length, in_channels)
        Returns:
            logits of shape (batch_size, sequence_length)
        """
        x = x.permute(0, 2, 1)  # (batch, in_channels, seq_len)
        # print(x.shape)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.out_layer(x)  # (batch, 1, seq_len)
        x = x.squeeze(1)       # (batch, seq_len)
        x = F.sigmoid(x)
        return x  # You can apply sigmoid outside if needed
