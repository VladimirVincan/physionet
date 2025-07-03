import torch
import torch.nn as nn
import torch.nn.functional as Functional
from mamba_ssm import Mamba

from SleepNet import Sleep_model_MultiTarget


class BiMamba(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()

        #(batch_size, input_dim, time_steps) -> (batch_size, hidden_dim, time_steps)
        self.proj = nn.Conv1d(in_channels=input_dim,
                              out_channels=hidden_dim,
                              kernel_size=1)
        self.mamba_f = Mamba(hidden_dim)
        self.mamba_b = Mamba(hidden_dim)

    def forward(self, x):
        x = self.proj(x)
        x = x.permute(0, 2, 1)
        x_b = self.mamba_b(x[:, torch.arange(x.size(1) - 1, -1, -1), :])
        x_f = self.mamba_f(x)
        x = torch.cat((x_f, x_b), dim=2)
        x = x.permute(0, 2, 1)
        return x


class SkipMamba(nn.Module):
    def __init__(self, in_channels, settings, out_channels=4, hiddenSize=32):
        super(SkipMamba, self).__init__()

        # Store parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.settings = settings

        # Bidirectional LSTM to apply temporally across input channels
        self.bimamba = BiMamba(input_dim=in_channels, hidden_dim=hiddenSize).to(settings['device'])

        # Output convolution to map the LSTM hidden states from forward and backward pass to the output shape
        self.outputConv1 = nn.Conv1d(in_channels=hiddenSize*2, out_channels=hiddenSize, groups=1, kernel_size=1, padding=0)
        self.outputConv1 = nn.utils.weight_norm(self.outputConv1, name='weight')

        self.outputConv2 = nn.Conv1d(in_channels=hiddenSize, out_channels=out_channels, groups=1, kernel_size=1, padding=0)
        self.outputConv2 = nn.utils.weight_norm(self.outputConv2, name='weight')

        # Residual mapping
        self.identMap1 = nn.Conv1d(in_channels=in_channels, out_channels=hiddenSize, groups=1, kernel_size=1, padding=0)

    def forward(self, x):
        y = self.bimamba(x)
        y = Functional.tanh((self.outputConv1(y) + self.identMap1(x)) / 1.41421)
        y = self.outputConv2(y)

        return y


class BiMambaBlock(nn.Module):
    def __init__(self, in_channels, settings, out_channels=4, hiddenSize=32):
        super().__init__()

        self.bimamba1 = BiMamba(input_dim=in_channels, hidden_dim=hiddenSize).to(settings['device'])
        self.bimamba2 = BiMamba(input_dim=hidden_channels, hidden_dim=hiddenSize).to(settings['device'])
        self.bimamba3 = BiMamba(input_dim=hidden_channels, hidden_dim=hiddenSize).to(settings['device'])
        self.bimamba4 = BiMamba(input_dim=hidden_channels, hidden_dim=hiddenSize).to(settings['device'])
        self.bimamba5 = BiMamba(input_dim=hidden_channels, hidden_dim=hiddenSize).to(settings['device'])
        self.bimamba6 = BiMamba(input_dim=hidden_channels, hidden_dim=hiddenSize).to(settings['device'])

        self.output = nn.Linear(hiddenSize, out_channels)

    def forward(self, x):
        x = self.bimamba1(x)
        x = self.bimamba2(x)
        x = self.bimamba3(x)
        x = self.bimamba4(x)
        x = self.bimamba5(x)
        x = self.bimamba6(x)
        x = self.output(x)
        x = torch.sigmoid(x)

        return x


class MambaSleepNet(Sleep_model_MultiTarget):
    def __init__(self, settings, num_signals=12):
        super().__init__(settings, num_signals)
        # self.skipLSTM = SkipMamba(((14*self.channelMultiplier)+1)*self.numSignals, settings=settings, hiddenSize=self.channelMultiplier*64, out_channels=4)
        self.skipLSTM = BiMambaBlock(((14*self.channelMultiplier)+1)*self.numSignals, settings=settings, hiddenSize=self.channelMultiplier*64, out_channels=4)
