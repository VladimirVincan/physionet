import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, kernel_size=21):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size, padding='same'),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_channels, out_channels, kernel_size, padding='same'),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DresdenDeepSleep(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.name = 'DresdenDeepSleep'
        # Downsampling
        self.enc1 = ConvBlock(in_channels, 20, 20)
        self.enc2 = ConvBlock(20, 25, 25)
        self.enc3 = ConvBlock(25, 30, 30)
        self.enc4 = ConvBlock(30, 60, 60)
        self.enc5 = ConvBlock(60, 120, 120)
        self.enc6 = ConvBlock(120, 240, 240)
        self.enc7 = ConvBlock(240, 480, 480)

        self.pool2 = nn.MaxPool1d(2)
        self.pool4 = nn.MaxPool1d(4)

        # Bottleneck
        self.bottleneck = ConvBlock(480, 480, 480)


        # Upsampling
        self.upsample = nn.Upsample(scale_factor=2)
        self.dec7 = ConvBlock(960, 480, 240)
        self.dec6 = ConvBlock(480, 360, 120)
        self.dec5 = ConvBlock(240, 180, 60)
        self.dec4 = ConvBlock(120, 90, 30)
        self.dec3 = ConvBlock(60, 45, 25)
        self.dec2 = ConvBlock(50, 27, 21)
        self.dec1 = ConvBlock(41, 23, 21)

        # output
        self.output = nn.Conv1d(21, 1, 1, padding='same')
        # self.output = nn.Sequential(
        #     nn.Conv1d(21, 2, 1, padding='same'),
        #     nn.Sigmoid()
        # )

    # start and end are used in eval if signal was padded
    def forward(self, x, start=0, end=2_097_152):
        x = x.permute(0, 2, 1)  # B x N x 13 -> B x 13 x N

        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool2(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool2(e3))
        e5 = self.enc5(self.pool2(e4))
        e6 = self.enc6(self.pool2(e5))
        e7 = self.enc7(self.pool2(e6))

        # Bottleneck
        b = self.bottleneck(self.pool2(e7))

        # Decoder
        d7 = self.upsample(b)
        d7 = self.dec7(torch.cat([d7, e7], dim=1))
        d6 = self.upsample(d7)
        d6 = self.dec6(torch.cat([d6, e6], dim=1))
        d5 = self.upsample(d6)
        d5 = self.dec5(torch.cat([d5, e5], dim=1))
        d4 = self.upsample(d5)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))
        d3 = self.upsample(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.upsample(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.upsample(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        out = self.output(d1)

        if not self.training:
            out = torch.nn.functional.upsample(out, scale_factor=4, mode='linear')
            out = out[start*4 : (end+1)*4]

        out = out.squeeze(1)  # B x 1 x N -> B x N
        return out
