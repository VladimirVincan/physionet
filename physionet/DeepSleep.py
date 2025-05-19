import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding='same'),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(out_channels),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding='same'),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(out_channels),
        )

    def forward(self, x):
        return self.block(x)


class DeepSleep(nn.Module):
    def __init__(self, in_channels=13, out_channels=1):
        super().__init__()
        # Downsampling
        self.enc1 = ConvBlock(in_channels, 15)
        self.enc2 = ConvBlock(15, 18)
        self.enc3 = ConvBlock(18, 21)
        self.enc4 = ConvBlock(21, 25)
        self.enc5 = ConvBlock(25, 30)
        self.enc6 = ConvBlock(30, 60)
        self.enc7 = ConvBlock(60, 120)
        self.enc8 = ConvBlock(120, 240)

        self.pool2 = nn.MaxPool1d(2)
        self.pool4 = nn.MaxPool1d(4)

        # Bottleneck
        self.bottleneck = ConvBlock(240, 480)

        # Upsampling
        self.up8 = nn.ConvTranspose1d(480, 240, kernel_size=4, stride=4)
        self.dec8 = ConvBlock(480, 240)
        self.up7 = nn.ConvTranspose1d(240, 120, kernel_size=4, stride=4)
        self.dec7 = ConvBlock(240, 120)
        self.up6 = nn.ConvTranspose1d(120, 60, kernel_size=4, stride=4)
        self.dec6 = ConvBlock(120, 60)
        self.up5 = nn.ConvTranspose1d(60, 30, kernel_size=4, stride=4)
        self.dec5 = ConvBlock(60, 30)
        self.up4 = nn.ConvTranspose1d(30, 25, kernel_size=4, stride=4)
        self.dec4 = ConvBlock(50, 25)
        self.up3 = nn.ConvTranspose1d(25, 21, kernel_size=4, stride=4)
        self.dec3 = ConvBlock(42, 21)
        self.up2 = nn.ConvTranspose1d(21, 18, kernel_size=4, stride=4)
        self.dec2 = ConvBlock(36, 18)
        self.up1 = nn.ConvTranspose1d(18, 15, kernel_size=2, stride=2)
        self.dec1 = nn.Conv1d(30, 1, 7, padding='same')

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.permute(0, 2, 1)  # B x N x 13 -> B x 13 x N

        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool2(e1))
        e3 = self.enc3(self.pool4(e2))
        e4 = self.enc4(self.pool4(e3))
        e5 = self.enc5(self.pool4(e4))
        e6 = self.enc6(self.pool4(e5))
        e7 = self.enc7(self.pool4(e6))
        e8 = self.enc8(self.pool4(e7))

        # Bottleneck
        b = self.bottleneck(self.pool4(e8))

        # Decoder
        d8 = self.up8(b)
        d8 = self.dec8(torch.cat([d8, e8], dim=1))
        d7 = self.up7(d8)
        d7 = self.dec7(torch.cat([d7, e7], dim=1))
        d6 = self.up6(d7)
        d6 = self.dec6(torch.cat([d6, e6], dim=1))
        d5 = self.up5(d6)
        d5 = self.dec5(torch.cat([d5, e5], dim=1))
        d4 = self.up4(d5)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))
        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        out = self.sigmoid(d1)
        out = out.squeeze(1)  # B x 1 x N -> B x N

        return out
