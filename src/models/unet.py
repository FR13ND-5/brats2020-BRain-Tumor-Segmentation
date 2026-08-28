"""U-Net (Ronneberger et al., 2015) — the baseline.

Mechanism: encoder-decoder with plain skip connections that copy encoder
features straight into the decoder at each resolution.
"""
import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """(Conv3x3 -> BN -> ReLU) x 2 — the basic block shared by all our models."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=3, features=(64, 128, 256, 512)):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.encoders = nn.ModuleList()
        prev = in_channels
        for f in features:
            self.encoders.append(DoubleConv(prev, f))
            prev = f
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        prev = features[-1] * 2
        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(prev, f, 2, stride=2))
            self.decoders.append(DoubleConv(f * 2, f))
            prev = f
        self.head = nn.Conv2d(features[0], out_channels, 1)

    def forward(self, x):
        skips = []
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for up, dec, skip in zip(self.ups, self.decoders, reversed(skips)):
            x = up(x)
            x = dec(torch.cat([skip, x], dim=1))
        return self.head(x)
