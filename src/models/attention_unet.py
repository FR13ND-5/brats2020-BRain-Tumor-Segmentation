"""Attention U-Net (Oktay et al., 2018).

Mechanism: same encoder-decoder as U-Net, but each skip connection passes
through an attention gate. The decoder signal (what we are reconstructing)
gates the encoder features (what we copied), suppressing irrelevant regions
before concatenation — the network learns *where to look*.
"""
import torch
import torch.nn as nn

from .unet import DoubleConv


class AttentionGate(nn.Module):
    """Additive attention: alpha = sigmoid(psi(relu(W_g*g + W_x*x))), out = alpha * x."""

    def __init__(self, gate_channels, skip_channels, inter_channels):
        super().__init__()
        self.w_gate = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, 1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        self.w_skip = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, 1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

    def forward(self, gate, skip):
        alpha = self.psi(torch.relu(self.w_gate(gate) + self.w_skip(skip)))
        return alpha * skip


class AttentionUNet(nn.Module):
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
        self.gates = nn.ModuleList()
        self.decoders = nn.ModuleList()
        prev = features[-1] * 2
        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(prev, f, 2, stride=2))
            self.gates.append(AttentionGate(f, f, f // 2))
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
        for up, gate, dec, skip in zip(self.ups, self.gates, self.decoders,
                                       reversed(skips)):
            x = up(x)
            x = dec(torch.cat([gate(x, skip), x], dim=1))
        return self.head(x)
