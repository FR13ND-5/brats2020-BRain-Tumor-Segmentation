"""TransUNet-style hybrid (after Chen et al., 2021).

Mechanism: a CNN encoder extracts local features, then the bottleneck feature
map is flattened into tokens and run through a Transformer encoder, giving
every spatial position a global receptive field — self-attention relates
distant regions that convolutions only reach through many layers. A U-Net
decoder with skip connections restores resolution.
"""
import torch
import torch.nn as nn

from .unet import DoubleConv


class TransUNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=3, features=(64, 128, 256, 512),
                 num_layers=4, num_heads=8, image_size=240):
        super().__init__()
        embed_dim = features[-1] * 2  # 1024 at the bottleneck
        self.pool = nn.MaxPool2d(2)
        self.encoders = nn.ModuleList()
        prev = in_channels
        for f in features:
            self.encoders.append(DoubleConv(prev, f))
            prev = f
        self.bottleneck = DoubleConv(features[-1], embed_dim)

        # ponytail: pos embedding fixed to image_size (15x15 tokens for 240);
        # interpolate it if other input sizes are ever needed.
        n_tokens = (image_size // 2 ** len(features)) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, n_tokens, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 2,
            dropout=0.1, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)

        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        prev = embed_dim
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
        x = self.bottleneck(x)  # (B, C, h, w) — 15x15 for 240x240 input

        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # (B, h*w, C)
        tokens = self.transformer(tokens + self.pos_embed)
        x = tokens.transpose(1, 2).reshape(b, c, h, w)

        for up, dec, skip in zip(self.ups, self.decoders, reversed(skips)):
            x = up(x)
            x = dec(torch.cat([skip, x], dim=1))
        return self.head(x)
