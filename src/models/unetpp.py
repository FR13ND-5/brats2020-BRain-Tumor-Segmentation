"""UNet++ (Zhou et al., 2018).

Mechanism: replaces U-Net's plain skips with nested, dense skip pathways.
Node X(i,j) at resolution row i fuses all previous nodes of its row with the
upsampled node from the row below, so encoder features are gradually refined
before reaching the decoder — bridging the semantic gap between encoder and
decoder features.
"""
import torch
import torch.nn as nn

from .unet import DoubleConv


class UNetPP(nn.Module):
    # ponytail: no deep supervision — single output head; add if the report
    # needs the pruning/ensembling story.
    def __init__(self, in_channels=4, out_channels=3,
                 features=(32, 64, 128, 256, 512)):
        super().__init__()
        self.depth = len(features)
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.blocks = nn.ModuleDict()
        for i in range(self.depth):            # row = resolution level
            for j in range(self.depth - i):    # column = position along the row
                if j == 0:
                    in_ch = in_channels if i == 0 else features[i - 1]
                else:
                    in_ch = features[i] * j + features[i + 1]
                self.blocks[f"x{i}{j}"] = DoubleConv(in_ch, features[i])
        self.head = nn.Conv2d(features[0], out_channels, 1)

    def forward(self, x):
        grid = {}
        for i in range(self.depth):
            grid[(i, 0)] = self.blocks[f"x{i}0"](
                x if i == 0 else self.pool(grid[(i - 1, 0)]))
        for j in range(1, self.depth):
            for i in range(self.depth - j):
                row_feats = [grid[(i, k)] for k in range(j)]
                below = self.up(grid[(i + 1, j - 1)])
                grid[(i, j)] = self.blocks[f"x{i}{j}"](
                    torch.cat(row_feats + [below], dim=1))
        return self.head(grid[(0, self.depth - 1)])
