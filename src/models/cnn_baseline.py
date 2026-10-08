"""Small CNN/ResNet chess baseline."""

from __future__ import annotations

import torch
from torch import nn

from models.heads import PolicyValueHeads


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class CNNBaseline(nn.Module):
    def __init__(self, in_planes: int = 18, channels: int = 192, n_blocks: int = 6, dropout: float = 0.1):
        super().__init__()
        self.model_name = "CNNBaseline"
        self.stem = nn.Sequential(
            nn.Conv2d(in_planes, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(channels, dropout) for _ in range(n_blocks)])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.heads = PolicyValueHeads(channels, dropout=dropout)

    def forward(self, planes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.blocks(self.stem(planes))
        x = self.pool(x).flatten(1)
        return self.heads(x)
