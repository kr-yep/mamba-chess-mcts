"""Shared policy and value heads."""

from __future__ import annotations

import torch
from torch import nn

from chess_env.action_space import ACTION_SIZE


class PolicyValueHeads(nn.Module):
    def __init__(self, d_model: int, policy_dim: int = ACTION_SIZE, dropout: float = 0.1):
        super().__init__()
        self.policy = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, policy_dim),
        )
        self.value = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.policy(x), self.value(x)
