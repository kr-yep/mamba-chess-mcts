"""Positional embeddings for board sequences."""

from __future__ import annotations

import torch
from torch import nn


class Board2DPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_row_col: int = 9):
        super().__init__()
        self.row = nn.Embedding(max_row_col, d_model)
        self.col = nn.Embedding(max_row_col, d_model)
        self.square = nn.Embedding(65, d_model)

    def forward(self, x: torch.Tensor, rows: torch.Tensor, cols: torch.Tensor) -> torch.Tensor:
        square_ids = torch.clamp(rows, 0, 7) * 8 + torch.clamp(cols, 0, 7)
        square_ids = torch.where((rows == 8) | (cols == 8), torch.full_like(square_ids, 64), square_ids)
        return x + self.row(rows) + self.col(cols) + self.square(square_ids)
