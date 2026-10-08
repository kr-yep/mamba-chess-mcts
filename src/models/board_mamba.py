"""Board-Mamba policy-value model."""

from __future__ import annotations

import torch
from torch import nn

from chess_env.board_encoding import VOCAB_SIZE
from models.heads import PolicyValueHeads
from models.positional_encoding import Board2DPositionalEncoding

try:
    from mamba_ssm.modules.mamba_simple import Mamba

    HAS_MAMBA = True
except Exception:
    Mamba = None
    HAS_MAMBA = False


class MambaFallbackBlock(nn.Module):
    """Explicit fallback used only when mamba-ssm is unavailable."""

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(d_model, d_model // 2, batch_first=True, bidirectional=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.gru(x)
        return self.norm(x + self.dropout(y))


class BoardMamba(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 6,
        d_state: int = 16,
        dropout: float = 0.1,
        use_2d_pe: bool = False,
        vocab_size: int = VOCAB_SIZE,
    ):
        super().__init__()
        self.model_name = "BoardMamba" if HAS_MAMBA else "MambaFallback"
        self.use_2d_pe = use_2d_pe
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.sequence_pos = nn.Embedding(96, d_model)
        self.pos2d = Board2DPositionalEncoding(d_model) if use_2d_pe else None
        self.dropout = nn.Dropout(dropout)
        if HAS_MAMBA:
            self.layers = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.LayerNorm(d_model),
                        Mamba(d_model=d_model, d_state=d_state),
                    )
                    for _ in range(n_layers)
                ]
            )
        else:
            self.layers = nn.ModuleList([MambaFallbackBlock(d_model, dropout) for _ in range(n_layers)])
        self.final_norm = nn.LayerNorm(d_model)
        self.heads = PolicyValueHeads(d_model, dropout=dropout)

    def forward(
        self, tokens: torch.Tensor, rows: torch.Tensor | None = None, cols: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0)
        x = self.token_embedding(tokens) + self.sequence_pos(positions)
        if self.pos2d is not None:
            if rows is None or cols is None:
                raise ValueError("rows and cols are required when use_2d_pe=True")
            x = self.pos2d(x, rows, cols)
        x = self.dropout(x)
        for layer in self.layers:
            if HAS_MAMBA:
                residual = x
                x = residual + layer(x)
            else:
                x = layer(x)
        cls = self.final_norm(x[:, 0])
        return self.heads(cls)
