"""Model factory."""

from __future__ import annotations

from models.board_mamba import BoardMamba
from models.cnn_baseline import CNNBaseline


def build_model(name: str, **kwargs):
    name = name.lower()
    if name == "board_mamba":
        return BoardMamba(use_2d_pe=False, **kwargs)
    if name == "board_mamba_2dpe":
        return BoardMamba(use_2d_pe=True, **kwargs)
    if name == "board_mamba_shuffled":
        return BoardMamba(use_2d_pe=False, **kwargs)
    if name == "cnn_baseline":
        return CNNBaseline(**kwargs)
    raise ValueError(f"Unknown model: {name}")
