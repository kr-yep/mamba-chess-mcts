"""Legal move masks for the fixed chess action space."""

from __future__ import annotations

import chess
import torch

from chess_env.action_space import ACTION_SIZE, legal_moves_to_action_ids


def make_legal_mask(board: chess.Board, device: torch.device | str | None = None) -> torch.BoolTensor:
    """Return a `[4672]` boolean tensor with legal actions set to true."""

    mask = torch.zeros(ACTION_SIZE, dtype=torch.bool, device=device)
    ids = legal_moves_to_action_ids(board)
    if ids:
        mask[torch.tensor(ids, dtype=torch.long, device=device)] = True
    return mask


def mask_illegal_logits(logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    """Set illegal logits to `-inf` without changing legal logits."""

    if logits.shape[-1] != ACTION_SIZE:
        raise ValueError(f"Expected last dim {ACTION_SIZE}, got {logits.shape[-1]}")
    while legal_mask.ndim < logits.ndim:
        legal_mask = legal_mask.unsqueeze(0)
    return logits.masked_fill(~legal_mask, torch.finfo(logits.dtype).min)
