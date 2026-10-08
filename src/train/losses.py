"""Training losses."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from chess_env.legal_mask import mask_illegal_logits


def masked_policy_loss(logits: torch.Tensor, targets: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    masked = mask_illegal_logits(logits, legal_mask)
    return F.cross_entropy(masked, targets)


def value_loss(pred: torch.Tensor, target: torch.Tensor, kind: str = "smooth_l1") -> torch.Tensor:
    pred = pred.squeeze(-1)
    if kind == "mse":
        return F.mse_loss(pred, target)
    return F.smooth_l1_loss(pred, target)
