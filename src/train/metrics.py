"""Policy and value metrics."""

from __future__ import annotations

import numpy as np
import torch

from chess_env.legal_mask import mask_illegal_logits


def topk_accuracy(logits: torch.Tensor, targets: torch.Tensor, legal_mask: torch.Tensor, k: int) -> float:
    masked = mask_illegal_logits(logits, legal_mask)
    topk = masked.topk(k, dim=-1).indices
    return (topk == targets.unsqueeze(-1)).any(dim=-1).float().mean().item()


def legal_move_rate(logits: torch.Tensor, legal_mask: torch.Tensor) -> float:
    pred = mask_illegal_logits(logits, legal_mask).argmax(dim=-1)
    return legal_mask.gather(1, pred.unsqueeze(-1)).float().mean().item()


def value_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    p = pred.detach().cpu().float().view(-1).numpy()
    t = target.detach().cpu().float().view(-1).numpy()
    mae = float(np.mean(np.abs(p - t)))
    rmse = float(np.sqrt(np.mean((p - t) ** 2)))
    p_std = np.std(p)
    t_std = np.std(t)
    corr = float(np.corrcoef(p, t)[0, 1]) if len(p) > 1 and np.isfinite(p_std) and np.isfinite(t_std) and p_std > 0 and t_std > 0 else 0.0
    if not np.isfinite(corr):
        corr = 0.0
    sign_acc = float(np.mean(np.sign(p) == np.sign(t)))
    return {"mae": mae, "rmse": rmse, "pearson": corr, "sign_accuracy": sign_acc}
