"""Elo estimates from score rates."""

from __future__ import annotations

import math


def score_to_elo(score: float, eps: float = 1e-4) -> float:
    score = min(max(score, eps), 1 - eps)
    return -400.0 * math.log10(1.0 / score - 1.0)
