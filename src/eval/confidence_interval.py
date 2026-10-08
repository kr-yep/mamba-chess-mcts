"""Confidence intervals."""

from __future__ import annotations

import math


def wilson_interval(wins: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    phat = wins / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    margin = z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n) / denom
    return center - margin, center + margin
