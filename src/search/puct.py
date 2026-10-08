"""PUCT scoring."""

from __future__ import annotations

import math


def puct_score(q: float, prior: float, parent_n: int, child_n: int, c_puct: float = 1.5) -> float:
    return q + c_puct * prior * math.sqrt(max(parent_n, 1)) / (1 + child_n)
