"""Timing helpers."""

from __future__ import annotations

import time
from contextlib import contextmanager


@contextmanager
def timer():
    start = time.perf_counter()
    box = {"elapsed": 0.0}
    try:
        yield box
    finally:
        box["elapsed"] = time.perf_counter() - start
