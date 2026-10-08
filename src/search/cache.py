"""Simple position evaluation cache."""

from __future__ import annotations


class EvalCache:
    def __init__(self, max_size: int = 100000):
        self.max_size = max_size
        self._data: dict[str, tuple[dict[int, float], float]] = {}

    def get(self, key: str):
        return self._data.get(key)

    def put(self, key: str, value):
        if len(self._data) >= self.max_size:
            self._data.pop(next(iter(self._data)))
        self._data[key] = value
