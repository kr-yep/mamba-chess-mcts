"""Checkpoint helpers."""

from __future__ import annotations

from pathlib import Path

import torch


def save_checkpoint(path: str | Path, model, optimizer=None, **metadata) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": model.state_dict(), "metadata": metadata}
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, path)


def load_checkpoint(path: str | Path, model, optimizer=None, map_location="cpu") -> dict:
    payload = torch.load(path, map_location=map_location)
    model.load_state_dict(payload["model"])
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    return payload.get("metadata", {})
