"""Load models with architecture metadata from checkpoints."""

from __future__ import annotations

from pathlib import Path

import torch

from models.model_factory import build_model


def build_model_from_checkpoint(
    checkpoint_path: str | Path | None,
    fallback_name: str,
    device,
):
    """Build a model using checkpoint metadata when available."""

    if checkpoint_path is None:
        return build_model(fallback_name).to(device), fallback_name

    payload = torch.load(checkpoint_path, map_location=device)
    cfg = payload.get("metadata", {}).get("config", {})
    model_cfg = cfg.get("model", {})
    model_name = model_cfg.get("name", fallback_name)
    kwargs = model_cfg.get("kwargs", {})
    model = build_model(model_name, **kwargs).to(device)
    model.load_state_dict(payload["model"])
    return model, model_name
