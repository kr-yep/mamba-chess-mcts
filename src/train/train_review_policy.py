"""Fine-tune a checkpoint on Stockfish review correction targets."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from train.losses import masked_policy_loss
from train.metrics import topk_accuracy
from train.train_policy import PolicyDataset
from utils.checkpoint import save_checkpoint
from utils.config import load_config
from utils.device import get_device
from utils.model_loading import build_model_from_checkpoint
from utils.seed import set_seed


def run_epoch(model, loader, optimizer, device, train: bool) -> dict[str, float]:
    model.train(train)
    total_loss = top1 = top3 = top5 = n = 0.0
    for batch in tqdm(loader, desc="review train" if train else "review eval"):
        target = batch["target"].to(device)
        mask = batch["legal_mask"].to(device)
        if "planes" in batch:
            logits, _ = model(batch["planes"].to(device))
        else:
            logits, _ = model(batch["tokens"].to(device), batch["rows"].to(device), batch["cols"].to(device))
        loss = masked_policy_loss(logits, target, mask)
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        bs = target.shape[0]
        total_loss += loss.item() * bs
        top1 += topk_accuracy(logits, target, mask, 1) * bs
        top3 += topk_accuracy(logits, target, mask, 3) * bs
        top5 += topk_accuracy(logits, target, mask, 5) * bs
        n += bs
    return {"loss": total_loss / max(n, 1), "top1": top1 / max(n, 1), "top3": top3 / max(n, 1), "top5": top5 / max(n, 1)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_review_policy.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 2028))
    device = get_device()
    model_name = cfg["model"]["name"]
    model, loaded_name = build_model_from_checkpoint(cfg["init_checkpoint"], model_name, device)
    shuffled = loaded_name == "board_mamba_shuffled" or model_name == "board_mamba_shuffled"

    ds = PolicyDataset(cfg["data"]["review_policy"], model_name, shuffled=shuffled)
    n_val = max(1, int(len(ds) * cfg["data"].get("val_frac", 0.1)))
    n_train = len(ds) - n_val
    if n_train <= 0:
        raise ValueError("Review dataset is too small for training.")
    train_ds, val_ds = torch.utils.data.random_split(
        ds, [n_train, n_val], generator=torch.Generator().manual_seed(cfg.get("seed", 2028))
    )
    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False, num_workers=0)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"].get("weight_decay", 0.01)
    )

    rows = []
    for epoch in range(cfg["train"]["epochs"]):
        tr = run_epoch(model, train_loader, optimizer, device, train=True)
        va = run_epoch(model, val_loader, optimizer, device, train=False)
        row = {
            "epoch": epoch,
            "model": getattr(model, "model_name", loaded_name),
            **{f"train_{k}": v for k, v in tr.items()},
            **{f"val_{k}": v for k, v in va.items()},
        }
        rows.append(row)
        print(row)

    out = Path(cfg["output"]["metrics"])
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    save_checkpoint(cfg["output"]["checkpoint"], model, optimizer, config=cfg)


if __name__ == "__main__":
    main()
