"""Evaluate a checkpoint on policy imitation and Stockfish value validation sets."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from train.metrics import topk_accuracy, value_metrics
from train.train_policy import PolicyDataset
from train.train_value import ValueDataset
from utils.device import get_device
from utils.model_loading import build_model_from_checkpoint


@torch.no_grad()
def eval_policy(model, loader, device) -> dict[str, float]:
    model.eval()
    top1 = top3 = top5 = n = 0.0
    for batch in tqdm(loader, desc="eval policy"):
        target = batch["target"].to(device)
        mask = batch["legal_mask"].to(device)
        if "planes" in batch:
            logits, _ = model(batch["planes"].to(device))
        else:
            logits, _ = model(batch["tokens"].to(device), batch["rows"].to(device), batch["cols"].to(device))
        bs = target.shape[0]
        top1 += topk_accuracy(logits, target, mask, 1) * bs
        top3 += topk_accuracy(logits, target, mask, 3) * bs
        top5 += topk_accuracy(logits, target, mask, 5) * bs
        n += bs
    return {"policy_top1": top1 / max(n, 1), "policy_top3": top3 / max(n, 1), "policy_top5": top5 / max(n, 1)}


@torch.no_grad()
def eval_value(model, loader, device) -> dict[str, float]:
    model.eval()
    preds = []
    targets = []
    for batch in tqdm(loader, desc="eval value"):
        target = batch["value"].to(device)
        if "planes" in batch:
            _, pred = model(batch["planes"].to(device))
        else:
            _, pred = model(batch["tokens"].to(device), batch["rows"].to(device), batch["cols"].to(device))
        preds.append(pred.detach())
        targets.append(target.detach())
    return {f"value_{k}": v for k, v in value_metrics(torch.cat(preds), torch.cat(targets)).items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="board_mamba")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--policy-val", default="data/processed/policy_val.parquet")
    parser.add_argument("--value-val", default="data/value_labels/stockfish_values_val.parquet")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output", default="experiments/results/checkpoint_eval.csv")
    args = parser.parse_args()
    device = get_device()
    model, model_name = build_model_from_checkpoint(args.checkpoint, args.model, device)
    shuffled = model_name == "board_mamba_shuffled"
    policy_ds = PolicyDataset(args.policy_val, model_name, shuffled=shuffled)
    value_ds = ValueDataset(args.value_val, model_name, shuffled=shuffled)
    metrics = {
        "checkpoint": args.checkpoint,
        "model": getattr(model, "model_name", model_name),
        **eval_policy(model, DataLoader(policy_ds, batch_size=args.batch_size), device),
        **eval_value(model, DataLoader(value_ds, batch_size=args.batch_size), device),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(out, index=False)
    print(metrics)


if __name__ == "__main__":
    main()
