"""Fine-tune with both MCTS self-play visit targets and Stockfish review targets."""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from train.losses import masked_policy_loss, value_loss
from train.metrics import topk_accuracy, value_metrics
from train.train_policy import PolicyDataset
from train.train_self_play import SelfPlayDataset, soft_policy_loss
from utils.checkpoint import save_checkpoint
from utils.config import load_config
from utils.device import get_device
from utils.model_loading import build_model_from_checkpoint
from utils.seed import set_seed


def forward(model, batch, device):
    if "planes" in batch:
        return model(batch["planes"].to(device))
    return model(batch["tokens"].to(device), batch["rows"].to(device), batch["cols"].to(device))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_mixed_self_play_review.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 2029))
    device = get_device()

    model_name = cfg["model"]["name"]
    model, loaded_name = build_model_from_checkpoint(cfg["init_checkpoint"], model_name, device)
    shuffled = loaded_name == "board_mamba_shuffled" or model_name == "board_mamba_shuffled"
    sp_ds = SelfPlayDataset(cfg["data"]["self_play"], model_name, shuffled=shuffled)
    review_ds = PolicyDataset(cfg["data"]["review_policy"], model_name, shuffled=shuffled)

    batch_size = cfg["train"]["batch_size"]
    sp_loader = DataLoader(sp_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    review_loader = DataLoader(review_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"].get("weight_decay", 0.01)
    )
    lambda_value = cfg["train"].get("lambda_value", 0.25)
    review_weight = cfg["train"].get("review_weight", 0.25)

    rows = []
    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        review_iter = itertools.cycle(review_loader)
        losses = []
        sp_top1 = sp_top3 = sp_top5 = sp_n = 0.0
        review_top1 = review_top3 = review_top5 = review_n = 0.0
        preds = []
        targets = []
        for sp_batch in tqdm(sp_loader, desc="mixed train"):
            rv_batch = next(review_iter)

            sp_policy = sp_batch["policy"].to(device)
            sp_value_target = sp_batch["value"].to(device)
            sp_mask = sp_batch["legal_mask"].to(device)
            sp_hard = sp_batch["hard_target"].to(device)
            sp_logits, sp_value = forward(model, sp_batch, device)

            rv_target = rv_batch["target"].to(device)
            rv_mask = rv_batch["legal_mask"].to(device)
            rv_logits, _ = forward(model, rv_batch, device)

            loss = (
                soft_policy_loss(sp_logits, sp_policy, sp_mask)
                + lambda_value * value_loss(sp_value, sp_value_target)
                + review_weight * masked_policy_loss(rv_logits, rv_target, rv_mask)
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            bs = sp_value_target.shape[0]
            sp_top1 += topk_accuracy(sp_logits, sp_hard, sp_mask, 1) * bs
            sp_top3 += topk_accuracy(sp_logits, sp_hard, sp_mask, 3) * bs
            sp_top5 += topk_accuracy(sp_logits, sp_hard, sp_mask, 5) * bs
            sp_n += bs
            preds.append(sp_value.detach())
            targets.append(sp_value_target.detach())

            rbs = rv_target.shape[0]
            review_top1 += topk_accuracy(rv_logits, rv_target, rv_mask, 1) * rbs
            review_top3 += topk_accuracy(rv_logits, rv_target, rv_mask, 3) * rbs
            review_top5 += topk_accuracy(rv_logits, rv_target, rv_mask, 5) * rbs
            review_n += rbs

        vm = value_metrics(torch.cat(preds), torch.cat(targets))
        row = {
            "epoch": epoch,
            "model": getattr(model, "model_name", loaded_name),
            "train_loss": sum(losses) / max(len(losses), 1),
            "self_play_top1": sp_top1 / max(sp_n, 1),
            "self_play_top3": sp_top3 / max(sp_n, 1),
            "self_play_top5": sp_top5 / max(sp_n, 1),
            "review_top1": review_top1 / max(review_n, 1),
            "review_top3": review_top3 / max(review_n, 1),
            "review_top5": review_top5 / max(review_n, 1),
            **{f"self_play_value_{k}": v for k, v in vm.items()},
        }
        rows.append(row)
        print(row)

    out = Path(cfg["output"]["metrics"])
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    save_checkpoint(cfg["output"]["checkpoint"], model, optimizer, config=cfg)


if __name__ == "__main__":
    main()
