"""Joint policy-value training."""

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
from train.train_value import ValueDataset
from utils.checkpoint import save_checkpoint
from utils.config import load_config
from utils.device import get_device
from utils.model_loading import build_model_from_checkpoint
from utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_joint_real_mamba.yaml")
    parser.add_argument("--init-checkpoint")
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 2025))
    device = get_device()

    model_name = cfg["model"]["name"]
    checkpoint = args.init_checkpoint or cfg.get("init_checkpoint")
    model, loaded_name = build_model_from_checkpoint(checkpoint, model_name, device)
    shuffled = loaded_name == "board_mamba_shuffled" or model_name == "board_mamba_shuffled"

    policy_train = PolicyDataset(cfg["data"]["train"], model_name, shuffled=shuffled)
    policy_val = PolicyDataset(cfg["data"]["val"], model_name, shuffled=shuffled)
    value_train = ValueDataset(cfg["data"]["value_train"], model_name, shuffled=shuffled)
    value_val = ValueDataset(cfg["data"]["value_val"], model_name, shuffled=shuffled)

    batch_size = cfg["train"]["batch_size"]
    policy_loader = DataLoader(policy_train, batch_size=batch_size, shuffle=True, num_workers=0)
    value_loader = DataLoader(value_train, batch_size=batch_size, shuffle=True, num_workers=0)
    policy_val_loader = DataLoader(policy_val, batch_size=batch_size, shuffle=False, num_workers=0)
    value_val_loader = DataLoader(value_val, batch_size=batch_size, shuffle=False, num_workers=0)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"].get("weight_decay", 0.01)
    )
    lambda_value = cfg["train"].get("lambda_value", 0.5)

    rows = []
    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        train_losses = []
        value_iter = itertools.cycle(value_loader)
        for p_batch in tqdm(policy_loader, desc="joint train"):
            v_batch = next(value_iter)

            p_target = p_batch["target"].to(device)
            p_mask = p_batch["legal_mask"].to(device)
            if "planes" in p_batch:
                p_logits, _ = model(p_batch["planes"].to(device))
                _, v_pred = model(v_batch["planes"].to(device))
            else:
                p_logits, _ = model(
                    p_batch["tokens"].to(device), p_batch["rows"].to(device), p_batch["cols"].to(device)
                )
                _, v_pred = model(
                    v_batch["tokens"].to(device), v_batch["rows"].to(device), v_batch["cols"].to(device)
                )
            v_target = v_batch["value"].to(device)
            loss = masked_policy_loss(p_logits, p_target, p_mask) + lambda_value * value_loss(v_pred, v_target)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        p_top1 = p_top3 = p_top5 = p_n = 0.0
        v_preds = []
        v_targets = []
        with torch.no_grad():
            for p_batch in tqdm(policy_val_loader, desc="joint policy val"):
                target = p_batch["target"].to(device)
                mask = p_batch["legal_mask"].to(device)
                if "planes" in p_batch:
                    logits, _ = model(p_batch["planes"].to(device))
                else:
                    logits, _ = model(
                        p_batch["tokens"].to(device), p_batch["rows"].to(device), p_batch["cols"].to(device)
                    )
                bs = target.shape[0]
                p_top1 += topk_accuracy(logits, target, mask, 1) * bs
                p_top3 += topk_accuracy(logits, target, mask, 3) * bs
                p_top5 += topk_accuracy(logits, target, mask, 5) * bs
                p_n += bs
            for v_batch in tqdm(value_val_loader, desc="joint value val"):
                target = v_batch["value"].to(device)
                if "planes" in v_batch:
                    _, pred = model(v_batch["planes"].to(device))
                else:
                    _, pred = model(v_batch["tokens"].to(device), v_batch["rows"].to(device), v_batch["cols"].to(device))
                v_preds.append(pred.detach())
                v_targets.append(target.detach())

        vm = value_metrics(torch.cat(v_preds), torch.cat(v_targets))
        row = {
            "epoch": epoch,
            "model": getattr(model, "model_name", model_name),
            "train_joint_loss": sum(train_losses) / max(len(train_losses), 1),
            "val_top1": p_top1 / max(p_n, 1),
            "val_top3": p_top3 / max(p_n, 1),
            "val_top5": p_top5 / max(p_n, 1),
            **{f"val_value_{k}": v for k, v in vm.items()},
        }
        rows.append(row)
        print(row)

    out = Path(cfg["output"]["joint_metrics"])
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    save_checkpoint(cfg["output"]["joint_checkpoint"], model, optimizer, config=cfg)


if __name__ == "__main__":
    main()
