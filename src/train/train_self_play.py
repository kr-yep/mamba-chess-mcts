"""Fine-tune policy-value models from MCTS self-play targets."""

from __future__ import annotations

import argparse
from pathlib import Path

import chess
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from chess_env.action_space import ACTION_SIZE
from chess_env.board_encoding import BoardEncodingConfig, encode_board_planes, encode_board_sequence
from chess_env.legal_mask import make_legal_mask, mask_illegal_logits
from train.losses import value_loss
from train.metrics import topk_accuracy, value_metrics
from utils.checkpoint import save_checkpoint
from utils.config import load_config
from utils.device import get_device
from utils.model_loading import build_model_from_checkpoint
from utils.seed import set_seed


class SelfPlayDataset(Dataset):
    def __init__(self, path: str | Path, model_type: str, shuffled: bool = False):
        self.df = pd.read_parquet(path)
        self.model_type = model_type
        self.enc_cfg = BoardEncodingConfig(shuffled_order=shuffled)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        board = chess.Board(row.fen)
        policy = torch.zeros(ACTION_SIZE, dtype=torch.float32)
        action_ids = list(row.action_ids)
        probs = list(row.visit_probs)
        if action_ids:
            policy[torch.tensor(action_ids, dtype=torch.long)] = torch.tensor(probs, dtype=torch.float32)
        item = {
            "policy": policy,
            "value": torch.tensor(float(row.value_target), dtype=torch.float32),
            "legal_mask": make_legal_mask(board),
            "hard_target": torch.tensor(int(action_ids[int(torch.tensor(probs).argmax())]) if action_ids else 0),
        }
        if self.model_type.startswith("cnn"):
            item["planes"] = torch.from_numpy(encode_board_planes(board))
        else:
            enc = encode_board_sequence(board, self.enc_cfg)
            item["tokens"] = torch.from_numpy(enc["tokens"])
            item["rows"] = torch.from_numpy(enc["rows"])
            item["cols"] = torch.from_numpy(enc["cols"])
        return item


def soft_policy_loss(logits: torch.Tensor, target_policy: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    masked = mask_illegal_logits(logits, legal_mask)
    log_probs = F.log_softmax(masked, dim=-1)
    denom = target_policy.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    target = target_policy / denom
    return -(target * log_probs).sum(dim=-1).mean()


def run_epoch(model, loader, optimizer, device, lambda_value: float, train: bool) -> dict[str, float]:
    model.train(train)
    losses = []
    top1 = top3 = top5 = n = 0.0
    preds = []
    targets = []
    for batch in tqdm(loader, desc="self-play train" if train else "self-play eval"):
        policy = batch["policy"].to(device)
        value_target = batch["value"].to(device)
        legal_mask = batch["legal_mask"].to(device)
        hard_target = batch["hard_target"].to(device)
        if "planes" in batch:
            logits, value_pred = model(batch["planes"].to(device))
        else:
            logits, value_pred = model(batch["tokens"].to(device), batch["rows"].to(device), batch["cols"].to(device))
        loss = soft_policy_loss(logits, policy, legal_mask) + lambda_value * value_loss(value_pred, value_target)
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        bs = value_target.shape[0]
        losses.append(loss.item())
        top1 += topk_accuracy(logits, hard_target, legal_mask, 1) * bs
        top3 += topk_accuracy(logits, hard_target, legal_mask, 3) * bs
        top5 += topk_accuracy(logits, hard_target, legal_mask, 5) * bs
        n += bs
        preds.append(value_pred.detach())
        targets.append(value_target.detach())
    vm = value_metrics(torch.cat(preds), torch.cat(targets))
    return {
        "loss": sum(losses) / max(len(losses), 1),
        "top1": top1 / max(n, 1),
        "top3": top3 / max(n, 1),
        "top5": top5 / max(n, 1),
        **{f"value_{k}": v for k, v in vm.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_self_play_real_mamba.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 2025))
    device = get_device()
    model_name = cfg["model"]["name"]
    model, loaded_name = build_model_from_checkpoint(cfg["init_checkpoint"], model_name, device)
    shuffled = loaded_name == "board_mamba_shuffled" or model_name == "board_mamba_shuffled"
    ds = SelfPlayDataset(cfg["data"]["self_play"], model_name, shuffled=shuffled)
    n_val = max(1, int(len(ds) * cfg["data"].get("val_frac", 0.1)))
    n_train = len(ds) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        ds, [n_train, n_val], generator=torch.Generator().manual_seed(cfg.get("seed", 2025))
    )
    loader_kwargs = {"batch_size": cfg["train"]["batch_size"], "num_workers": 0}
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"].get("weight_decay", 0.01)
    )
    rows = []
    for epoch in range(cfg["train"]["epochs"]):
        tr = run_epoch(model, train_loader, optimizer, device, cfg["train"].get("lambda_value", 1.0), True)
        va = run_epoch(model, val_loader, optimizer, device, cfg["train"].get("lambda_value", 1.0), False)
        row = {"epoch": epoch, "model": getattr(model, "model_name", model_name), **{f"train_{k}": v for k, v in tr.items()}, **{f"val_{k}": v for k, v in va.items()}}
        rows.append(row)
        print(row)
    out = Path(cfg["output"]["metrics"])
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    save_checkpoint(cfg["output"]["checkpoint"], model, optimizer, config=cfg)


if __name__ == "__main__":
    main()
