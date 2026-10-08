"""Policy imitation training."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

import chess
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from chess_env.board_encoding import BoardEncodingConfig, encode_board_planes, encode_board_sequence
from chess_env.legal_mask import make_legal_mask
from models.model_factory import build_model
from train.losses import masked_policy_loss
from train.metrics import legal_move_rate, topk_accuracy
from utils.checkpoint import save_checkpoint
from utils.config import load_config
from utils.device import get_device
from utils.seed import set_seed


class PolicyDataset(Dataset):
    def __init__(self, path: str | Path, model_type: str, shuffled: bool = False):
        self.df = pd.read_parquet(path)
        self.model_type = model_type
        self.enc_cfg = BoardEncodingConfig(shuffled_order=shuffled)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        board = chess.Board(row.fen)
        target = int(row.played_action_id)
        mask = make_legal_mask(board)
        item = {"target": torch.tensor(target), "legal_mask": mask}
        if self.model_type.startswith("cnn"):
            item["planes"] = torch.from_numpy(encode_board_planes(board))
        else:
            enc = encode_board_sequence(board, self.enc_cfg)
            item["tokens"] = torch.from_numpy(enc["tokens"])
            item["rows"] = torch.from_numpy(enc["rows"])
            item["cols"] = torch.from_numpy(enc["cols"])
        return item


def run_epoch(model, loader, optimizer, device, train: bool) -> dict[str, float]:
    model.train(train)
    total_loss = total_top1 = total_top3 = total_top5 = total_legal = n = 0
    for batch in tqdm(loader, desc="train" if train else "eval"):
        target = batch["target"].to(device)
        legal_mask = batch["legal_mask"].to(device)
        if "planes" in batch:
            logits, _ = model(batch["planes"].to(device))
        else:
            logits, _ = model(batch["tokens"].to(device), batch["rows"].to(device), batch["cols"].to(device))
        loss = masked_policy_loss(logits, target, legal_mask)
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        bs = target.shape[0]
        total_loss += loss.item() * bs
        total_top1 += topk_accuracy(logits, target, legal_mask, 1) * bs
        total_top3 += topk_accuracy(logits, target, legal_mask, 3) * bs
        total_top5 += topk_accuracy(logits, target, legal_mask, 5) * bs
        total_legal += legal_move_rate(logits, legal_mask) * bs
        n += bs
    return {
        "loss": total_loss / max(n, 1),
        "top1": total_top1 / max(n, 1),
        "top3": total_top3 / max(n, 1),
        "top5": total_top5 / max(n, 1),
        "legal_move_rate": total_legal / max(n, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_board_mamba.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 2025))
    device = get_device()
    model_name = cfg["model"]["name"]
    model = build_model(model_name, **cfg["model"].get("kwargs", {})).to(device)
    shuffled = model_name == "board_mamba_shuffled"
    train_ds = PolicyDataset(cfg["data"]["train"], model_name, shuffled=shuffled)
    val_ds = PolicyDataset(cfg["data"]["val"], model_name, shuffled=shuffled)
    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"].get("weight_decay", 0.01))
    results = []
    for epoch in range(cfg["train"]["epochs"]):
        train_metrics = run_epoch(model, train_loader, optimizer, device, True)
        val_metrics = run_epoch(model, val_loader, optimizer, device, False)
        row = {"epoch": epoch, "model": getattr(model, "model_name", model_name), **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}}
        results.append(row)
        print(row)
    out = Path(cfg["output"]["metrics"])
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(out, index=False)
    save_checkpoint(cfg["output"]["checkpoint"], model, optimizer, config=cfg)


if __name__ == "__main__":
    main()
