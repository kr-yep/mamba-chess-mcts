"""Value prediction training on Stockfish labels."""

from __future__ import annotations

import argparse
from pathlib import Path

import chess
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from chess_env.board_encoding import BoardEncodingConfig, encode_board_planes, encode_board_sequence
from models.model_factory import build_model
from train.losses import value_loss
from train.metrics import value_metrics
from utils.checkpoint import save_checkpoint
from utils.config import load_config
from utils.device import get_device
from utils.seed import set_seed


class ValueDataset(Dataset):
    def __init__(self, path: str | Path, model_type: str, shuffled: bool = False):
        self.df = pd.read_parquet(path)
        self.model_type = model_type
        self.enc_cfg = BoardEncodingConfig(shuffled_order=shuffled)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        board = chess.Board(row.fen)
        item = {"value": torch.tensor(float(row.value), dtype=torch.float32)}
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
    losses = []
    preds = []
    targets = []
    for batch in tqdm(loader, desc="train" if train else "eval"):
        target = batch["value"].to(device)
        if "planes" in batch:
            _, value = model(batch["planes"].to(device))
        else:
            _, value = model(batch["tokens"].to(device), batch["rows"].to(device), batch["cols"].to(device))
        loss = value_loss(value, target)
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        losses.append(loss.item())
        preds.append(value.detach())
        targets.append(target.detach())
    metrics = value_metrics(torch.cat(preds), torch.cat(targets))
    metrics["loss"] = float(sum(losses) / max(len(losses), 1))
    return metrics


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
    train_ds = ValueDataset(cfg["data"]["value_train"], model_name, shuffled=shuffled)
    val_ds = ValueDataset(cfg["data"]["value_val"], model_name, shuffled=shuffled)
    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"])
    results = []
    for epoch in range(cfg["train"]["epochs"]):
        tr = run_epoch(model, train_loader, optimizer, device, True)
        va = run_epoch(model, val_loader, optimizer, device, False)
        row = {"epoch": epoch, "model": getattr(model, "model_name", model_name), **{f"train_{k}": v for k, v in tr.items()}, **{f"val_{k}": v for k, v in va.items()}}
        results.append(row)
        print(row)
    out = Path(cfg["output"].get("value_metrics", "experiments/results/value_metrics.csv"))
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(out, index=False)
    checkpoint_path = cfg["output"].get("value_checkpoint", cfg["output"]["checkpoint"])
    save_checkpoint(checkpoint_path, model, optimizer, config=cfg)


if __name__ == "__main__":
    main()
