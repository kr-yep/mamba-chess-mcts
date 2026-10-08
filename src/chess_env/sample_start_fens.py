"""Sample diverse non-terminal starting FENs for self-play evaluation/training."""

from __future__ import annotations

import argparse
from pathlib import Path

import chess
import pandas as pd

from utils.seed import set_seed


def sample_start_fens(
    input_path: str,
    output_path: str,
    count: int = 200,
    min_ply: int = 8,
    max_ply: int = 40,
    seed: int = 2025,
) -> None:
    set_seed(seed)
    df = pd.read_parquet(input_path)
    needed = {"fen", "ply"}
    missing = needed.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {input_path}: {sorted(missing)}")
    sub = df[(df["ply"] >= min_ply) & (df["ply"] <= max_ply)].copy()
    sub = sub.drop_duplicates("fen")
    fens: list[str] = []
    for fen in sub.sample(frac=1.0, random_state=seed)["fen"].astype(str):
        board = chess.Board(fen)
        if board.is_game_over(claim_draw=True):
            continue
        if board.legal_moves.count() == 0:
            continue
        fens.append(fen)
        if len(fens) >= count:
            break
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(fens) + "\n", encoding="utf-8")
    print({"input": input_path, "output": output_path, "fens": len(fens), "min_ply": min_ply, "max_ply": max_ply})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/policy_train.parquet")
    parser.add_argument("--output", default="data/interim/self_play_start_fens.txt")
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--min-ply", type=int, default=8)
    parser.add_argument("--max-ply", type=int, default=40)
    parser.add_argument("--seed", type=int, default=2025)
    args = parser.parse_args()
    sample_start_fens(args.input, args.output, args.count, args.min_ply, args.max_ply, args.seed)


if __name__ == "__main__":
    main()
