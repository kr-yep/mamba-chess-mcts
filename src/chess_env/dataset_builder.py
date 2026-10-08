"""Build train/val/test policy datasets with whole-game splits."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def split_by_game(
    df: pd.DataFrame, seed: int = 2025, val_frac: float = 0.1, test_frac: float = 0.1
) -> dict[str, pd.DataFrame]:
    game_ids = df["game_id"].drop_duplicates().to_numpy()
    rng = np.random.default_rng(seed)
    rng.shuffle(game_ids)
    n = len(game_ids)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    test_ids = set(game_ids[:n_test])
    val_ids = set(game_ids[n_test : n_test + n_val])

    splits = {
        "test": df[df["game_id"].isin(test_ids)].copy(),
        "val": df[df["game_id"].isin(val_ids)].copy(),
        "train": df[~df["game_id"].isin(test_ids | val_ids)].copy(),
    }
    return splits


def build_policy_dataset(input_path: str | Path, output_dir: str | Path, seed: int = 2025) -> None:
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(input_path)
    splits = split_by_game(df, seed=seed)
    for name, split_df in splits.items():
        split_df.to_parquet(output_dir / f"policy_{name}.parquet", index=False)


def split_value_labels(
    input_path: str | Path,
    output_dir: str | Path,
    seed: int = 2025,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
) -> None:
    """Split Stockfish value labels by row after duplicate FEN removal."""

    df = pd.read_parquet(input_path).drop_duplicates("fen").reset_index(drop=True)
    rng = np.random.default_rng(seed)
    indices = np.arange(len(df))
    rng.shuffle(indices)
    n_test = int(len(indices) * test_frac)
    n_val = int(len(indices) * val_frac)
    split_indices = {
        "test": indices[:n_test],
        "val": indices[n_test : n_test + n_val],
        "train": indices[n_test + n_val :],
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, idx in split_indices.items():
        df.iloc[idx].to_parquet(output_dir / f"stockfish_values_{name}.parquet", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/interim/policy_positions.parquet")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--value-labels")
    args = parser.parse_args()
    if args.value_labels:
        split_value_labels(args.value_labels, Path(args.value_labels).parent, args.seed)
    else:
        build_policy_dataset(args.input, args.output_dir, args.seed)


if __name__ == "__main__":
    main()
