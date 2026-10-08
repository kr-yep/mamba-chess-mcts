"""Measure MCTS move timing."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import chess
import pandas as pd

from search.batched_eval import ModelEvaluator
from search.mcts import MCTS
from utils.device import get_device
from utils.model_loading import build_model_from_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="board_mamba")
    parser.add_argument("--checkpoint")
    parser.add_argument("--simulations", nargs="+", type=int, default=[32, 64])
    parser.add_argument("--positions", type=int, default=8)
    parser.add_argument("--output", default="experiments/results/mcts_timing.csv")
    args = parser.parse_args()

    device = get_device()
    model, model_name = build_model_from_checkpoint(args.checkpoint, args.model, device)
    evaluator = ModelEvaluator(model, device, model_name, shuffled=model_name == "board_mamba_shuffled")

    boards = []
    board = chess.Board()
    for move in list(board.legal_moves)[: args.positions]:
        b = chess.Board()
        b.push(move)
        boards.append(b)

    rows = []
    for sims in args.simulations:
        search = MCTS(evaluator, simulations=sims)
        start = time.perf_counter()
        for b in boards:
            search.choose_move(b)
        elapsed = time.perf_counter() - start
        rows.append(
            {
                "model": getattr(model, "model_name", model_name),
                "simulations": sims,
                "positions": len(boards),
                "total_time_sec": elapsed,
                "avg_move_time": elapsed / max(len(boards), 1),
            }
        )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
