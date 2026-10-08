"""Best-move agreement against Stockfish labels."""

from __future__ import annotations

import argparse
from pathlib import Path

import chess
import pandas as pd
import torch

from chess_env.action_space import action_id_to_move
from search.batched_eval import ModelEvaluator
from utils.device import get_device
from utils.model_loading import build_model_from_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", default="data/value_labels/stockfish_values.parquet")
    parser.add_argument("--model", default="board_mamba")
    parser.add_argument("--checkpoint")
    parser.add_argument("--output", default="experiments/results/best_move_agreement.csv")
    args = parser.parse_args()
    device = get_device()
    model, model_name = build_model_from_checkpoint(args.checkpoint, args.model, device)
    evaluator = ModelEvaluator(model, device, model_name)
    df = pd.read_parquet(args.labels)
    correct = 0
    rows = []
    for row in df.itertuples(index=False):
        board = chess.Board(row.fen)
        priors, _ = evaluator.evaluate(board)
        aid = max(priors, key=priors.get)
        move = action_id_to_move(board, aid)
        agree = move is not None and move.uci() == row.best_move_uci
        correct += int(agree)
        rows.append({"fen": row.fen, "pred_move_uci": move.uci() if move else None, "best_move_uci": row.best_move_uci, "agree": agree})
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print({"agreement": correct / max(len(df), 1), "positions": len(df)})


if __name__ == "__main__":
    main()
