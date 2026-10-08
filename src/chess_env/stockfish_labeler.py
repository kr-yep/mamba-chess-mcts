"""Stockfish value-label generation."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import chess
import chess.engine
import pandas as pd
from tqdm import tqdm


def pov_score_to_value(score: chess.engine.PovScore) -> tuple[int | None, int | None, float]:
    """Return centipawn, mate score, and tanh value from side-to-move perspective."""

    pov = score.pov(score.turn)
    mate = pov.mate()
    if mate is not None:
        value = 1.0 if mate > 0 else -1.0
        return None, mate, value

    cp = pov.score(mate_score=100000)
    if cp is None:
        return None, None, 0.0
    return cp, None, math.tanh(cp / 400.0)


def label_positions(
    input_path: str | Path,
    output_path: str | Path,
    stockfish_path: str,
    limit_positions: int = 50000,
    depth: int = 10,
    time_limit: float | None = None,
) -> None:
    df = pd.read_parquet(input_path).drop_duplicates("fen").head(limit_positions)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    limit = chess.engine.Limit(depth=depth) if time_limit is None else chess.engine.Limit(time=time_limit)
    with chess.engine.SimpleEngine.popen_uci(stockfish_path) as engine:
        for row in tqdm(df.itertuples(index=False), total=len(df), desc="Stockfish labels"):
            board = chess.Board(row.fen)
            info = engine.analyse(board, limit)
            score = info["score"]
            cp, mate, value = pov_score_to_value(score)
            best = engine.play(board, limit).move
            wdl = score.pov(board.turn).wdl(model="sf")
            rows.append(
                {
                    "fen": row.fen,
                    "stockfish_depth": depth,
                    "stockfish_nodes": info.get("nodes"),
                    "centipawn": cp,
                    "mate_score": mate,
                    "value": value,
                    "wdl_win": wdl.wins,
                    "wdl_draw": wdl.draws,
                    "wdl_loss": wdl.losses,
                    "best_move_uci": best.uci() if best else None,
                }
            )
    pd.DataFrame(rows).to_parquet(output_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/policy_train.parquet")
    parser.add_argument("--output", default="data/value_labels/stockfish_values.parquet")
    parser.add_argument("--stockfish-path", default=os.environ.get("STOCKFISH_PATH"))
    parser.add_argument("--limit-positions", type=int, default=50000)
    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument("--time-limit", type=float)
    args = parser.parse_args()
    if not args.stockfish_path:
        raise SystemExit("Set --stockfish-path or STOCKFISH_PATH.")
    label_positions(
        args.input,
        args.output,
        args.stockfish_path,
        args.limit_positions,
        args.depth,
        args.time_limit,
    )


if __name__ == "__main__":
    main()
