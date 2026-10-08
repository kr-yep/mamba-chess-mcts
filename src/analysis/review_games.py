"""Stockfish review for self-play games.

The reviewer looks for draw/loss positions where the played move appears to
give up value compared with Stockfish's preferred move. The output can be used
as a small corrective policy dataset.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import chess
import chess.engine
import pandas as pd
from tqdm import tqdm

from chess_env.action_space import move_to_action_id


def score_to_value(score: chess.engine.PovScore, turn: chess.Color) -> tuple[int | None, int | None, float]:
    pov = score.pov(turn)
    mate = pov.mate()
    if mate is not None:
        return None, mate, 1.0 if mate > 0 else -1.0
    cp = pov.score(mate_score=100000)
    if cp is None:
        return None, None, 0.0
    return cp, None, math.tanh(cp / 400.0)


def game_is_bad_or_drawish(row: pd.Series) -> bool:
    result = row.get("result")
    side = row.get("side_to_move")
    if result == "1/2-1/2":
        return True
    if result == "1-0" and side == "black":
        return True
    if result == "0-1" and side == "white":
        return True
    return False


def review_positions(
    input_path: str,
    output_path: str,
    policy_output_path: str | None,
    stockfish_path: str,
    limit_positions: int = 1000,
    depth: int = 8,
    min_value_gain: float = 0.15,
) -> None:
    df = pd.read_parquet(input_path)
    if "played_move_uci" not in df.columns:
        raise ValueError(
            "Self-play data does not contain played_move_uci. Regenerate self-play data with the updated generator."
        )
    candidates = df[df.apply(game_is_bad_or_drawish, axis=1)].head(limit_positions)
    limit = chess.engine.Limit(depth=depth)
    rows: list[dict] = []
    policy_rows: list[dict] = []

    with chess.engine.SimpleEngine.popen_uci(stockfish_path) as engine:
        for row in tqdm(candidates.itertuples(index=False), total=len(candidates), desc="review"):
            board = chess.Board(row.fen)
            if board.is_game_over(claim_draw=True):
                continue
            played = chess.Move.from_uci(row.played_move_uci)
            if played not in board.legal_moves:
                continue

            current_info = engine.analyse(board, limit)
            current_cp, current_mate, current_value = score_to_value(current_info["score"], board.turn)
            best = engine.play(board, limit).move
            if best is None or best == played:
                continue

            played_board = board.copy(stack=False)
            played_board.push(played)
            played_info = engine.analyse(played_board, limit)
            _, _, played_reply_value = score_to_value(played_info["score"], played_board.turn)
            played_value = -played_reply_value

            best_board = board.copy(stack=False)
            best_board.push(best)
            best_info = engine.analyse(best_board, limit)
            _, _, best_reply_value = score_to_value(best_info["score"], best_board.turn)
            best_value = -best_reply_value

            gain = best_value - played_value
            if gain < min_value_gain:
                continue

            best_action_id = move_to_action_id(board, best)
            rows.append(
                {
                    "game_id": row.game_id,
                    "ply": row.ply,
                    "fen": row.fen,
                    "side_to_move": row.side_to_move,
                    "result": row.result,
                    "played_move_uci": row.played_move_uci,
                    "best_move_uci": best.uci(),
                    "best_action_id": best_action_id,
                    "stockfish_depth": depth,
                    "current_centipawn": current_cp,
                    "current_mate_score": current_mate,
                    "current_value": current_value,
                    "played_value": played_value,
                    "best_value": best_value,
                    "value_gain": gain,
                }
            )
            policy_rows.append(
                {
                    "game_id": f"review_{row.game_id}",
                    "ply": row.ply,
                    "fen": row.fen,
                    "side_to_move": row.side_to_move,
                    "legal_moves_uci": " ".join(move.uci() for move in board.legal_moves),
                    "played_move_uci": best.uci(),
                    "played_action_id": best_action_id,
                    "result": row.result,
                    "white_elo": None,
                    "black_elo": None,
                    "time_control": "self_play_review",
                    "review_value_gain": gain,
                }
            )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out, index=False)
    pd.DataFrame(rows).to_csv(out.with_suffix(".csv"), index=False)
    if policy_output_path:
        policy_out = Path(policy_output_path)
        policy_out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(policy_rows).to_parquet(policy_out, index=False)
    print({"reviewed": len(candidates), "critical_positions": len(rows), "output": output_path})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/self_play/self_play_real_mamba_reward_mild.parquet")
    parser.add_argument("--output", default="experiments/results/review_reward_mild.parquet")
    parser.add_argument("--policy-output", default="data/processed/review_policy.parquet")
    parser.add_argument("--stockfish-path", default=os.environ.get("STOCKFISH_PATH"))
    parser.add_argument("--limit-positions", type=int, default=1000)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--min-value-gain", type=float, default=0.15)
    args = parser.parse_args()
    if not args.stockfish_path:
        raise SystemExit("Set --stockfish-path or STOCKFISH_PATH.")
    review_positions(
        args.input,
        args.output,
        args.policy_output,
        args.stockfish_path,
        args.limit_positions,
        args.depth,
        args.min_value_gain,
    )


if __name__ == "__main__":
    main()
