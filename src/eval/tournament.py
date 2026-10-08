"""Internal tournament runner."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import torch

from eval.confidence_interval import wilson_interval
from eval.elo_estimator import score_to_elo
from eval.play_agents import MCTSAgent, PolicyAgent, play_game
from search.batched_eval import ModelEvaluator
from utils.device import get_device
from utils.model_loading import build_model_from_checkpoint


def load_start_fens(path: str | None, max_fens: int | None = None) -> list[str | None]:
    if not path:
        return [None]
    lines = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if max_fens:
        lines = lines[:max_fens]
    if not lines:
        raise ValueError(f"No start FENs loaded from {path}")
    return lines


def _agent_score(row: pd.Series, agent: str) -> float:
    if row["result"] == "1/2-1/2":
        return 0.5
    if row["result"] == "1-0":
        return 1.0 if row["white"] == agent else 0.0
    if row["result"] == "0-1":
        return 1.0 if row["black"] == agent else 0.0
    return 0.5


def summarize(df: pd.DataFrame, agent: str = "mcts") -> dict:
    scores = df.apply(lambda row: _agent_score(row, agent), axis=1)
    wins = int((scores == 1.0).sum())
    draws = int((scores == 0.5).sum())
    losses = int((scores == 0.0).sum())
    n = len(df)
    score = float(scores.sum() / max(n, 1))
    lo, hi = wilson_interval(float(scores.sum()), n)
    return {
        "games": n,
        "agent": agent,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "score": score,
        "score_ci_low": lo,
        "score_ci_high": hi,
        "elo_diff": score_to_elo(score),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="board_mamba")
    parser.add_argument("--checkpoint")
    parser.add_argument("--games", type=int, default=50)
    parser.add_argument("--simulations", type=int, default=32)
    parser.add_argument("--start-fens")
    parser.add_argument("--max-start-fens", type=int)
    parser.add_argument("--max-plies", type=int, default=300)
    parser.add_argument("--output", default="experiments/results/tournament_results.csv")
    args = parser.parse_args()
    device = get_device()
    model, model_name = build_model_from_checkpoint(args.checkpoint, args.model, device)
    evaluator = ModelEvaluator(model, device, model_name, shuffled=model_name == "board_mamba_shuffled")
    policy = PolicyAgent(evaluator)
    mcts = MCTSAgent(evaluator, simulations=args.simulations)
    start_fens = load_start_fens(args.start_fens, args.max_start_fens)
    rows = []
    start = time.perf_counter()
    for i in range(args.games):
        start_fen = start_fens[i % len(start_fens)]
        result = (
            play_game(mcts, policy, max_plies=args.max_plies, start_fen=start_fen)
            if i % 2 == 0
            else play_game(policy, mcts, max_plies=args.max_plies, start_fen=start_fen)
        )
        rows.append(
            {
                "game": i,
                "start_fen": start_fen or "startpos",
                "white": "mcts" if i % 2 == 0 else "policy",
                "black": "policy" if i % 2 == 0 else "mcts",
                "result": result,
            }
        )
    elapsed = time.perf_counter() - start
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    summary = summarize(df, agent="mcts")
    summary["elapsed_sec"] = elapsed
    pd.DataFrame([summary]).to_csv(out.with_name("tournament_summary.csv"), index=False)
    print(summary)


if __name__ == "__main__":
    main()
