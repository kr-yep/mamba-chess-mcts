"""Head-to-head evaluation between two checkpoints or agent settings."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

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


def build_agent(model_name: str, checkpoint: str, agent_type: str, simulations: int, device):
    model, loaded_name = build_model_from_checkpoint(checkpoint, model_name, device)
    evaluator = ModelEvaluator(model, device, loaded_name, shuffled=loaded_name == "board_mamba_shuffled")
    if agent_type == "policy":
        return PolicyAgent(evaluator), getattr(model, "model_name", loaded_name)
    if agent_type == "mcts":
        return MCTSAgent(evaluator, simulations=simulations), getattr(model, "model_name", loaded_name)
    raise ValueError(f"Unknown agent type: {agent_type}")


def score_for_a(row: pd.Series) -> float:
    if row["result"] == "1/2-1/2":
        return 0.5
    if row["result"] == "1-0":
        return 1.0 if row["white"] == "A" else 0.0
    if row["result"] == "0-1":
        return 1.0 if row["black"] == "A" else 0.0
    return 0.5


def summarize(df: pd.DataFrame) -> dict:
    scores = df.apply(score_for_a, axis=1)
    wins = int((scores == 1.0).sum())
    draws = int((scores == 0.5).sum())
    losses = int((scores == 0.0).sum())
    score = float(scores.mean()) if len(scores) else 0.0
    lo, hi = wilson_interval(float(scores.sum()), len(scores))
    return {
        "games": len(scores),
        "a_wins": wins,
        "draws": draws,
        "a_losses": losses,
        "a_score": score,
        "a_score_ci_low": lo,
        "a_score_ci_high": hi,
        "elo_diff_a_minus_b": score_to_elo(score),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-a", default="board_mamba")
    parser.add_argument("--checkpoint-a", required=True)
    parser.add_argument("--agent-a", choices=["policy", "mcts"], default="mcts")
    parser.add_argument("--model-b", default="board_mamba")
    parser.add_argument("--checkpoint-b", required=True)
    parser.add_argument("--agent-b", choices=["policy", "mcts"], default="mcts")
    parser.add_argument("--games", type=int, default=50)
    parser.add_argument("--simulations-a", type=int, default=32)
    parser.add_argument("--simulations-b", type=int, default=32)
    parser.add_argument("--start-fens")
    parser.add_argument("--max-start-fens", type=int)
    parser.add_argument("--max-plies", type=int, default=300)
    parser.add_argument("--output", default="experiments/results/head_to_head.csv")
    args = parser.parse_args()

    device = get_device()
    agent_a, loaded_a = build_agent(args.model_a, args.checkpoint_a, args.agent_a, args.simulations_a, device)
    agent_b, loaded_b = build_agent(args.model_b, args.checkpoint_b, args.agent_b, args.simulations_b, device)
    start_fens = load_start_fens(args.start_fens, args.max_start_fens)

    rows = []
    start = time.perf_counter()
    for game_idx in range(args.games):
        a_white = game_idx % 2 == 0
        start_fen = start_fens[game_idx % len(start_fens)]
        result = (
            play_game(agent_a, agent_b, max_plies=args.max_plies, start_fen=start_fen)
            if a_white
            else play_game(agent_b, agent_a, max_plies=args.max_plies, start_fen=start_fen)
        )
        rows.append(
            {
                "game": game_idx,
                "start_fen": start_fen or "startpos",
                "white": "A" if a_white else "B",
                "black": "B" if a_white else "A",
                "result": result,
                "model_a": loaded_a,
                "agent_a": args.agent_a,
                "checkpoint_a": args.checkpoint_a,
                "model_b": loaded_b,
                "agent_b": args.agent_b,
                "checkpoint_b": args.checkpoint_b,
            }
        )
    elapsed = time.perf_counter() - start

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    summary = summarize(df)
    summary["elapsed_sec"] = elapsed
    summary["model_a"] = loaded_a
    summary["agent_a"] = args.agent_a
    summary["checkpoint_a"] = args.checkpoint_a
    summary["model_b"] = loaded_b
    summary["agent_b"] = args.agent_b
    summary["checkpoint_b"] = args.checkpoint_b
    pd.DataFrame([summary]).to_csv(out.with_name(f"{out.stem}_summary.csv"), index=False)
    print(summary)


if __name__ == "__main__":
    main()
