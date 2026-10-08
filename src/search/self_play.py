"""Generate AlphaZero-style self-play data with MCTS visit targets."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import chess
import numpy as np
import pandas as pd
from tqdm import tqdm

from chess_env.action_space import move_to_action_id
from search.batched_eval import ModelEvaluator
from search.mcts import MCTS
from utils.device import get_device
from utils.model_loading import build_model_from_checkpoint
from utils.seed import set_seed


def load_start_fens(path: str | None, max_fens: int | None = None) -> list[str]:
    if not path:
        return [chess.STARTING_FEN]
    source = Path(path)
    if source.suffix == ".parquet":
        df = pd.read_parquet(source)
        if "fen" not in df.columns:
            raise ValueError(f"Start FEN parquet must contain a 'fen' column: {path}")
        fens = df["fen"].dropna().astype(str).drop_duplicates().tolist()
    else:
        fens = [line.strip() for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    if max_fens:
        fens = fens[:max_fens]
    if not fens:
        raise ValueError(f"No start FENs loaded from {path}")
    return fens


def result_value_for_side_to_move(
    result: str,
    turn: bool,
    win_value: float = 1.0,
    draw_value: float = 0.0,
    loss_value: float = -1.0,
) -> float:
    """Convert final result to value from the stored position's side-to-move perspective."""

    if result == "1/2-1/2":
        return draw_value
    white_won = result == "1-0"
    return win_value if white_won == (turn == chess.WHITE) else loss_value


def visits_to_policy(root) -> tuple[list[int], list[float]]:
    moves = []
    visits = []
    for move, child in root.children.items():
        moves.append(move)
        visits.append(child.visit_count)
    total = sum(visits)
    if total <= 0:
        prob = 1.0 / max(len(visits), 1)
        return [move_to_action_id(root.board, m) for m in moves], [prob for _ in moves]
    return [move_to_action_id(root.board, m) for m in moves], [v / total for v in visits]


def sample_move_from_visits(root, temperature: float, rng: np.random.Generator) -> chess.Move:
    children = list(root.children.values())
    if not children:
        raise ValueError("No children to sample from.")
    if temperature <= 0:
        return max(children, key=lambda child: child.visit_count).move
    visits = np.asarray([child.visit_count for child in children], dtype=np.float64)
    logits = np.log(visits + 1e-8) / temperature
    probs = np.exp(logits - logits.max())
    probs = probs / probs.sum()
    return children[int(rng.choice(len(children), p=probs))].move


def generate_self_play(
    model_name: str,
    checkpoint: str,
    output: str,
    opponent_checkpoint: str | None = None,
    opponent_model: str | None = None,
    games: int = 20,
    simulations: int = 32,
    max_plies: int = 160,
    temperature_plies: int = 16,
    seed: int = 2025,
    win_value: float = 1.0,
    draw_value: float = 0.0,
    loss_value: float = -1.0,
    record_player: str = "both",
    start_fens: str | None = None,
    max_start_fens: int | None = None,
) -> None:
    set_seed(seed)
    rng = np.random.default_rng(seed)
    device = get_device()
    model, loaded_name = build_model_from_checkpoint(checkpoint, model_name, device)
    evaluator = ModelEvaluator(model, device, loaded_name, shuffled=loaded_name == "board_mamba_shuffled")
    search = MCTS(evaluator, simulations=simulations)
    opponent_search = search
    if opponent_checkpoint:
        opponent_name = opponent_model or model_name
        opponent, loaded_opponent_name = build_model_from_checkpoint(opponent_checkpoint, opponent_name, device)
        opponent_evaluator = ModelEvaluator(
            opponent, device, loaded_opponent_name, shuffled=loaded_opponent_name == "board_mamba_shuffled"
        )
        opponent_search = MCTS(opponent_evaluator, simulations=simulations)
    start_positions = load_start_fens(start_fens, max_start_fens)

    rows = []
    for game_idx in tqdm(range(games), desc="self-play games"):
        board = chess.Board(start_positions[game_idx % len(start_positions)])
        trajectory = []
        for ply in range(max_plies):
            if board.is_game_over(claim_draw=True):
                break
            active_search = search if board.turn == chess.WHITE else opponent_search
            root = active_search.run(board)
            if not root.children:
                break
            action_ids, probs = visits_to_policy(root)
            temp = 1.0 if ply < temperature_plies else 0.0
            move = sample_move_from_visits(root, temp, rng)
            trajectory.append(
                {
                    "game_id": f"sp_{seed}_{game_idx}",
                    "ply": ply,
                    "start_fen": start_positions[game_idx % len(start_positions)],
                    "fen": board.fen(),
                    "side_to_move": "white" if board.turn == chess.WHITE else "black",
                    "action_ids": action_ids,
                    "visit_probs": probs,
                    "played_move_uci": move.uci(),
                    "played_action_id": move_to_action_id(board, move),
                    "turn": bool(board.turn),
                    "player": "main" if active_search is search else "opponent",
                }
            )
            board.push(move)

        result = board.result(claim_draw=True) if board.is_game_over(claim_draw=True) else "1/2-1/2"
        for item in trajectory:
            player = item["player"]
            if record_player != "both" and player != record_player:
                continue
            rows.append(
                {
                    **{k: v for k, v in item.items() if k != "turn"},
                    "result": result,
                    "value_target": result_value_for_side_to_move(
                        result,
                        item["turn"],
                        win_value=win_value,
                        draw_value=draw_value,
                        loss_value=loss_value,
                    ),
                    "win_value": win_value,
                    "draw_value": draw_value,
                    "loss_value": loss_value,
                }
            )

    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out, index=False)
    print({"games": games, "positions": len(rows), "output": str(out)})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="board_mamba")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--opponent-checkpoint")
    parser.add_argument("--opponent-model")
    parser.add_argument("--output", default="data/self_play/self_play.parquet")
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--simulations", type=int, default=32)
    parser.add_argument("--max-plies", type=int, default=160)
    parser.add_argument("--temperature-plies", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--win-value", type=float, default=1.0)
    parser.add_argument("--draw-value", type=float, default=0.0)
    parser.add_argument("--loss-value", type=float, default=-1.0)
    parser.add_argument("--record-player", choices=["both", "main", "opponent"], default="both")
    parser.add_argument("--start-fens")
    parser.add_argument("--max-start-fens", type=int)
    args = parser.parse_args()
    generate_self_play(
        args.model,
        args.checkpoint,
        args.output,
        args.opponent_checkpoint,
        args.opponent_model,
        args.games,
        args.simulations,
        args.max_plies,
        args.temperature_plies,
        args.seed,
        args.win_value,
        args.draw_value,
        args.loss_value,
        args.record_player,
        args.start_fens,
        args.max_start_fens,
    )


if __name__ == "__main__":
    main()
