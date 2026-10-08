"""PUCT Monte Carlo Tree Search."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import chess
import torch

from chess_env.action_space import action_id_to_move, move_to_action_id
from models.model_factory import build_model
from search.batched_eval import ModelEvaluator
from search.cache import EvalCache
from search.node import Node
from search.puct import puct_score
from utils.device import get_device
from utils.model_loading import build_model_from_checkpoint


def terminal_value(board: chess.Board) -> float | None:
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return None
    if outcome.winner is None:
        return 0.0
    return 1.0 if outcome.winner == board.turn else -1.0


class MCTS:
    def __init__(self, evaluator: ModelEvaluator, simulations: int = 32, c_puct: float = 1.5):
        self.evaluator = evaluator
        self.simulations = simulations
        self.c_puct = c_puct
        self.cache = EvalCache()

    def select_child(self, node: Node) -> Node:
        return max(
            node.children.values(),
            key=lambda child: puct_score(-child.q, child.prior, node.visit_count, child.visit_count, self.c_puct),
        )

    def expand(self, node: Node) -> float:
        tv = terminal_value(node.board)
        if tv is not None:
            return tv
        key = node.board.transposition_key() if hasattr(node.board, "transposition_key") else node.board.fen()
        cached = self.cache.get(str(key))
        if cached is None:
            priors, value = self.evaluator.evaluate(node.board)
            self.cache.put(str(key), (priors, value))
        else:
            priors, value = cached
        for action_id, prior in priors.items():
            move = action_id_to_move(node.board, action_id)
            if move is None:
                continue
            child_board = node.board.copy(stack=True)
            child_board.push(move)
            node.children[move] = Node(child_board, prior=prior, move=move, parent=node)
        return value

    def run(self, board: chess.Board) -> Node:
        root = Node(board.copy(stack=True), prior=1.0)
        self.expand(root)
        for _ in range(self.simulations):
            node = root
            while node.is_expanded:
                node = self.select_child(node)
            value = self.expand(node)
            node.backup(value)
        return root

    def choose_move(self, board: chess.Board, temperature: float = 0.0) -> chess.Move:
        root = self.run(board)
        if not root.children:
            raise ValueError("No legal moves available.")
        if temperature == 0:
            return max(root.children.values(), key=lambda child: child.visit_count).move
        visits = torch.tensor([child.visit_count for child in root.children.values()], dtype=torch.float32)
        probs = torch.softmax(torch.log(visits + 1e-8) / temperature, dim=0)
        idx = int(torch.multinomial(probs, 1).item())
        return list(root.children.values())[idx].move


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="board_mamba")
    parser.add_argument("--checkpoint")
    parser.add_argument("--fen", default=chess.STARTING_FEN)
    parser.add_argument("--simulations", type=int, default=32)
    args = parser.parse_args()
    device = get_device()
    model, model_name = build_model_from_checkpoint(args.checkpoint, args.model, device)
    evaluator = ModelEvaluator(model, device, model_name, shuffled=model_name == "board_mamba_shuffled")
    move = MCTS(evaluator, simulations=args.simulations).choose_move(chess.Board(args.fen))
    print(move.uci())


if __name__ == "__main__":
    main()
