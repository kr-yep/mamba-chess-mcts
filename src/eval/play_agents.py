"""Simple chess agents."""

from __future__ import annotations

import chess
import torch

from chess_env.action_space import action_id_to_move
from chess_env.legal_mask import make_legal_mask, mask_illegal_logits
from search.mcts import MCTS


class PolicyAgent:
    def __init__(self, evaluator):
        self.evaluator = evaluator

    def choose_move(self, board: chess.Board) -> chess.Move:
        priors, _ = self.evaluator.evaluate(board)
        action_id = max(priors, key=priors.get)
        move = action_id_to_move(board, action_id)
        if move is None:
            return next(iter(board.legal_moves))
        return move


class MCTSAgent:
    def __init__(self, evaluator, simulations: int = 32):
        self.search = MCTS(evaluator, simulations=simulations)

    def choose_move(self, board: chess.Board) -> chess.Move:
        return self.search.choose_move(board)


def play_game(white, black, max_plies: int = 300, start_fen: str | None = None) -> str:
    board = chess.Board(start_fen) if start_fen else chess.Board()
    for _ in range(max_plies):
        if board.is_game_over(claim_draw=True):
            break
        agent = white if board.turn == chess.WHITE else black
        board.push(agent.choose_move(board))
    return board.result(claim_draw=True)
