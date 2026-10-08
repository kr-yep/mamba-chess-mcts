import chess

from chess_env.action_space import move_to_action_id
from search.mcts import MCTS


class DummyEvaluator:
    def evaluate(self, board):
        moves = list(board.legal_moves)
        p = 1.0 / len(moves) if moves else 0.0
        return {move_to_action_id(board, move): p for move in moves}, 0.0


def test_mcts_returns_legal_move():
    board = chess.Board()
    move = MCTS(DummyEvaluator(), simulations=4).choose_move(board)
    assert move in board.legal_moves


def test_mcts_terminal_state_raises_cleanly():
    board = chess.Board("7k/5Q2/7K/8/8/8/8/8 b - - 0 1")
    try:
        MCTS(DummyEvaluator(), simulations=4).choose_move(board)
    except ValueError as exc:
        assert "No legal moves" in str(exc)
    else:
        raise AssertionError("Expected terminal board to have no move")
