import chess
import chess.engine

from chess_env.stockfish_labeler import pov_score_to_value


def test_cp_value_from_side_to_move():
    score = chess.engine.PovScore(chess.engine.Cp(400), chess.WHITE)
    cp, mate, value = pov_score_to_value(score)
    assert cp == 400
    assert mate is None
    assert 0.7 < value < 0.8


def test_mate_value():
    score = chess.engine.PovScore(chess.engine.Mate(3), chess.WHITE)
    cp, mate, value = pov_score_to_value(score)
    assert cp is None
    assert mate == 3
    assert value == 1.0
