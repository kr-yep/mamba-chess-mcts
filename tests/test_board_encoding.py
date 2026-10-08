import chess

from chess_env.board_encoding import (
    BoardEncodingConfig,
    encode_board_planes,
    encode_board_sequence,
    normal_square_order,
    shuffled_square_order,
)


def test_sequence_encoding_shape():
    enc = encode_board_sequence(chess.Board())
    assert enc["tokens"].shape == (70,)
    assert enc["rows"].shape == (70,)
    assert enc["cols"].shape == (70,)


def test_shuffled_order_is_fixed_and_different():
    assert list(shuffled_square_order(123)) == list(shuffled_square_order(123))
    assert list(shuffled_square_order(123)) != normal_square_order()


def test_planes_shape_and_piece_count():
    planes = encode_board_planes(chess.Board())
    assert planes.shape == (18, 8, 8)
    assert planes[:12].sum() == 32
