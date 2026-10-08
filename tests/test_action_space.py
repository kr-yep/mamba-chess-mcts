import chess

from chess_env.action_space import action_id_to_move, legal_moves_to_action_ids, move_to_action_id


def assert_roundtrip(board: chess.Board, move_uci: str):
    move = chess.Move.from_uci(move_uci)
    aid = move_to_action_id(board, move)
    decoded = action_id_to_move(board, aid)
    assert decoded == move


def test_starting_position_roundtrip():
    board = chess.Board()
    assert len(legal_moves_to_action_ids(board)) == 20
    assert_roundtrip(board, "e2e4")
    assert_roundtrip(board, "g1f3")


def test_capture_roundtrip():
    board = chess.Board()
    board.push_san("e4")
    board.push_san("d5")
    assert_roundtrip(board, "e4d5")


def test_castling_roundtrip():
    board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    assert_roundtrip(board, "e1g1")
    assert_roundtrip(board, "e1c1")


def test_en_passant_roundtrip():
    board = chess.Board()
    for san in ["e4", "a6", "e5", "d5"]:
        board.push_san(san)
    assert_roundtrip(board, "e5d6")


def test_queen_promotion_roundtrip():
    board = chess.Board("8/P7/8/8/8/8/8/k6K w - - 0 1")
    assert_roundtrip(board, "a7a8q")


def test_underpromotion_roundtrip():
    board = chess.Board("8/P7/8/8/8/8/8/k6K w - - 0 1")
    for suffix in ["n", "b", "r"]:
        assert_roundtrip(board, f"a7a8{suffix}")


def test_check_position_only_legal_moves_encoded():
    board = chess.Board("4k3/8/8/8/8/8/4r3/4K3 w - - 0 1")
    moves = [action_id_to_move(board, aid) for aid in legal_moves_to_action_ids(board)]
    assert all(move in board.legal_moves for move in moves)
