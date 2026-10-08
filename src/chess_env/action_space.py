"""AlphaZero-style chess action space.

The fixed action space is 64 from-squares times 73 move types:

- 56 queen-like moves: 8 directions x distances 1..7
- 8 knight moves
- 9 underpromotion moves: 3 directions x {knight, bishop, rook}

Queen promotions are represented as normal queen-like moves to the promotion
square. Castling is represented as a king move by distance two.
"""

from __future__ import annotations

from dataclasses import dataclass

import chess

ACTION_PLANES = 73
ACTION_SIZE = 64 * ACTION_PLANES

DIRECTIONS: tuple[tuple[int, int], ...] = (
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1),
    (1, 1),
    (1, -1),
    (-1, 1),
    (-1, -1),
)

KNIGHT_DELTAS: tuple[tuple[int, int], ...] = (
    (1, 2),
    (2, 1),
    (2, -1),
    (1, -2),
    (-1, -2),
    (-2, -1),
    (-2, 1),
    (-1, 2),
)

UNDERPROMOTION_PIECES: tuple[int, ...] = (
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
)

PROMOTION_FILE_DELTAS: tuple[int, ...] = (-1, 0, 1)


@dataclass(frozen=True)
class DecodedMoveType:
    df: int
    dr: int
    promotion: int | None = None


def _square_file_rank(square: int) -> tuple[int, int]:
    return chess.square_file(square), chess.square_rank(square)


def _on_board(file_idx: int, rank_idx: int) -> bool:
    return 0 <= file_idx < 8 and 0 <= rank_idx < 8


def _queen_plane(df: int, dr: int) -> int | None:
    for direction_index, (unit_df, unit_dr) in enumerate(DIRECTIONS):
        for distance in range(1, 8):
            if df == unit_df * distance and dr == unit_dr * distance:
                return direction_index * 7 + (distance - 1)
    return None


def _knight_plane(df: int, dr: int) -> int | None:
    try:
        return 56 + KNIGHT_DELTAS.index((df, dr))
    except ValueError:
        return None


def _underpromotion_plane(board: chess.Board, move: chess.Move, df: int) -> int | None:
    if move.promotion not in UNDERPROMOTION_PIECES:
        return None

    piece = board.piece_at(move.from_square)
    if piece is None or piece.piece_type != chess.PAWN:
        return None

    try:
        file_index = PROMOTION_FILE_DELTAS.index(df)
        piece_index = UNDERPROMOTION_PIECES.index(move.promotion)
    except ValueError:
        return None

    return 64 + file_index * len(UNDERPROMOTION_PIECES) + piece_index


def move_to_action_id(board: chess.Board, move: chess.Move) -> int:
    """Convert a chess move to an integer action id.

    Raises:
        ValueError: if the move cannot be represented by the 4672 action space.
    """

    if move not in board.legal_moves:
        raise ValueError(f"Illegal move for board: {move.uci()}")

    from_file, from_rank = _square_file_rank(move.from_square)
    to_file, to_rank = _square_file_rank(move.to_square)
    df = to_file - from_file
    dr = to_rank - from_rank

    if move.promotion in UNDERPROMOTION_PIECES:
        plane = _underpromotion_plane(board, move, df)
    else:
        plane = _queen_plane(df, dr)
        if plane is None:
            plane = _knight_plane(df, dr)

    if plane is None:
        raise ValueError(f"Move cannot be encoded: {move.uci()}")
    return move.from_square * ACTION_PLANES + plane


def _decode_move_type(board: chess.Board, from_square: int, plane: int) -> DecodedMoveType:
    if 0 <= plane < 56:
        direction_index, distance_index = divmod(plane, 7)
        unit_df, unit_dr = DIRECTIONS[direction_index]
        distance = distance_index + 1
        return DecodedMoveType(unit_df * distance, unit_dr * distance)

    if 56 <= plane < 64:
        df, dr = KNIGHT_DELTAS[plane - 56]
        return DecodedMoveType(df, dr)

    if 64 <= plane < 73:
        relative = plane - 64
        file_delta_index, piece_index = divmod(relative, len(UNDERPROMOTION_PIECES))
        df = PROMOTION_FILE_DELTAS[file_delta_index]
        piece = board.piece_at(from_square)
        dr = 1 if piece and piece.color == chess.WHITE else -1
        return DecodedMoveType(df, dr, UNDERPROMOTION_PIECES[piece_index])

    raise ValueError(f"Invalid action plane: {plane}")


def action_id_to_move(board: chess.Board, action_id: int) -> chess.Move | None:
    """Convert an action id to a legal move for the given board, or `None`."""

    if action_id < 0 or action_id >= ACTION_SIZE:
        return None

    from_square, plane = divmod(action_id, ACTION_PLANES)
    from_file, from_rank = _square_file_rank(from_square)
    decoded = _decode_move_type(board, from_square, plane)
    to_file = from_file + decoded.df
    to_rank = from_rank + decoded.dr
    if not _on_board(to_file, to_rank):
        return None

    to_square = chess.square(to_file, to_rank)
    move = chess.Move(from_square, to_square, promotion=decoded.promotion)

    if move in board.legal_moves:
        return move

    # Queen promotion is encoded as the normal forward/capture queen-like move.
    piece = board.piece_at(from_square)
    if piece and piece.piece_type == chess.PAWN and decoded.promotion is None:
        promotion_rank = 7 if piece.color == chess.WHITE else 0
        if to_rank == promotion_rank:
            queen_move = chess.Move(from_square, to_square, promotion=chess.QUEEN)
            if queen_move in board.legal_moves:
                return queen_move

    return None


def legal_moves_to_action_ids(board: chess.Board) -> list[int]:
    """Return all legal moves represented as action ids."""

    return [move_to_action_id(board, move) for move in board.legal_moves]
