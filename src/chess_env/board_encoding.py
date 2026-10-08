"""Board encodings for sequence and CNN models."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import chess
import numpy as np

CLS_TOKEN = 0
EMPTY_TOKEN = 1
PIECE_TOKEN_OFFSET = 2
SIDE_TOKEN_OFFSET = 14
CASTLING_TOKEN_OFFSET = 16
EP_TOKEN_OFFSET = 32
HALFMOVE_BUCKET_OFFSET = 41
FULLMOVE_BUCKET_OFFSET = 49
VOCAB_SIZE = 65
SEQUENCE_LENGTH = 70


PIECE_TO_INDEX = {
    (chess.WHITE, chess.PAWN): 0,
    (chess.WHITE, chess.KNIGHT): 1,
    (chess.WHITE, chess.BISHOP): 2,
    (chess.WHITE, chess.ROOK): 3,
    (chess.WHITE, chess.QUEEN): 4,
    (chess.WHITE, chess.KING): 5,
    (chess.BLACK, chess.PAWN): 6,
    (chess.BLACK, chess.KNIGHT): 7,
    (chess.BLACK, chess.BISHOP): 8,
    (chess.BLACK, chess.ROOK): 9,
    (chess.BLACK, chess.QUEEN): 10,
    (chess.BLACK, chess.KING): 11,
}


@dataclass(frozen=True)
class BoardEncodingConfig:
    use_2d_pe: bool = False
    shuffled_order: bool = False
    shuffle_seed: int = 2025
    include_counters: bool = True


def normal_square_order() -> list[int]:
    """Return a1, b1, ..., h8 square order."""

    return list(chess.SQUARES)


@lru_cache(maxsize=16)
def shuffled_square_order(seed: int = 2025) -> tuple[int, ...]:
    rng = np.random.default_rng(seed)
    order = np.array(normal_square_order())
    rng.shuffle(order)
    return tuple(int(x) for x in order)


def square_order(config: BoardEncodingConfig) -> list[int]:
    if config.shuffled_order:
        return list(shuffled_square_order(config.shuffle_seed))
    return normal_square_order()


def piece_token(piece: chess.Piece | None) -> int:
    if piece is None:
        return EMPTY_TOKEN
    return PIECE_TOKEN_OFFSET + PIECE_TO_INDEX[(piece.color, piece.piece_type)]


def castling_token(board: chess.Board) -> int:
    mask = 0
    if board.has_kingside_castling_rights(chess.WHITE):
        mask |= 1
    if board.has_queenside_castling_rights(chess.WHITE):
        mask |= 2
    if board.has_kingside_castling_rights(chess.BLACK):
        mask |= 4
    if board.has_queenside_castling_rights(chess.BLACK):
        mask |= 8
    return CASTLING_TOKEN_OFFSET + mask


def ep_token(board: chess.Board) -> int:
    if board.ep_square is None:
        return EP_TOKEN_OFFSET + 8
    return EP_TOKEN_OFFSET + chess.square_file(board.ep_square)


def _bucket(value: int, max_bucket: int = 7) -> int:
    return min(max(value, 0), max_bucket)


def encode_board_sequence(
    board: chess.Board, config: BoardEncodingConfig | None = None
) -> dict[str, np.ndarray]:
    """Encode a board as token ids plus square row/column metadata."""

    config = config or BoardEncodingConfig()
    tokens: list[int] = [CLS_TOKEN]
    rows: list[int] = [8]
    cols: list[int] = [8]

    for square in square_order(config):
        tokens.append(piece_token(board.piece_at(square)))
        cols.append(chess.square_file(square))
        rows.append(chess.square_rank(square))

    tokens.append(SIDE_TOKEN_OFFSET + int(board.turn == chess.BLACK))
    rows.append(8)
    cols.append(8)
    tokens.append(castling_token(board))
    rows.append(8)
    cols.append(8)
    tokens.append(ep_token(board))
    rows.append(8)
    cols.append(8)

    if config.include_counters:
        tokens.append(HALFMOVE_BUCKET_OFFSET + _bucket(board.halfmove_clock))
        rows.append(8)
        cols.append(8)
        tokens.append(FULLMOVE_BUCKET_OFFSET + _bucket(board.fullmove_number - 1))
        rows.append(8)
        cols.append(8)

    return {
        "tokens": np.asarray(tokens, dtype=np.int64),
        "rows": np.asarray(rows, dtype=np.int64),
        "cols": np.asarray(cols, dtype=np.int64),
    }


def encode_board_planes(board: chess.Board) -> np.ndarray:
    """Encode board as CNN planes `[18, 8, 8]`.

    Planes:
    - 12 piece planes
    - side to move
    - four castling-right planes
    - en-passant target plane
    """

    planes = np.zeros((18, 8, 8), dtype=np.float32)
    for square, piece in board.piece_map().items():
        idx = PIECE_TO_INDEX[(piece.color, piece.piece_type)]
        file_idx = chess.square_file(square)
        rank_idx = chess.square_rank(square)
        planes[idx, rank_idx, file_idx] = 1.0

    planes[12, :, :] = 1.0 if board.turn == chess.WHITE else 0.0
    planes[13, :, :] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
    planes[14, :, :] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
    planes[15, :, :] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
    planes[16, :, :] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0
    if board.ep_square is not None:
        planes[17, chess.square_rank(board.ep_square), chess.square_file(board.ep_square)] = 1.0
    return planes
