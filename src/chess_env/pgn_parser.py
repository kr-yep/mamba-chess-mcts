"""Streaming Lichess PGN parser."""

from __future__ import annotations

import argparse
import io
from pathlib import Path
from typing import Iterator

import chess
import chess.pgn
import pandas as pd
import zstandard as zstd
from tqdm import tqdm

from chess_env.action_space import move_to_action_id


def _open_text(path: Path):
    if path.suffix == ".zst":
        fh = path.open("rb")
        dctx = zstd.ZstdDecompressor(max_window_size=2**31)
        stream = dctx.stream_reader(fh)
        return io.TextIOWrapper(stream, encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def _result_to_value(result: str) -> float | None:
    if result == "1-0":
        return 1.0
    if result == "0-1":
        return -1.0
    if result == "1/2-1/2":
        return 0.0
    return None


def iter_policy_rows(
    pgn_path: str | Path,
    max_games: int | None = None,
    min_elo: int = 1800,
    min_ply: int = 10,
    allowed_time_controls: set[str] | None = None,
    exclude_bullet: bool = True,
) -> Iterator[dict]:
    """Yield position rows from a PGN file, splitting must be done by game id later."""

    allowed_time_controls = allowed_time_controls or {"rapid", "classical", "blitz"}
    pgn_path = Path(pgn_path)
    with _open_text(pgn_path) as pgn:
        game_index = 0
        with tqdm(total=max_games, desc="Parsing games") as pbar:
            while max_games is None or game_index < max_games:
                game = chess.pgn.read_game(pgn)
                if game is None:
                    break
                game_index += 1
                pbar.update(1)

                headers = game.headers
                event = headers.get("Event", "").lower()
                if exclude_bullet and "bullet" in event:
                    continue
                if allowed_time_controls and not any(tc in event for tc in allowed_time_controls):
                    continue
                if headers.get("Variant", "Standard") != "Standard":
                    continue
                if headers.get("Termination", "") == "Abandoned":
                    continue

                try:
                    white_elo = int(headers.get("WhiteElo", 0))
                    black_elo = int(headers.get("BlackElo", 0))
                except ValueError:
                    continue
                if min(white_elo, black_elo) < min_elo:
                    continue

                moves = list(game.mainline_moves())
                if len(moves) < min_ply:
                    continue

                board = game.board()
                game_id = headers.get("Site", f"{pgn_path.stem}-{game_index}").rstrip("/").split("/")[-1]
                result = headers.get("Result", "*")
                for ply, move in enumerate(moves):
                    if move not in board.legal_moves:
                        break
                    legal_moves = [m.uci() for m in board.legal_moves]
                    yield {
                        "game_id": game_id,
                        "ply": ply,
                        "fen": board.fen(),
                        "side_to_move": "white" if board.turn == chess.WHITE else "black",
                        "legal_moves_uci": legal_moves,
                        "played_move_uci": move.uci(),
                        "played_action_id": move_to_action_id(board, move),
                        "result": result,
                        "result_value_white": _result_to_value(result),
                        "white_elo": white_elo,
                        "black_elo": black_elo,
                        "time_control": headers.get("TimeControl", ""),
                    }
                    board.push(move)


def write_policy_rows(args: argparse.Namespace) -> None:
    rows = list(
        iter_policy_rows(
            args.input,
            max_games=args.max_games,
            min_elo=args.min_elo,
            min_ply=args.min_ply,
        )
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(output, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-games", type=int, default=10000)
    parser.add_argument("--min-elo", type=int, default=1800)
    parser.add_argument("--min-ply", type=int, default=10)
    write_policy_rows(parser.parse_args())


if __name__ == "__main__":
    main()
