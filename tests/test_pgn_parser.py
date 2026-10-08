from pathlib import Path

from chess_env.pgn_parser import iter_policy_rows


def test_parse_small_pgn(tmp_path: Path):
    pgn = tmp_path / "one.pgn"
    pgn.write_text(
        """[Event "Rated rapid game"]
[Site "https://lichess.org/testgame"]
[White "A"]
[Black "B"]
[Result "1-0"]
[WhiteElo "2000"]
[BlackElo "2000"]
[TimeControl "600+0"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 1-0
""",
        encoding="utf-8",
    )
    rows = list(iter_policy_rows(pgn, max_games=1, min_elo=1800, min_ply=10))
    assert len(rows) == 10
    assert rows[0]["played_move_uci"] == "e2e4"
    assert rows[0]["game_id"] == "testgame"
