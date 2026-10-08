#!/usr/bin/env bash
set -euo pipefail
URL="${LICHESS_URL:-https://database.lichess.org/standard/lichess_db_standard_rated_2013-01.pgn.zst}"
OUT="${LICHESS_OUT:-data/raw/lichess_sample.pgn.zst}"
mkdir -p "$(dirname "$OUT")"
if command -v wget >/dev/null 2>&1; then
  wget -c "$URL" -O "$OUT"
else
  curl -L "$URL" -o "$OUT"
fi
