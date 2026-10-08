#!/usr/bin/env bash
set -euo pipefail
python -m chess_env.pgn_parser \
  --input "${PGN_INPUT:-data/raw/lichess_sample.pgn.zst}" \
  --output data/interim/policy_positions.parquet \
  --max-games "${MAX_GAMES:-10000}" \
  --min-elo "${MIN_ELO:-1800}" \
  --min-ply "${MIN_PLY:-10}"
python -m chess_env.dataset_builder \
  --input data/interim/policy_positions.parquet \
  --output-dir data/processed \
  --seed "${SEED:-2025}"
