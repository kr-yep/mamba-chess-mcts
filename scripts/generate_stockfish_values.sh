#!/usr/bin/env bash
set -euo pipefail
python -m chess_env.stockfish_labeler \
  --input "${VALUE_INPUT:-data/processed/policy_train.parquet}" \
  --output "${VALUE_OUTPUT:-data/value_labels/stockfish_values.parquet}" \
  --limit-positions "${LIMIT_POSITIONS:-50000}" \
  --depth "${STOCKFISH_DEPTH:-10}" \
  --stockfish-path "${STOCKFISH_PATH:?Set STOCKFISH_PATH to the Stockfish binary}"
