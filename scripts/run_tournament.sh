#!/usr/bin/env bash
set -euo pipefail
python -m eval.tournament \
  --model "${MODEL:-board_mamba}" \
  --checkpoint "${CHECKPOINT:-experiments/checkpoints/board_mamba.pt}" \
  --games "${GAMES:-50}" \
  --simulations "${SIMULATIONS:-32}"
