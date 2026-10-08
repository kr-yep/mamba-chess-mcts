#!/usr/bin/env bash
set -euo pipefail
python -m search.mcts \
  --model "${MODEL:-board_mamba}" \
  --checkpoint "${CHECKPOINT:-experiments/checkpoints/board_mamba.pt}" \
  --simulations "${SIMULATIONS:-32}"
python -m eval.mcts_timing \
  --model "${MODEL:-board_mamba}" \
  --checkpoint "${CHECKPOINT:-experiments/checkpoints/board_mamba.pt}" \
  --simulations "${SIMULATIONS:-32}"
