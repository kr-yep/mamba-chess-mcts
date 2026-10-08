#!/usr/bin/env bash
set -euo pipefail
python -m search.self_play \
  --model board_mamba \
  --checkpoint "${CHECKPOINT:-experiments/checkpoints/board_mamba_real_joint.pt}" \
  --output "${OUTPUT:-data/self_play/self_play_real_mamba.parquet}" \
  --games "${GAMES:-20}" \
  --simulations "${SIMULATIONS:-32}" \
  --max-plies "${MAX_PLIES:-160}" \
  --temperature-plies "${TEMPERATURE_PLIES:-16}" \
  --seed "${SEED:-2025}"
python -m train.train_self_play --config configs/train_self_play_real_mamba.yaml
