#!/usr/bin/env bash
set -euo pipefail
python -m train.train_policy --config configs/train_board_mamba.yaml
