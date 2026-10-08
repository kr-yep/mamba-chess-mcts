#!/usr/bin/env bash
set -euo pipefail

export STOCKFISH_PATH="${STOCKFISH_PATH:-tools/stockfish/stockfish/stockfish-ubuntu-x86-64-avx2}"

MAX_GAMES="${MAX_GAMES:-1000}" MIN_ELO="${MIN_ELO:-1600}" bash scripts/prepare_policy_dataset.sh
LIMIT_POSITIONS="${LIMIT_POSITIONS:-200}" STOCKFISH_DEPTH="${STOCKFISH_DEPTH:-4}" bash scripts/generate_stockfish_values.sh
python -m chess_env.dataset_builder --value-labels data/value_labels/stockfish_values.parquet

python -m train.train_policy --config configs/train_smoke_mamba.yaml
python -m train.train_policy --config configs/train_smoke_cnn.yaml
python -m train.train_value --config configs/train_smoke_mamba.yaml
python -m train.train_value --config configs/train_smoke_cnn.yaml

python -m search.mcts --model board_mamba --checkpoint experiments/checkpoints/board_mamba_smoke.pt --simulations 4
python -m eval.mcts_timing \
  --model board_mamba \
  --checkpoint experiments/checkpoints/board_mamba_smoke.pt \
  --simulations 4 8 \
  --positions 4
python -m eval.tournament \
  --model board_mamba \
  --checkpoint experiments/checkpoints/board_mamba_smoke.pt \
  --games 2 \
  --simulations 4 \
  --output experiments/results/tournament_results_smoke.csv

python -m analysis.make_tables
python -m analysis.make_plots
python -m analysis.summarize_results
