#!/usr/bin/env bash
set -euo pipefail
conda env create -f environment.yml
echo "Run: conda activate mamba-chess-mcts && pip install -e ."
