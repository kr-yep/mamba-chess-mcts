#!/usr/bin/env bash
set -euo pipefail

CFG=${1:-configs/gate_review_vs_mild.yaml}

python - <<'PY' "$CFG"
import subprocess
import sys
import yaml

cfg_path = sys.argv[1]
with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

cmd = [
    sys.executable,
    "-m",
    "eval.head_to_head",
    "--model-a",
    cfg.get("model_a", "board_mamba"),
    "--checkpoint-a",
    cfg["checkpoint_a"],
    "--agent-a",
    cfg.get("agent_a", "mcts"),
    "--model-b",
    cfg.get("model_b", "board_mamba"),
    "--checkpoint-b",
    cfg["checkpoint_b"],
    "--agent-b",
    cfg.get("agent_b", "mcts"),
    "--games",
    str(cfg.get("games", 100)),
    "--simulations-a",
    str(cfg.get("simulations_a", 32)),
    "--simulations-b",
    str(cfg.get("simulations_b", 32)),
    "--max-plies",
    str(cfg.get("max_plies", 300)),
    "--output",
    cfg["output"],
]
if cfg.get("start_fens"):
    cmd += ["--start-fens", cfg["start_fens"]]
if cfg.get("max_start_fens"):
    cmd += ["--max-start-fens", str(cfg["max_start_fens"])]
subprocess.run(cmd, check=True)
PY
