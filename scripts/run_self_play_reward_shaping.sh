#!/usr/bin/env bash
set -euo pipefail

CFG=${1:-configs/self_play_reward_shaping.yaml}

python - <<'PY' "$CFG"
import subprocess
import sys
import yaml

cfg_path = sys.argv[1]
with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

reward = cfg.get("reward", {})
cmd = [
    sys.executable,
    "-m",
    "search.self_play",
    "--model",
    cfg["model"],
    "--checkpoint",
    cfg["checkpoint"],
    "--output",
    cfg["output"],
    "--games",
    str(cfg.get("games", 20)),
    "--simulations",
    str(cfg.get("simulations", 16)),
    "--max-plies",
    str(cfg.get("max_plies", 120)),
    "--temperature-plies",
    str(cfg.get("temperature_plies", 16)),
    "--seed",
    str(cfg.get("seed", 2026)),
    "--win-value",
    str(reward.get("win_value", 1.0)),
    "--draw-value",
    str(reward.get("draw_value", 0.0)),
    "--loss-value",
    str(reward.get("loss_value", -1.0)),
    "--record-player",
    str(cfg.get("record_player", "both")),
]
if cfg.get("opponent_checkpoint"):
    cmd += ["--opponent-checkpoint", cfg["opponent_checkpoint"]]
if cfg.get("opponent_model"):
    cmd += ["--opponent-model", cfg["opponent_model"]]
if cfg.get("start_fens"):
    cmd += ["--start-fens", cfg["start_fens"]]
if cfg.get("max_start_fens"):
    cmd += ["--max-start-fens", str(cfg["max_start_fens"])]
subprocess.run(cmd, check=True)
PY
