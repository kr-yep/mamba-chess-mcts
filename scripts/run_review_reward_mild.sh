#!/usr/bin/env bash
set -euo pipefail

CFG=${1:-configs/review_reward_mild.yaml}

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
    "analysis.review_games",
    "--input",
    cfg["input"],
    "--output",
    cfg["output"],
    "--policy-output",
    cfg["policy_output"],
    "--limit-positions",
    str(cfg.get("limit_positions", 500)),
    "--depth",
    str(cfg.get("depth", 8)),
    "--min-value-gain",
    str(cfg.get("min_value_gain", 0.12)),
]
subprocess.run(cmd, check=True)
PY
