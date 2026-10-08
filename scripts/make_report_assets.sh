#!/usr/bin/env bash
set -euo pipefail
python -m analysis.make_tables
python -m analysis.make_plots
python -m analysis.summarize_results
