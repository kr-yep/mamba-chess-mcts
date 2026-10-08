"""Summarize completed and missing result artifacts."""

from __future__ import annotations

from pathlib import Path


EXPECTED = [
    "experiments/results/policy_metrics.csv",
    "experiments/results/value_metrics.csv",
    "experiments/results/tournament_results.csv",
    "experiments/results/mcts_timing.csv",
    "experiments/tables/policy_table.md",
    "experiments/tables/value_table.md",
    "experiments/tables/tournament_table.md",
    "experiments/tables/timing_table.md",
    "experiments/plots/policy_accuracy.png",
    "experiments/plots/value_mae.png",
    "experiments/plots/mcts_timing.png",
]


def main() -> None:
    for item in EXPECTED:
        status = "done" if Path(item).exists() else "not completed"
        print(f"{status}: {item}")


if __name__ == "__main__":
    main()
