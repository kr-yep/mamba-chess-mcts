"""Create Markdown tables from result CSVs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


TABLES = {
    "policy_metrics.csv": "policy_table.md",
    "value_metrics.csv": "value_table.md",
    "tournament_summary.csv": "tournament_table.md",
    "mcts_timing.csv": "timing_table.md",
    "policy_ablation_pilot.csv": "policy_ablation_pilot.md",
    "value_pilot_summary.csv": "value_pilot_summary.md",
    "joint_pilot_summary.csv": "joint_pilot_summary.md",
    "checkpoint_eval_self_play_real_mamba.csv": "self_play_checkpoint_eval.md",
    "checkpoint_eval_reward_shaped_real_mamba.csv": "reward_shaped_checkpoint_eval.md",
    "checkpoint_eval_reward_mild_real_mamba.csv": "reward_mild_checkpoint_eval.md",
    "checkpoint_eval_review_policy_real_mamba.csv": "review_policy_checkpoint_eval.md",
    "checkpoint_eval_mixed_review_real_mamba.csv": "mixed_review_checkpoint_eval.md",
    "self_play_checkpoint_comparison.csv": "self_play_checkpoint_comparison.md",
    "self_play_metrics_real_mamba.csv": "self_play_training.md",
    "self_play_reward_shaped_metrics_real_mamba.csv": "self_play_reward_shaped_training.md",
    "self_play_reward_mild_metrics_real_mamba.csv": "self_play_reward_mild_training.md",
    "mcts_timing_self_play_real_mamba.csv": "self_play_mcts_timing.md",
    "mcts_timing_reward_shaped_real_mamba.csv": "reward_shaped_mcts_timing.md",
    "head_to_head_self_play_vs_joint_summary.csv": "head_to_head_self_play_vs_joint.md",
    "head_to_head_reward_shaped_vs_self_play_summary.csv": "head_to_head_reward_shaped_vs_self_play.md",
    "head_to_head_reward_mild_vs_self_play_summary.csv": "head_to_head_reward_mild_vs_self_play.md",
    "head_to_head_review_policy_vs_reward_mild_summary.csv": "head_to_head_review_policy_vs_reward_mild.md",
    "head_to_head_review_policy_vs_reward_mild_gated_summary.csv": "head_to_head_review_policy_vs_reward_mild_gated.md",
    "head_to_head_mixed_review_vs_reward_mild_gated_summary.csv": "head_to_head_mixed_review_vs_reward_mild_gated.md",
    "review_policy_metrics_real_mamba.csv": "review_policy_training.md",
    "mixed_self_play_review_metrics_real_mamba.csv": "mixed_self_play_review_training.md",
}


def main() -> None:
    results = Path("experiments/results")
    tables = Path("experiments/tables")
    tables.mkdir(parents=True, exist_ok=True)
    for csv_name, table_name in TABLES.items():
        path = results / csv_name
        out = tables / table_name
        if path.exists():
            df = pd.read_csv(path)
            out.write_text(df.to_markdown(index=False), encoding="utf-8")
        else:
            out.write_text(f"Not completed: `{path}` does not exist.\n", encoding="utf-8")


if __name__ == "__main__":
    main()
