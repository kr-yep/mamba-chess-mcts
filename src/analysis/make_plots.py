"""Create standard plots from experiment CSVs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _save_line(df, x, y, hue, title, output):
    plt.figure(figsize=(7, 4))
    if hue in df:
        for name, sub in df.groupby(hue):
            plt.plot(sub[x], sub[y], marker="o", label=name)
        plt.legend()
    else:
        plt.plot(df[x], df[y], marker="o")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()


def main() -> None:
    results = Path("experiments/results")
    plots = Path("experiments/plots")
    plots.mkdir(parents=True, exist_ok=True)

    policy = results / "policy_metrics.csv"
    if policy.exists():
        df = pd.read_csv(policy)
        y = "val_top1" if "val_top1" in df else df.columns[-1]
        _save_line(df, "epoch", y, "model", "Policy accuracy", plots / "policy_accuracy.png")

    value = results / "value_metrics.csv"
    if value.exists():
        df = pd.read_csv(value)
        y = "val_mae" if "val_mae" in df else df.columns[-1]
        _save_line(df, "epoch", y, "model", "Value MAE", plots / "value_mae.png")

    timing = results / "mcts_timing.csv"
    if timing.exists():
        df = pd.read_csv(timing)
        _save_line(df, "simulations", "avg_move_time", "model", "MCTS timing", plots / "mcts_timing.png")

    ablation = results / "policy_ablation_pilot.csv"
    if ablation.exists():
        df = pd.read_csv(ablation)
        plt.figure(figsize=(8, 4))
        plt.bar(df["model"], df["val_top1"])
        plt.ylabel("Validation top-1")
        plt.xticks(rotation=20, ha="right")
        plt.title("Policy imitation pilot")
        plt.tight_layout()
        plt.savefig(plots / "policy_accuracy.png", dpi=160)
        plt.close()

    value_summary = results / "value_pilot_summary.csv"
    if value_summary.exists():
        df = pd.read_csv(value_summary)
        plt.figure(figsize=(6, 4))
        plt.bar(df["model"], df["val_mae"])
        plt.ylabel("Validation MAE")
        plt.title("Value prediction pilot")
        plt.tight_layout()
        plt.savefig(plots / "value_mae.png", dpi=160)
        plt.close()

    self_play = results / "self_play_metrics_real_mamba.csv"
    if self_play.exists():
        df = pd.read_csv(self_play)
        plt.figure(figsize=(7, 4))
        plt.plot(df["epoch"], df["val_top1"], marker="o", label="policy top-1")
        plt.plot(df["epoch"], df["val_value_mae"], marker="o", label="value MAE")
        plt.xlabel("Epoch")
        plt.title("Self-play fine-tuning")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots / "self_play_training.png", dpi=160)
        plt.close()

    shaped = results / "self_play_reward_shaped_metrics_real_mamba.csv"
    if shaped.exists():
        df = pd.read_csv(shaped)
        plt.figure(figsize=(7, 4))
        plt.plot(df["epoch"], df["val_top1"], marker="o", label="policy top-1")
        plt.plot(df["epoch"], df["val_value_mae"], marker="o", label="value MAE")
        plt.xlabel("Epoch")
        plt.title("Reward-shaped self-play fine-tuning")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots / "self_play_reward_shaped_training.png", dpi=160)
        plt.close()

    mild = results / "self_play_reward_mild_metrics_real_mamba.csv"
    if mild.exists():
        df = pd.read_csv(mild)
        plt.figure(figsize=(7, 4))
        plt.plot(df["epoch"], df["val_top1"], marker="o", label="policy top-1")
        plt.plot(df["epoch"], df["val_value_mae"], marker="o", label="value MAE")
        plt.xlabel("Epoch")
        plt.title("Mild reward-shaped self-play fine-tuning")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots / "self_play_reward_mild_training.png", dpi=160)
        plt.close()

    self_play_timing = results / "mcts_timing_self_play_real_mamba.csv"
    if self_play_timing.exists():
        df = pd.read_csv(self_play_timing)
        _save_line(df, "simulations", "avg_move_time", "model", "Self-play checkpoint MCTS timing", plots / "mcts_timing.png")

    review = results / "review_reward_mild.csv"
    if review.exists():
        df = pd.read_csv(review)
        if len(df):
            plt.figure(figsize=(7, 4))
            plt.hist(df["value_gain"], bins=20)
            plt.xlabel("Stockfish value gain")
            plt.ylabel("Positions")
            plt.title("Post-game review critical positions")
            plt.tight_layout()
            plt.savefig(plots / "review_value_gain.png", dpi=160)
            plt.close()


if __name__ == "__main__":
    main()
