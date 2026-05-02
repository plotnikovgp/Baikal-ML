"""Plot training metrics for selected noise_sig experiments (metrics.csv)."""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

BASE = Path("/home/plotnikovgp/baikal/Baikal-ML/pytorch_training/checkpoints/noise_sig_experiments")

# Main comparison: h / i (tres10 OR |label|)
EXPERIMENTS = {
    "h (tres10 OR |label|, old 2020)": "h_tres10_or_labels_old_2020_hs128",
    "i (tres10 OR |label|, merged smaller)": "i_tres10_or_labels_merged_smaller_hs128",
}

COLORS = {
    "h (tres10 OR |label|, old 2020)": "#1f77b4",
    "i (tres10 OR |label|, merged smaller)": "#ff7f0e",
}

DA_EXPERIMENT_DIR = "g_original_labels_da_hs128"

plt.rcParams.update({"font.size": 11, "figure.dpi": 120})


def load_metrics(name: str):
    path = BASE / name / "metrics.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def main():
    out_dir = Path("plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    for label, dirname in EXPERIMENTS.items():
        df = load_metrics(dirname)
        if df is None:
            print(f"Skip (no metrics): {dirname}", file=sys.stderr)
            continue
        c = COLORS[label]
        steps = df["step"]

        if "val/precision" in df.columns:
            axes[0, 0].plot(steps, df["val/precision"], color=c, label=label, lw=2)

        if "val/recall" in df.columns:
            axes[0, 1].plot(steps, df["val/recall"], color=c, label=label, lw=2)

        if "val/auc" in df.columns:
            axes[1, 0].plot(steps, df["val/auc"], color=c, label=label, lw=2)

        if "val/loss" in df.columns:
            axes[1, 1].plot(steps, df["val/loss"], color=c, label=label, lw=2, alpha=0.95)
        if "train/loss" in df.columns:
            tl = df.dropna(subset=["train/loss"])
            if len(tl) > 0:
                axes[1, 1].plot(
                    tl["step"],
                    tl["train/loss"],
                    color=c,
                    label=f"{label} (train)",
                    lw=1.2,
                    ls="--",
                    alpha=0.7,
                )

    axes[0, 0].set_title("Val precision (at recall≥0.9 where applicable)")
    axes[0, 0].set_ylabel("Precision")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.4)

    axes[0, 1].set_title("Val recall")
    axes[0, 1].set_ylabel("Recall")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.4)

    axes[1, 0].set_title("Val AUC")
    axes[1, 0].set_ylabel("AUC")
    axes[1, 0].set_xlabel("Step")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.4)

    axes[1, 1].set_title("Val loss (solid) / train loss (dashed)")
    axes[1, 1].set_ylabel("Loss")
    axes[1, 1].set_xlabel("Step")
    axes[1, 1].legend(fontsize=7)
    axes[1, 1].grid(True, alpha=0.4)

    fig.suptitle(
        "Training metrics — h (old 2020) vs i (merged smaller), tres10 OR |label|",
        fontweight="bold",
        fontsize=14,
    )
    plt.tight_layout()
    out1 = out_dir / "training_metrics.png"
    fig.savefig(out1, bbox_inches="tight")
    print(f"Saved {out1}")
    plt.close(fig)

    df_g = load_metrics(DA_EXPERIMENT_DIR)
    if df_g is None:
        print("Skip DA figure (no g metrics)", file=sys.stderr)
        return

    fig2, axes2 = plt.subplots(2, 2, figsize=(16, 10))
    steps = df_g["step"]

    ax = axes2[0, 0]
    if "val/mc_2020_domain_accuracy" in df_g.columns:
        ax.plot(
            steps, df_g["val/mc_2020_domain_accuracy"], color="#2ca02c", lw=2, label="MC domain acc"
        )
    if "val/exp_data_domain_accuracy" in df_g.columns:
        ax.plot(
            steps,
            df_g["val/exp_data_domain_accuracy"],
            color="#1f77b4",
            lw=2,
            label="EXP domain acc",
        )
    ax.axhline(0.5, color="gray", ls="--", lw=1, alpha=0.5)
    ax.set_title("Val domain accuracy")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.4)

    ax = axes2[0, 1]
    if "val/mc_2020_domain_auc" in df_g.columns:
        ax.plot(steps, df_g["val/mc_2020_domain_auc"], color="#2ca02c", lw=2, label="MC domain AUC")
    if "val/exp_data_domain_auc" in df_g.columns:
        ax.plot(
            steps, df_g["val/exp_data_domain_auc"], color="#1f77b4", lw=2, label="EXP domain AUC"
        )
    ax.axhline(0.5, color="gray", ls="--", lw=1, alpha=0.5)
    ax.set_title("Val domain AUC")
    ax.set_ylabel("AUC")
    ax.legend()
    ax.grid(True, alpha=0.4)

    ax = axes2[1, 0]
    train_steps = df_g.dropna(subset=["train/domain_loss"])
    if len(train_steps) > 0:
        ax.plot(
            train_steps["step"],
            train_steps["train/domain_loss"],
            color="#d62728",
            lw=2,
            label="domain loss",
        )
        ax.plot(
            train_steps["step"],
            train_steps["train/noise_sig_loss"],
            color="#2ca02c",
            lw=2,
            label="noise_sig loss",
        )
    ax.set_title("Train: domain loss vs noise/sig loss")
    ax.set_ylabel("Loss")
    ax.set_xlabel("Step")
    ax.legend()
    ax.grid(True, alpha=0.4)

    ax = axes2[1, 1]
    if "val/mc_2020_precision" in df_g.columns:
        ax.plot(steps, df_g["val/mc_2020_precision"], color="#d62728", lw=2, label="MC precision")
    if "val/mc_2020_auc" in df_g.columns:
        ax.plot(steps, df_g["val/mc_2020_auc"], color="#9467bd", lw=2, label="MC AUC")
    ax.set_title("g: val noise/sig on MC")
    ax.set_ylabel("Value")
    ax.set_xlabel("Step")
    ax.legend()
    ax.grid(True, alpha=0.4)

    fig2.suptitle("Domain adaptation — g (original labels + DA)", fontweight="bold", fontsize=14)
    plt.tight_layout()
    out2 = out_dir / "da_metrics.png"
    fig2.savefig(out2, bbox_inches="tight")
    print(f"Saved {out2}")
    plt.close(fig2)


if __name__ == "__main__":
    main()
