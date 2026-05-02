"""Plot val/train metrics for one or more experiment dirs (metrics.csv)."""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams.update({"font.size": 11, "figure.dpi": 120})


def load_metrics(mpath: Path):
    if not mpath.is_file():
        return None
    return pd.read_csv(mpath)


def four_panel_fig(df, title: str, out_path: Path):
    if df is None or "step" not in getattr(df, "columns", ()):
        return False
    s = df["step"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    if "val/precision" in df.columns:
        axes[0, 0].plot(s, df["val/precision"], lw=1.2, color="#1f77b4")
    axes[0, 0].set_title("Val precision")
    axes[0, 0].set_ylabel("Precision")
    axes[0, 0].grid(True, alpha=0.3)

    if "val/recall" in df.columns:
        axes[0, 1].plot(s, df["val/recall"], lw=1.2, color="#1f77b4")
    axes[0, 1].set_title("Val recall")
    axes[0, 1].set_ylabel("Recall")
    axes[0, 1].grid(True, alpha=0.3)

    if "val/auc" in df.columns:
        axes[1, 0].plot(s, df["val/auc"], lw=1.2, color="#1f77b4")
    axes[1, 0].set_title("Val AUC")
    axes[1, 0].set_ylabel("AUC")
    axes[1, 0].set_xlabel("Step")
    axes[1, 0].grid(True, alpha=0.3)

    if "val/loss" in df.columns:
        axes[1, 1].plot(s, df["val/loss"], lw=1.2, color="#1f77b4", label="val/loss", alpha=0.95)
    if "train/loss" in df.columns:
        tl = df.dropna(subset=["train/loss"])
        if len(tl) > 0:
            axes[1, 1].plot(
                tl["step"],
                tl["train/loss"],
                lw=1.0,
                color="#d62728",
                ls="--",
                label="train/loss",
                alpha=0.8,
            )
    axes[1, 1].set_title("Loss")
    axes[1, 1].set_ylabel("Loss")
    axes[1, 1].set_xlabel("Step")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend(fontsize=8, loc="best")

    fig.suptitle(title, fontweight="bold", fontsize=12)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return True


def compare_fig(pairs, out_path: Path, title: str):
    series = []
    for label, mpath in pairs:
        df = load_metrics(mpath)
        if df is not None and "step" in df.columns:
            series.append((label, df))
    if not series:
        return False

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    colors = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728")
    for (label, df), c in zip(series, colors):
        s = df["step"]
        if "val/precision" in df.columns:
            axes[0, 0].plot(s, df["val/precision"], lw=1.2, color=c, label=label)
        if "val/recall" in df.columns:
            axes[0, 1].plot(s, df["val/recall"], lw=1.2, color=c, label=label)
        if "val/auc" in df.columns:
            axes[1, 0].plot(s, df["val/auc"], lw=1.2, color=c, label=label)
        if "val/loss" in df.columns:
            axes[1, 1].plot(s, df["val/loss"], lw=1.1, color=c, label=label, alpha=0.95)
        if "train/loss" in df.columns:
            tl = df.dropna(subset=["train/loss"])
            if len(tl) > 0:
                axes[1, 1].plot(
                    tl["step"],
                    tl["train/loss"],
                    lw=0.9,
                    color=c,
                    ls="--",
                    alpha=0.5,
                )

    axes[0, 0].set_title("Val precision")
    axes[0, 0].set_ylabel("Precision")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=7, loc="best")

    axes[0, 1].set_title("Val recall")
    axes[0, 1].set_ylabel("Recall")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend(fontsize=7, loc="best")

    axes[1, 0].set_title("Val AUC")
    axes[1, 0].set_ylabel("AUC")
    axes[1, 0].set_xlabel("Step")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend(fontsize=7, loc="best")

    axes[1, 1].set_title("Loss (val solid, train faint dashed)")
    axes[1, 1].set_ylabel("Loss")
    axes[1, 1].set_xlabel("Step")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend(fontsize=7, loc="best")

    fig.suptitle(title, fontweight="bold", fontsize=12)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("plots"),
    )
    args = p.parse_args()
    out_dir = args.out_dir

    base = Path(
        "/home/plotnikovgp/baikal/Baikal-ML/pytorch_training/checkpoints/noise_sig_experiments"
    )

    h1 = base / "h_tres10_or_labels_old_2020_hs128" / "metrics.csv"
    h2 = base / "h_tres10_or_labels_old_data_hs128" / "metrics.csv"

    n1 = four_panel_fig(
        load_metrics(h1),
        "h: tres10 OR |label|>0 — old 2020 (ivkhar) val metrics",
        out_dir / "h_tres10_or_labels_old_2020_hs128_train.png",
    )
    four_panel_fig(
        load_metrics(h2),
        "h: tres10 OR |label|>0 — old_data (same dir name variant)",
        out_dir / "h_tres10_or_labels_old_data_hs128_train.png",
    )

    if n1:
        print(f"Saved {out_dir / 'h_tres10_or_labels_old_2020_hs128_train.png'}")
    if not h2.is_file():
        print(
            f"No metrics at {h2} (empty run or Hydra-only dir); skipped second per-run plot.",
            file=sys.stderr,
        )

    compare_fig(
        [
            ("old_2020 (ivkhar)", h1),
            ("old_data (path)", h2),
        ],
        out_dir / "h_or_labels_old_paths_compare.png",
        "h runs: only directories with metrics.csv are drawn",
    )
    if h1.is_file() or h2.is_file():
        print(f"Saved {out_dir / 'h_or_labels_old_paths_compare.png'}")


if __name__ == "__main__":
    main()
