"""Plot per-epoch discriminator, independent-probe, and hit-task metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def rows(folder: Path, name: str):
    return json.loads((folder / f"{name}.json").read_text())["epochs"]


def series(row_list, particle, metric, stat="auc"):
    return [row["validation"][particle][metric][stat] for row in row_list]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = {
        name: rows(args.input_dir, name)
        for name in (
            "control",
            "da_masked_lr_only",
            "paper_control",
            "paper_da",
            "muatm_control",
            "muatm_da",
        )
    }
    epochs = [r["epoch"] for r in data["control"]]
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.5), sharex=True)
    plt.rcParams.update({"font.size": 10})

    def auc_panel(ax, control_name, da_name, particle, title):
        c, d = data[control_name], data[da_name]
        ax.plot(
            epochs,
            series(c, particle, "probe_masked"),
            "o-",
            color="#313638",
            label="Frozen-encoder probe, control",
        )
        ax.plot(
            epochs,
            series(d, particle, "probe_masked"),
            "o-",
            color="#1368aa",
            label="Frozen-encoder probe, DA",
        )
        ax.plot(
            epochs,
            series(d, particle, "head", "separability_auc"),
            "s--",
            color="#d15a30",
            label="Training discriminator, DA",
        )
        ax.axhline(0.5, color="0.7", lw=1)
        ax.set_title(title)
        ax.set_ylabel("MC vs data ROC AUC")
        ax.set_ylim(0.48, 1.01)
        ax.grid(alpha=0.22)

    auc_panel(
        axes[0, 0], "control", "da_masked_lr_only", "mixed", "1:1:1 MC mix; no z flip"
    )
    auc_panel(
        axes[0, 1], "paper_control", "paper_da", "mixed", "2:1:1 MC mix; random z flip"
    )
    auc_panel(
        axes[1, 0],
        "muatm_control",
        "muatm_da",
        "muatm",
        "2:1:1 task MC; muatm-only DA stream",
    )
    ax = axes[1, 1]
    for name, color, linestyle in (
        ("paper_control", "#313638", "-"),
        ("paper_da", "#1368aa", "--"),
    ):
        row_list = data[name]
        for metric, marker, label in (
            ("precision_at_0p5", "o", "precision"),
            ("recall_at_0p5", "s", "recall"),
        ):
            y = [row["validation"]["hit_task"][metric] for row in row_list]
            ax.plot(
                epochs,
                y,
                marker=marker,
                linestyle=linestyle,
                color=color,
                label=f"{name.replace('paper_', '')} {label}",
            )
    ax.set_title("MC signal-hit task, 2:1:1 + z flip")
    ax.set_ylabel("Precision / recall at p=0.5")
    ax.set_ylim(0.89, 0.98)
    ax.grid(alpha=0.22)
    for row in axes:
        for ax in row:
            ax.set_xticks(epochs)
            ax.set_xlabel("Fine-tuning epoch")
            ax.legend(fontsize=8, loc="best")
    fig.suptitle(
        "Noise-suppression domain adaptation: training head vs independent probe",
        fontsize=15,
    )
    fig.text(
        0.5,
        0.005,
        "Frozen logreg fitted on disjoint probe-train events, evaluated on held-out MC/experimental events; 2s8h raw-hit cut.",
        ha="center",
        fontsize=9,
        color="0.35",
    )
    fig.tight_layout(rect=(0, 0.025, 1, 0.96))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    print(args.output)


if __name__ == "__main__":
    main()
