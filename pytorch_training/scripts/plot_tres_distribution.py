"""Plot t_res distributions: all hits vs signal hits (MC only)."""

import sys
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OLD_DATA = "/home2/ivkhar/Baikal/data/normed/baikal_2020_sig-noise_mid-eq_normed.h5"
NEW_DATA = "/home/plotnikovgp/baikal/data/baikal_mc_merged_m50-na15-nue15_all_smaller_norm.h5"

MAX_EVENTS = 10_000
TRES_CUT = 10.0

plt.rcParams.update({"font.size": 12, "figure.dpi": 130})


def load_tres_data(h5_path, split="val", max_events=10_000):
    with h5py.File(h5_path, "r") as f:
        ev_starts = f[f"{split}/ev_starts/data"][:]
        n_events = min(max_events, len(ev_starts) - 1)
        end_hit = int(ev_starts[n_events])

        t_res = f[f"{split}/t_res/data"][:end_hit].astype(np.float32)
        labels = f[f"{split}/labels/data"][:end_hit].astype(np.float32)

    return t_res, labels, n_events


def plot_tres(t_res, labels, n_events, title, out_path):
    signal_mask = (np.abs(t_res) < TRES_CUT) | (np.abs(labels) > 0)
    tres_all = t_res
    tres_sig = t_res[signal_mask]

    bins = np.linspace(t_res.min(), t_res.max(), 300)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    ax.hist(tres_all, bins=bins, density=True, histtype="step", lw=2, color="#1f77b4")
    ax.set_xlabel("t_res [ns]")
    ax.set_ylabel("Density")
    ax.set_title(f"All hits ({len(tres_all):,} hits, {n_events:,} events)")
    ax.set_yscale("log")
    ax.axvline(-TRES_CUT, color="red", ls="--", alpha=0.5, label=f"±{TRES_CUT} ns")
    ax.axvline(TRES_CUT, color="red", ls="--", alpha=0.5)
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    tres_by_label = t_res[np.abs(labels) > 0]
    tres_by_tres = t_res[np.abs(t_res) < TRES_CUT]
    ax.hist(
        tres_sig,
        bins=bins,
        density=True,
        histtype="step",
        lw=2,
        color="#1f77b4",
        label=f"|t_res|<{TRES_CUT} OR |label|>0 ({len(tres_sig):,})",
    )
    ax.hist(
        tres_by_tres,
        bins=bins,
        density=True,
        histtype="step",
        lw=1.5,
        color="#2ca02c",
        ls="--",
        label=f"|t_res|<{TRES_CUT} only ({len(tres_by_tres):,})",
    )
    ax.hist(
        tres_by_label,
        bins=bins,
        density=True,
        histtype="step",
        lw=1.5,
        color="#ff7f0e",
        ls="--",
        label=f"|label|>0 only ({len(tres_by_label):,})",
    )
    ax.set_xlabel("t_res [ns]")
    ax.set_ylabel("Density")
    ax.set_title(f"Signal hits — |t_res|<{TRES_CUT} OR |label|>0")
    ax.set_yscale("log")
    ax.axvline(-TRES_CUT, color="red", ls="--", alpha=0.5)
    ax.axvline(TRES_CUT, color="red", ls="--", alpha=0.5)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontweight="bold", fontsize=14)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)

    n_only_label = int(((np.abs(labels) > 0) & (np.abs(t_res) >= TRES_CUT)).sum())
    n_only_tres = int(((np.abs(labels) == 0) & (np.abs(t_res) < TRES_CUT)).sum())
    n_both = int(((np.abs(labels) > 0) & (np.abs(t_res) < TRES_CUT)).sum())
    print(f"  label>0 AND |tres|<{TRES_CUT}: {n_both:,}")
    print(f"  label>0 AND |tres|>={TRES_CUT}: {n_only_label:,}")
    print(f"  label<=0 AND |tres|<{TRES_CUT}: {n_only_tres:,}")
    print(f"  Total signal (OR): {n_both + n_only_label + n_only_tres:,} / {len(t_res):,}")


def main():
    out_dir = Path("plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading OLD data...")
    t_old, l_old, n_old = load_tres_data(OLD_DATA, "val", MAX_EVENTS)
    plot_tres(
        t_old,
        l_old,
        n_old,
        f"t_res distribution — OLD 2020 data (val, {n_old:,} events)",
        out_dir / "tres_distribution_old.png",
    )

    print("\nLoading NEW (merged smaller) data...")
    t_new, l_new, n_new = load_tres_data(NEW_DATA, "val", MAX_EVENTS)
    plot_tres(
        t_new,
        l_new,
        n_new,
        f"t_res distribution — merged_smaller data (val, {n_new:,} events)",
        out_dir / "tres_distribution_new.png",
    )


if __name__ == "__main__":
    main()
