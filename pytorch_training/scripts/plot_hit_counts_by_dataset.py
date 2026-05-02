"""Compare signal vs noise hit counts by MC event type: two datasets x train/val/test (10k ev each)."""

from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

H5_2020 = "/home2/ivkhar/Baikal/data/normed/baikal_2020_sig-noise_mid-eq_normed.h5"
H5_MERGED = "/home/plotnikovgp/baikal/data/baikal_mc_merged_m50-na15-nue15_all_smaller_norm.h5"

MAX_EVENTS = 10_000
TRES_CUT = 10.0

TYPE_NAMES = ("muatm", "nuatm", "nue2")
DATASET_NAMES = [
    f"2020 (ivkhar)\n{H5_2020}",
    f"merged smaller\n{H5_MERGED}",
]
SPLITS = ("train", "val", "test")

plt.rcParams.update({"font.size": 10, "figure.dpi": 130})


def event_type_index(ev_id) -> int:
    if isinstance(ev_id, bytes):
        s = ev_id
    else:
        s = str(ev_id).encode("utf-8", errors="replace")
    if b"muatm" in s:
        return 0
    if b"nuatm" in s:
        return 1
    return 2


def count_hits_split(h5_path: str, split: str, max_events: int, tres_cut: float):
    with h5py.File(h5_path, "r") as f:
        ev_starts = f[f"{split}/ev_starts/data"][:]
        ev_ids = f[f"{split}/ev_ids/data"][:]
    n_ok = min(max_events, len(ev_starts) - 1)
    if n_ok <= 0:
        return None, 0, np.zeros(3, dtype=np.int64), np.zeros(3, dtype=np.int64)

    end = int(ev_starts[n_ok])
    with h5py.File(h5_path, "r") as f:
        labels = f[f"{split}/labels/data"][:end].astype(np.float32)
        t_res = f[f"{split}/t_res/data"][:end].astype(np.float32)

    signal = (np.abs(t_res) < tres_cut) | (labels > 0)

    sig_by_type = np.zeros(3, dtype=np.int64)
    noise_by_type = np.zeros(3, dtype=np.int64)

    for i in range(n_ok):
        t = event_type_index(ev_ids[i])
        a, b = int(ev_starts[i]), int(ev_starts[i + 1])
        sl = signal[a:b]
        sig_by_type[t] += int(sl.sum())
        noise_by_type[t] += int((~sl).sum())

    return n_ok, end, sig_by_type, noise_by_type


def main():
    out = Path("plots") / "hit_counts_signal_noise_by_type.png"
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 2, figsize=(14, 12), sharey=False)
    c_sig = "#2ca02c"
    c_noise = "#7f7f7f"

    for col, h5 in enumerate((H5_2020, H5_MERGED)):
        for row, split in enumerate(SPLITS):
            ax = axes[row, col]
            n_ok, _end, sig, noise = count_hits_split(h5, split, MAX_EVENTS, TRES_CUT)
            if n_ok is None:
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
                continue

            x = np.arange(3)
            w = 0.55
            ax.bar(x, sig, w, label="signal", color=c_sig, edgecolor="black", linewidth=0.5)
            ax.bar(
                x,
                noise,
                w,
                bottom=sig,
                label="noise",
                color=c_noise,
                edgecolor="black",
                linewidth=0.5,
            )
            ax.set_xticks(x)
            ax.set_xticklabels(
                [f"{n}\n(s:{sig[i]:,} n:{noise[i]:,})" for i, n in enumerate(TYPE_NAMES)]
            )
            tot_sig, tot_n = int(sig.sum()), int(noise.sum())
            ax.set_ylabel("Hits")
            st = f"{split.upper()}:  {n_ok:,} events,  {tot_sig + tot_n:,} hits  (signal {tot_sig:,} / noise {tot_n:,})"
            if row == 0:
                st = DATASET_NAMES[col] + "\n" + st
            ax.set_title(st, fontsize=8)
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        f"Signal vs noise hits by MC event type  (signal = |t_res|<{TRES_CUT} ns OR label>0)\n"
        f"Up to {MAX_EVENTS:,} events per split per dataset",
        fontweight="bold",
        fontsize=12,
    )
    plt.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
