#!/usr/bin/env python3
"""Paper-style hit precision/recall panels for MC particle types.

Input HDF5s must contain *all* classifier-scored raw hits (builder threshold 0).
The 8-hit/2-string event cut is recalculated at every probability threshold.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PARTICLES = (
    ("muatm", "Atmospheric muons (muatm)"),
    ("nuatm", "Atmospheric neutrinos (nuatm)"),
    ("nue2", "High-energy neutrinos (nue2)"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for particle, _ in PARTICLES:
        parser.add_argument(f"--{particle}", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--channels-per-string", type=int, default=36)
    return parser.parse_args()


def load_scores(path: Path, particle: str, channels_per_string: int) -> dict:
    with h5py.File(path) as h5:
        meta = json.loads(h5.attrs["metadata_json"])
        group = h5["test"]
        starts = group["ev_starts/data"][:]
        score = group["signal_probability/data"][:]
        channels = group["channels/data"][:]
        labels = group["labels/data"][:]
        ids = group["ev_ids/data"][:]
    if meta["arguments"]["threshold"] != 0.0:
        raise ValueError(f"{path}: classifier threshold must be 0 to retain all raw hits")
    if meta["arguments"]["particle"] != particle:
        raise ValueError(f"{path}: particle does not match {particle}")
    if len(ids) != len(starts) - 1 or starts[-1] != len(score):
        raise ValueError(f"{path}: inconsistent event/hit lengths")
    return {
        "score": score,
        "event": np.repeat(np.arange(len(ids), dtype=np.int32), np.diff(starts)),
        "string": channels.astype(np.int32) // channels_per_string,
        "truth": labels != 0,
        "n_events": len(ids),
        "meta": meta,
    }


def measure(sample: dict, threshold: float) -> dict:
    score = sample["score"]
    event = sample["event"]
    string = sample["string"]
    truth = sample["truth"]
    n_events = sample["n_events"]
    n_strings = int(string.max()) + 1
    selected = score >= threshold
    selected_event = event[selected]
    n_hits = np.bincount(selected_event, minlength=n_events)
    string_key = selected_event.astype(np.int64) * n_strings + string[selected]
    occupied = np.bincount(string_key, minlength=n_events * n_strings)
    n_unique_strings = np.count_nonzero(occupied.reshape(n_events, n_strings), axis=1)
    retained = (n_hits >= 8) & (n_unique_strings >= 2)
    retained_hits = retained[event]
    selected_retained = selected & retained_hits
    tp = int(np.count_nonzero(selected_retained & truth))
    fp = int(np.count_nonzero(selected_retained & ~truth))
    signal_total = int(np.count_nonzero(retained_hits & truth))
    return {
        "threshold": threshold,
        "retained_events": int(retained.sum()),
        "truth_signal_selected": tp,
        "truth_noise_selected": fp,
        "truth_signal_hits_in_retained_events": signal_total,
        "precision": tp / (tp + fp) if tp + fp else float("nan"),
        "recall": tp / signal_total if signal_total else float("nan"),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    thresholds = [round(float(x), 2) for x in np.arange(0.02, 1.0, 0.02)]
    by_particle = {}
    manifests = {}
    for particle, _ in PARTICLES:
        path = getattr(args, particle)
        sample = load_scores(path, particle, args.channels_per_string)
        by_particle[particle] = [
            {"particle": particle, **measure(sample, threshold)} for threshold in thresholds
        ]
        manifests[particle] = {
            "input": str(path),
            "sample_events": sample["n_events"],
            "source": sample["meta"]["arguments"]["source"],
            "sample_stride": sample["meta"]["arguments"].get("sample_stride", 1),
            "sample_offset": sample["meta"]["arguments"].get("sample_offset", 0),
            "classifier_checkpoint": sample["meta"]["arguments"]["checkpoint"],
        }

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True, layout="constrained")
    for ax, (particle, title) in zip(axes, PARTICLES, strict=True):
        rows = by_particle[particle]
        x = np.array([r["threshold"] for r in rows])
        precision = np.array([r["precision"] for r in rows])
        recall = np.array([r["recall"] for r in rows])
        ax.plot(x, precision, color="#d95f02", linewidth=2.2, label="precision")
        ax.plot(x, recall, color="#009e73", linewidth=2.2, linestyle="--", label="recall")
        ax.set_title(title)
        ax.set_xlim(0, 1)
        ax.set_ylim(0.85, 1.005)
        ax.set_xticks(np.arange(0, 1.01, 0.2))
        ax.set_yticks(np.arange(0.86, 1.001, 0.02))
        ax.set_xlabel("Classifier threshold p")
        ax.grid(True, alpha=0.2, which="both")
        ax.minorticks_on()
        ax.grid(True, alpha=0.08, which="minor")
        ax.legend(loc="lower center", fontsize=9)
    axes[0].set_ylabel("Hit-level precision / recall")
    fig.suptitle("Signal-hit selection after ≥8 hits on ≥2 strings · 2020 MC test", fontsize=13)
    fig.savefig(args.output_dir / "precision_recall_three_particles.png", dpi=180)
    plt.close(fig)

    rows = [row for particle, _ in PARTICLES for row in by_particle[particle]]
    with (args.output_dir / "precision_recall_three_particles.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "cohorts": manifests,
        "truth_signal": "MC hit label != 0 (track or cascade)",
        "truth_noise": "MC hit label == 0",
        "selection": "At each threshold: >=8 selected hits on >=2 unique strings",
        "precision_denominator": "all threshold-selected hits in retained events",
        "recall_denominator": "all truth-signal hits in retained events, including rejected hits",
        "threshold_range": [thresholds[0], thresholds[-1]],
        "figure_y_axis": [0.85, 1.005],
        "figure_y_axis_cropped": True,
        "full_values_in_csv": True,
    }
    (args.output_dir / "precision_recall_three_particles.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    summary = {
        particle: [row for row in by_particle[particle] if row["threshold"] in (0.5, 0.7)]
        for particle, _ in PARTICLES
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
