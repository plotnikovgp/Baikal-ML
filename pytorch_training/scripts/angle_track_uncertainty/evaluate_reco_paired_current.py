#!/usr/bin/env python3
"""Paired current-network/standard-reco comparison on the corrected reco test set.

The input prediction CSVs must be produced on the same p>=0.70 hit-selected
HDF5.  Only events present in the fair all-GT-signal sidecar are compared.
Official-quality cuts are applied to both predictions on identical event IDs.
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
from scipy.spatial.distance import pdist


def angles_to_vectors(theta_deg: np.ndarray, phi_deg: np.ndarray) -> np.ndarray:
    theta, phi = np.deg2rad(theta_deg), np.deg2rad(phi_deg)
    return np.column_stack(
        (np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta))
    )


def angular_errors(pred: np.ndarray, target: np.ndarray) -> dict[str, np.ndarray]:
    pred = pred / np.linalg.norm(pred, axis=1, keepdims=True)
    target = target / np.linalg.norm(target, axis=1, keepdims=True)
    direction = np.rad2deg(np.arccos(np.clip((pred * target).sum(1), -1, 1)))
    pred_theta = np.rad2deg(np.arccos(np.clip(pred[:, 2], -1, 1)))
    true_theta = np.rad2deg(np.arccos(np.clip(target[:, 2], -1, 1)))
    pred_phi = np.rad2deg(np.arctan2(pred[:, 1], pred[:, 0]))
    true_phi = np.rad2deg(np.arctan2(target[:, 1], target[:, 0]))
    phi = np.abs((pred_phi - true_phi + 180) % 360 - 180)
    return {"direction": direction, "theta": abs(pred_theta - true_theta), "phi": phi}


def read_predictions(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    ids = np.asarray([row["event_id"] for row in rows])
    if len(set(ids)) != len(ids):
        raise ValueError(f"Duplicate event IDs in {path}")
    vectors = np.asarray(
        [[float(row[f"direction_{axis}"]) for axis in "xyz"] for row in rows],
        dtype=np.float64,
    )
    if not np.isfinite(vectors).all():
        raise ValueError(f"Non-finite predictions in {path}")
    return ids, vectors


def gt_spans(
    group: h5py.Group, indices: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    starts = group["ev_starts/data"]
    features = group["data/data"]
    lengths = np.zeros(len(indices), dtype=np.float32)
    for out, index in enumerate(indices):
        low, high = int(starts[index]), int(starts[index + 1])
        xyz = np.asarray(features[low:high, 2:5], dtype=np.float64) * std[2:5] + mean[2:5]
        if len(xyz) > 1:
            lengths[out] = float(np.max(pdist(xyz)))
    return lengths


def quantiles(values: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    selected = values[mask]
    return {"q50": float(np.quantile(selected, 0.5)), "q68": float(np.quantile(selected, 0.68))}


def make_plot(rows: list[dict], output: Path, stage: str, comparison_label: str) -> None:
    current = [r for r in rows if r["stage"] == stage]
    x = np.asarray([r["min_gt_span_m"] for r in current])
    colors = {
        "standard_reco": "#d62728",
        "v4": "#999999",
        "v4_photon": "#1f77b4",
        "short_photon": "#2ca02c",
    }
    names = {
        "standard_reco": "Standard reco",
        "v4": "Universal v4",
        "v4_photon": "v4 + photon",
        "short_photon": comparison_label,
    }
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for axis, component in zip(axes, ("direction", "theta", "phi")):
        for model in names:
            for q, style in (("q50", "-"), ("q68", "--")):
                axis.plot(
                    x,
                    [r[f"{model}_{component}_{q}_deg"] for r in current],
                    color=colors[model],
                    linestyle=style,
                    marker="o",
                    markersize=3,
                    label=f"{names[model]} {q}",
                )
        axis.set(
            title=f"{component.title()} resolution",
            xlabel="Minimum GT-signal span [m]",
            ylabel="Error [deg]",
        )
        axis.grid(alpha=0.3)
        axis.set_ylim(bottom=0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9)
    fig.suptitle(f"Reco nue2 test, predicted-signal p≥0.70, ≥8 hits/≥2 strings — {stage}")
    fig.tight_layout(rect=(0, 0.16, 1, 0.94))
    fig.savefig(output / f"reco_paired_{stage}.png", dpi=190)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fair-source", type=Path, required=True)
    parser.add_argument("--quality-sidecar", type=Path, required=True)
    parser.add_argument(
        "--prediction", nargs=2, action="append", metavar=("NAME", "CSV"), required=True
    )
    parser.add_argument(
        "--comparison-label",
        default="Short + photon",
        help="Plot label for the short_photon comparison slot",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions = {name: read_predictions(Path(path)) for name, path in args.prediction}
    if set(predictions) != {"v4", "v4_photon", "short_photon"}:
        raise ValueError("Require v4, v4_photon, and short_photon prediction CSVs")
    ids = predictions["v4"][0]
    for name, (other_ids, _) in predictions.items():
        if not np.array_equal(ids, other_ids):
            raise ValueError(f"Prediction IDs/order mismatch: {name}")

    with h5py.File(args.fair_source, "r") as fair, h5py.File(args.quality_sidecar, "r") as quality:
        group = fair["test"]
        fair_ids = np.asarray(group["ev_ids/data"])
        quality_ids = np.asarray(quality["test/event_ids"])
        if not np.array_equal(fair_ids, quality_ids):
            raise ValueError("Fair and quality sidecar IDs differ")
        fair_lookup = {event_id.decode(): index for index, event_id in enumerate(fair_ids)}
        keep = np.asarray(
            [i for i, event_id in enumerate(ids) if event_id in fair_lookup], dtype=np.int64
        )
        fair_indices = np.asarray([fair_lookup[ids[i]] for i in keep], dtype=np.int64)
        if len(set(fair_indices)) != len(fair_indices):
            raise ValueError("Duplicate fair-source indices")
        ids = ids[keep]
        truth_angles = np.asarray(group["prime_prty/data"][:, :2])[fair_indices]
        reco_angles = np.column_stack(
            (
                np.asarray(quality["test/theta_deg"])[fair_indices],
                np.asarray(quality["test/phi_deg"])[fair_indices],
            )
        )
        official = np.asarray(quality["test/quality_mask"], dtype=bool)[fair_indices]
        mean = np.asarray(fair["norm_param/mean"], dtype=np.float64)
        std = np.asarray(fair["norm_param/std"], dtype=np.float64)
        spans = gt_spans(group, fair_indices, mean, std)

    truth = angles_to_vectors(truth_angles[:, 0], truth_angles[:, 1])
    pred_vectors = {name: vectors[keep] for name, (_, vectors) in predictions.items()}
    pred_vectors["standard_reco"] = angles_to_vectors(reco_angles[:, 0], reco_angles[:, 1])
    errors = {name: angular_errors(vector, truth) for name, vector in pred_vectors.items()}
    thresholds = (0, 100, 150, 200, 250, 300, 350, 400, 450)
    rows = []
    for stage, base in (
        ("no_official_cut", np.ones(len(ids), dtype=bool)),
        ("official_reco_cut", official),
    ):
        for threshold in thresholds:
            mask = base & (spans >= threshold)
            row = {"stage": stage, "min_gt_span_m": threshold, "count": int(mask.sum())}
            if not mask.any():
                continue
            for name, components in errors.items():
                for component, values in components.items():
                    for q, value in quantiles(values, mask).items():
                        row[f"{name}_{component}_{q}_deg"] = value
            rows.append(row)

    with (args.output_dir / "reco_paired_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "selection": "predicted signal p>=0.70; >=8 selected hits on >=2 unique strings; paired fair reco test IDs",
        "official_cut": "fCovMatrixStatus==3 and fInitRecDelta<20 and fFuncValue/(fNHits-5)<30",
        "span_definition": "maximum 3D separation between GT-signal hits (not reco path length)",
        "fair_source": str(args.fair_source),
        "quality_sidecar": str(args.quality_sidecar),
        "prediction_files": {name: str(path) for name, path in args.prediction},
        "comparison_label": args.comparison_label,
        "predicted_selected_events": len(predictions["v4"][0]),
        "paired_events": len(ids),
        "official_pass_events": int(official.sum()),
        "rows": rows,
    }
    (args.output_dir / "reco_paired_metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    np.savez_compressed(
        args.output_dir / "reco_paired_events.npz",
        event_ids=ids,
        fair_index=fair_indices,
        gt_span_m=spans,
        official_cut=official,
        **{
            f"{name}_{component}_error_deg": values
            for name, components in errors.items()
            for component, values in components.items()
        },
    )
    for stage in ("no_official_cut", "official_reco_cut"):
        make_plot(rows, args.output_dir, stage, args.comparison_label)
    print(
        json.dumps(
            {
                "paired_events": len(ids),
                "official_pass_events": int(official.sum()),
                "zero_span_rows": [r for r in rows if r["min_gt_span_m"] == 0],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
