#!/usr/bin/env python3
"""Fixed-cohort quality and uncertainty audit for a released Baikal model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist


def quantiles(values: np.ndarray) -> dict[str, float]:
    return {f"q{int(q * 100)}": float(np.quantile(values, q)) for q in (0.5, 0.68, 0.95)}


def line_errors(frame: pd.DataFrame, true_direction: np.ndarray, point: np.ndarray) -> tuple:
    predicted_direction = frame[["direction_x", "direction_y", "direction_z"]].to_numpy(float)
    predicted_direction /= np.linalg.norm(predicted_direction, axis=1, keepdims=True)
    angle = np.rad2deg(
        np.arccos(np.clip(np.sum(predicted_direction * true_direction, axis=1), -1, 1))
    )
    delta = frame[["anchor_x_m", "anchor_y_m", "anchor_z_m"]].to_numpy(float) - point
    transverse = delta - np.sum(delta * true_direction, axis=1, keepdims=True) * true_direction
    return angle, np.linalg.norm(transverse, axis=1)


def coverage(frame: pd.DataFrame, angle: np.ndarray, line: np.ndarray, mask: np.ndarray) -> dict:
    if not mask.any():
        return {"available_n": 0}
    fields = (
        ("angle68", "angle_r68_deg", angle),
        ("angle95", "angle_r95_deg", angle),
        ("line68", "anchor_r68_m", line),
        ("line95", "anchor_r95_m", line),
    )
    available = mask.copy()
    for _, column, _ in fields:
        available &= np.isfinite(frame[column].to_numpy(float))
    result = {"available_n": int(available.sum())}
    if not available.any():
        return result
    for label, column, error in fields:
        radii = frame[column].to_numpy(float)[available]
        if np.any(radii <= 0):
            raise ValueError(f"Nonpositive calibrated radius in {column}")
        result[label] = float(np.mean(error[available] <= radii))
    return result


def source_part_intervals(
    baseline: dict[str, np.ndarray], gate: dict[str, np.ndarray], parts: np.ndarray
) -> dict:
    rng = np.random.default_rng(20261014)
    groups = [np.flatnonzero(parts == part) for part in np.unique(parts)]
    samples = {f"{metric}_{name}": [] for metric in ("angle", "line") for name in ("q50", "q68")}
    for _ in range(400):
        selected = np.concatenate(
            [groups[index] for index in rng.integers(0, len(groups), len(groups))]
        )
        for metric in ("angle", "line"):
            for name, q in (("q50", 0.5), ("q68", 0.68)):
                samples[f"{metric}_{name}"].append(
                    np.quantile(gate[metric][selected], q)
                    - np.quantile(baseline[metric][selected], q)
                )
    return {key: np.quantile(values, [0.025, 0.975]).tolist() for key, values in samples.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--truth-points", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    baseline = pd.read_csv(args.baseline)
    gate = pd.read_csv(args.gate)
    if not baseline.event_id.is_unique or not gate.event_id.is_unique:
        raise ValueError("Duplicate prediction IDs")
    if not baseline.event_id.equals(gate.event_id):
        raise ValueError("Baseline and gate do not share identical ordered event IDs")
    if not np.array_equal(baseline.source_event_index, gate.source_event_index):
        raise ValueError("Baseline and gate source row indices differ")
    ids = baseline.event_id.to_numpy(str)
    source_index = baseline.source_event_index.to_numpy(np.int64)

    with np.load(args.truth_points) as truth:
        point_ids = truth["event_ids"].astype(str)
        if len(np.unique(point_ids)) != len(point_ids):
            raise ValueError("Truth-point IDs are not unique")
        lookup = pd.Index(point_ids).get_indexer(ids)
        if np.any(lookup < 0) or not truth["valid"][lookup].all():
            raise ValueError("Missing or invalid truth line in selected cohort")
        true_point = np.asarray(truth["point_m"][lookup], dtype=float)
        prime = np.asarray(truth["prime"][lookup], dtype=float)
    with h5py.File(args.data) as h5:
        group = h5["test"]
        source_ids = np.asarray(group["ev_ids/data"][source_index]).astype(str)
        if not np.array_equal(ids, source_ids):
            raise ValueError("Prediction IDs do not match selected HDF5 rows")
        source_prime = np.asarray(group["prime_prty/data"][source_index])
        if not np.allclose(prime[:, :2], source_prime[:, :2], atol=1e-4, rtol=0):
            raise ValueError("Truth-point and selected-HDF5 angles differ")
        starts = np.asarray(group["ev_starts/data"])
        prob = np.asarray(group["signal_probability/data"])
        labels = np.asarray(group["labels/data"])
        xyz = np.asarray(group["data/data"][:, 2:5], dtype=float)
        xyz = xyz * np.asarray(h5["norm_param/std"])[2:5] + np.asarray(h5["norm_param/mean"])[2:5]
    selected_span = np.empty(len(ids), dtype=float)
    selected_purity = np.empty(len(ids), dtype=float)
    selected_track_fraction = np.empty(len(ids), dtype=float)
    for row, index in enumerate(source_index):
        lo, hi = starts[index : index + 2]
        keep = prob[lo:hi] >= 0.7
        if int(keep.sum()) != int(baseline.n_hits.iloc[row]):
            raise ValueError(f"Baseline hit count differs from HDF5 at row {row}")
        event_xyz = xyz[lo:hi][keep]
        event_labels = labels[lo:hi][keep]
        selected_span[row] = float(np.max(pdist(event_xyz)))
        selected_purity[row] = float(np.mean(event_labels != 0))
        signal = np.count_nonzero(event_labels != 0)
        selected_track_fraction[row] = (
            float(np.count_nonzero(event_labels > 0) / signal) if signal else 0.0
        )

    theta, phi = np.deg2rad(prime[:, 0]), np.deg2rad(prime[:, 1])
    true_direction = np.column_stack(
        (np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta))
    )
    errors = {}
    for name, frame in (("baseline", baseline), ("gate", gate)):
        angle, line = line_errors(frame, true_direction, true_point)
        errors[name] = {"angle": angle, "line": line}
    rerun = gate.selected_threshold.to_numpy(float) < 0.6
    if not (
        baseline[["angle_r68_deg", "angle_r95_deg", "anchor_r68_m", "anchor_r95_m"]]
        .notna()
        .all()
        .all()
    ):
        raise ValueError("Baseline has missing calibrated radii")
    if (
        gate.loc[rerun, ["angle_r68_deg", "angle_r95_deg", "anchor_r68_m", "anchor_r95_m"]]
        .notna()
        .any()
        .any()
    ):
        raise ValueError("Rerun rows must not advertise validated radii")

    def describe(mask: np.ndarray) -> dict:
        if mask.sum() < 30:
            return {"n": int(mask.sum())}
        return {
            "n": int(mask.sum()),
            "reruns": int(np.sum(rerun & mask)),
            "baseline": {
                metric: quantiles(value[mask]) for metric, value in errors["baseline"].items()
            },
            "gate": {metric: quantiles(value[mask]) for metric, value in errors["gate"].items()},
            "baseline_coverage": coverage(
                baseline, errors["baseline"]["angle"], errors["baseline"]["line"], mask
            ),
            "gate_coverage_on_unchanged_rows": coverage(
                gate, errors["gate"]["angle"], errors["gate"]["line"], mask & ~rerun
            ),
        }

    report = {
        "inputs": {key: str(value) for key, value in vars(args).items() if key != "output"},
        "definitions": {
            "angle": "3D direction separation in degrees",
            "line": "predicted anchor transverse distance to the MC truth line in metres",
            "span": "maximum Euclidean separation of p>=0.70 classifier-selected hits in metres",
            "coverage": "fraction within checkpoint-provided 68% or 95% radius; gate reruns have no calibrated radii",
        },
        "source_parts": int(len(np.unique([event_id.rsplit("_", 1)[0] for event_id in ids]))),
        "all": describe(np.ones(len(ids), dtype=bool)),
        "unchanged": describe(~rerun),
        "rerun": describe(rerun),
        "source_part_bootstrap_gate_minus_baseline_ci95": source_part_intervals(
            errors["baseline"],
            errors["gate"],
            np.asarray([event_id.rsplit("_", 1)[0] for event_id in ids]),
        ),
        "slices": {},
        "uncertainty_deciles": [],
    }
    features = {
        "selected_hits": (baseline.n_hits.to_numpy(float), [8, 12, 16, 24, 40, np.inf]),
        "selected_span_m": (selected_span, [0, 100, 200, 300, 450, np.inf]),
        "truth_theta_deg": (prime[:, 0], [0, 30, 60, 90, 120, 180.001]),
        "selected_signal_purity": (selected_purity, [0, 0.9, 0.99, 1.001]),
        "selected_track_fraction_of_signal": (selected_track_fraction, [0, 0.1, 0.5, 0.9, 1.001]),
    }
    for name, (values, edges) in features.items():
        report["slices"][name] = {}
        for low, high in zip(edges[:-1], edges[1:]):
            report["slices"][name][f"{low:g}-{high:g}"] = describe(
                (values >= low) & (values < high)
            )
    ordered = np.argsort(baseline.sigma_angle_deg.to_numpy(float))
    for decile, sample in enumerate(np.array_split(ordered, 10), start=1):
        mask = np.zeros(len(ids), dtype=bool)
        mask[sample] = True
        report["uncertainty_deciles"].append(
            {
                "decile": decile,
                "median_sigma_angle_deg": float(np.median(baseline.sigma_angle_deg.iloc[sample])),
                **describe(mask),
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"all": report["all"], "source_parts": report["source_parts"]}, indent=2))


if __name__ == "__main__":
    main()
