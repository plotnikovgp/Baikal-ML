#!/usr/bin/env python3
"""Exploratory paired gate sweep on already saved single-checkpoint predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def angular_error(frame: pd.DataFrame, truth: np.ndarray) -> np.ndarray:
    theta, phi = np.deg2rad(truth[:, 0]), np.deg2rad(truth[:, 1])
    true_direction = np.column_stack(
        (np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta))
    )
    predicted = frame[["direction_x", "direction_y", "direction_z"]].to_numpy()
    predicted /= np.linalg.norm(predicted, axis=1, keepdims=True)
    return np.rad2deg(np.arccos(np.clip(np.sum(predicted * true_direction, axis=1), -1, 1)))


def quantiles(values: np.ndarray) -> list[float]:
    return [float(np.quantile(values, q)) for q in (0.5, 0.68, 0.95)]


def validate_photon_thresholds(values: list[float] | tuple[float, ...]) -> np.ndarray:
    thresholds = np.asarray(values, dtype=float)
    if not len(thresholds) or not np.isfinite(thresholds).all() or np.any(thresholds <= 0):
        raise ValueError("Photon thresholds must be finite and positive")
    if np.any(np.diff(thresholds) <= 0):
        raise ValueError("Photon thresholds must be strictly increasing")
    return thresholds


def paired_intervals(current: np.ndarray, baseline: np.ndarray, parts: np.ndarray) -> dict:
    rng = np.random.default_rng(20261013)
    groups = [np.flatnonzero(parts == part) for part in np.unique(parts)]
    event_deltas, part_deltas = [], []
    for _ in range(500):
        event_index = rng.integers(0, len(current), len(current))
        event_deltas.append(
            np.quantile(current[event_index], 0.68) - np.quantile(baseline[event_index], 0.68)
        )
        sampled = rng.integers(0, len(groups), len(groups))
        part_index = np.concatenate([groups[index] for index in sampled])
        part_deltas.append(
            np.quantile(current[part_index], 0.68) - np.quantile(baseline[part_index], 0.68)
        )
    return {
        "event_ci95_deg": np.quantile(event_deltas, [0.025, 0.975]).tolist(),
        "source_part_ci95_deg": np.quantile(part_deltas, [0.025, 0.975]).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--high", type=Path, required=True)
    parser.add_argument("--low", type=Path, required=True)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--photon-thresholds-ns",
        nargs="+",
        type=float,
        default=(5, 10, 20, 30, 50, 100, 200),
        help="Increasing positive thresholds; defaults to the original sweep",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    thresholds = validate_photon_thresholds(args.photon_thresholds_ns)
    high = pd.read_csv(args.high).set_index("event_id", verify_integrity=True)
    low = pd.read_csv(args.low).set_index("event_id", verify_integrity=True)
    if not high.index.isin(low.index).all():
        raise ValueError("High-threshold events are not contained in low-threshold CSV")
    low = low.reindex(high.index)
    with h5py.File(args.data) as h5:
        group = h5["test"]
        ids = np.asarray(group["ev_ids/data"]).astype(str)
        lookup = {event_id: index for index, event_id in enumerate(ids)}
        if len(lookup) != len(ids) or not high.index.isin(lookup).all():
            raise ValueError("HDF5 IDs do not cover unique prediction events")
        indices = np.asarray([lookup[event_id] for event_id in high.index], dtype=np.int64)
        truth = np.asarray(group["prime_prty/data"])[indices]
        starts = np.asarray(group["ev_starts/data"], dtype=np.int64)
        probability = np.asarray(group["signal_probability/data"])
        labels = np.asarray(group["labels/data"])
        marginal = probability < 0.7
        marginal_count_cum = np.r_[0, np.cumsum(marginal)]
        marginal_count = marginal_count_cum[starts[1:]] - marginal_count_cum[starts[:-1]]
        marginal_prob_cum = np.r_[0.0, np.cumsum(np.where(marginal, probability, 0.0))]
        marginal_prob_sum = marginal_prob_cum[starts[1:]] - marginal_prob_cum[starts[:-1]]
        marginal_signal_cum = np.r_[0, np.cumsum(marginal & (labels != 0))]
        marginal_signal_count = marginal_signal_cum[starts[1:]] - marginal_signal_cum[starts[:-1]]
        mean = np.asarray(h5["norm_param/mean"], dtype=np.float64)
        std = np.asarray(h5["norm_param/std"], dtype=np.float64)
        features = np.asarray(group["data/data"], dtype=np.float64) * std + mean
    error_high = angular_error(high, truth)
    error_low = angular_error(low, truth)
    added = low.n_hits.to_numpy() - high.n_hits.to_numpy()
    high_hits = high.n_hits.to_numpy()
    if np.any(added < 0):
        raise ValueError("Lower threshold removed hits")
    if not np.array_equal(added, marginal_count[indices]):
        raise ValueError("CSV hit difference does not match HDF5 marginal hits")
    upward = high.theta_deg.to_numpy() < 90
    result = {
        "n": len(high),
        "upward": int(upward.sum()),
        "with_added": int((added > 0).sum()),
        "baseline": quantiles(error_high),
        "global_low": quantiles(error_low),
        "gates": [],
        "added_histogram": {str(k): int((added == k).sum()) for k in range(11)},
        "added_gt_10": int((added > 10).sum()),
        "added_hit_bins": [],
        "photon_residual_gates": [],
        "inputs": {
            "data": str(args.data),
            "high": str(args.high),
            "low": str(args.low),
            "thresholds_ns": thresholds.tolist(),
        },
    }
    photon_max_abs_ns = np.full(len(high), np.inf)
    candidate_indices = np.flatnonzero(upward & (added > 0))
    directions = high[["direction_x", "direction_y", "direction_z"]].to_numpy()
    anchors = high[["anchor_x_m", "anchor_y_m", "anchor_z_m"]].to_numpy()
    for event in candidate_indices:
        lower, upper = starts[indices[event] : indices[event] + 2]
        selected = probability[lower:upper] >= 0.7
        extra = ~selected
        xyz = features[lower:upper, 2:5]
        times = features[lower:upper, 1]
        direction = directions[event] / np.linalg.norm(directions[event])
        delta = xyz - anchors[event]
        along = delta @ direction
        rho = np.linalg.norm(delta - along[:, None] * direction, axis=1)
        expected = (along + np.sqrt(1.36**2 - 1) * rho) / 0.299792458
        residual = times - expected
        photon_max_abs_ns[event] = np.max(np.abs(residual[extra] - np.median(residual[selected])))
    for threshold_ns in thresholds:
        gate = upward & (added > 0) & (photon_max_abs_ns < threshold_ns)
        candidate = np.where(gate, error_low, error_high)
        result["photon_residual_gates"].append(
            {
                "max_abs_ns": float(threshold_ns),
                "reruns": int(gate.sum()),
                "q50_q68_q95_deg": quantiles(candidate),
                "mean_error_delta_gate_deg": float(np.mean((error_low - error_high)[gate]))
                if gate.any()
                else None,
                "marginal_truth_signal_fraction": float(
                    np.sum(marginal_signal_count[indices][gate]) / np.sum(added[gate])
                )
                if gate.any()
                else None,
            }
        )
    if args.candidate is not None:
        candidate = pd.read_csv(args.candidate).set_index("event_id", verify_integrity=True)
        if not candidate.index.equals(high.index):
            raise ValueError("Candidate event IDs/order differ from baseline")
        used_low = candidate.selected_threshold.to_numpy() < 0.6
        expected_low = upward & (added > 0) & (photon_max_abs_ns < 50)
        candidate_error = angular_error(candidate, truth)
        parts = np.asarray([event_id.rsplit("_", 1)[0] for event_id in high.index])
        result["candidate"] = {
            "q50_q68_q95_deg": quantiles(candidate_error),
            "reruns": int(used_low.sum()),
            "gate_mismatches_at_50ns": int(np.count_nonzero(used_low != expected_low)),
            "q68_delta_deg": float(
                np.quantile(candidate_error, 0.68) - np.quantile(error_high, 0.68)
            ),
            "bootstrap": paired_intervals(candidate_error, error_high, parts),
        }
    for minimum, maximum in ((8, 12), (12, 16), (16, 24), (24, 40), (40, 1000)):
        chosen = upward & (added > 0) & (high_hits >= minimum) & (high_hits < maximum)
        if not chosen.any():
            continue
        result["added_hit_bins"].append(
            {
                "high_hits_range": [minimum, maximum],
                "n": int(chosen.sum()),
                "mean_error_delta_deg": float(np.mean((error_low - error_high)[chosen])),
                "marginal_truth_signal_fraction": float(
                    np.sum(marginal_signal_count[indices][chosen]) / np.sum(added[chosen])
                ),
                "mean_marginal_probability": float(
                    np.sum(marginal_prob_sum[indices][chosen]) / np.sum(added[chosen])
                ),
            }
        )
    for minimum_added in range(1, 9):
        gate = upward & (added >= minimum_added)
        candidate = np.where(gate, error_low, error_high)
        result["gates"].append(
            {
                "minimum_added": minimum_added,
                "reruns": int(gate.sum()),
                "q50_q68_q95_deg": quantiles(candidate),
                "mean_error_delta_gate_deg": float(np.mean((error_low - error_high)[gate]))
                if gate.any()
                else None,
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
