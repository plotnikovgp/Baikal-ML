#!/usr/bin/env python3
"""Paired angle/line audit of a two-pass anchor midpoint on independent MC."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def direction_from_angles(prime: np.ndarray) -> np.ndarray:
    theta, phi = np.deg2rad(prime[:, 0]), np.deg2rad(prime[:, 1])
    return np.column_stack(
        (np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta))
    )


def quantiles(values: np.ndarray) -> list[float]:
    return np.quantile(values, [0.5, 0.68, 0.95]).tolist()


def bootstrap_change(
    candidate: np.ndarray,
    baseline: np.ndarray,
    parts: np.ndarray,
    quantile: float,
    rng: np.random.Generator,
    trials: int,
) -> dict[str, list[float]]:
    unique_parts = np.unique(parts)
    part_indices = [np.flatnonzero(parts == part) for part in unique_parts]
    event_changes, part_changes = [], []
    for _ in range(trials):
        events = rng.integers(0, len(candidate), len(candidate))
        source_parts = np.concatenate(
            [part_indices[j] for j in rng.integers(0, len(unique_parts), len(unique_parts))]
        )
        event_changes.append(
            np.quantile(candidate[events], quantile) - np.quantile(baseline[events], quantile)
        )
        part_changes.append(
            np.quantile(candidate[source_parts], quantile)
            - np.quantile(baseline[source_parts], quantile)
        )
    return {
        "event_ci95": np.quantile(event_changes, [0.025, 0.975]).tolist(),
        "source_part_ci95": np.quantile(part_changes, [0.025, 0.975]).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--truth-points", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=500)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    old, new = pd.read_csv(args.baseline), pd.read_csv(args.candidate)
    if not old.event_id.equals(new.event_id):
        raise ValueError("Prediction event IDs/order differ")
    with np.load(args.truth_points) as points:
        truth_ids = points["event_ids"].astype(str)
        valid = points["valid"].astype(bool)
        prime = points["prime"]
        true_point = points["point_m"]
    _, inverse, counts = np.unique(truth_ids, return_inverse=True, return_counts=True)
    valid &= counts[inverse] == 1
    eligible_ids = pd.Index(truth_ids[valid])
    pred_unique = old.event_id.map(old.event_id.value_counts()).eq(1).to_numpy()
    pred_valid = pred_unique & old.event_id.isin(eligible_ids).to_numpy()
    excluded_prediction_rows = int((~pred_valid).sum())
    old, new = old.loc[pred_valid].copy(), new.loc[pred_valid].copy()
    indices = eligible_ids.get_indexer(old.event_id.to_numpy(str))
    if (indices < 0).any():
        raise ValueError("Missing truth IDs")
    truth_rows = np.flatnonzero(valid)[indices]
    truth_direction = direction_from_angles(prime[truth_rows])
    target = true_point[truth_rows]
    base_point = old[["anchor_x_m", "anchor_y_m", "anchor_z_m"]].to_numpy()
    new_point = new[["anchor_x_m", "anchor_y_m", "anchor_z_m"]].to_numpy()
    mid_point = 0.5 * (base_point + new_point)

    def line_error(point: np.ndarray) -> np.ndarray:
        delta = point - target
        transverse = (
            delta - np.sum(delta * truth_direction, axis=1, keepdims=True) * truth_direction
        )
        return np.linalg.norm(transverse, axis=1)

    def angle_error(frame: pd.DataFrame) -> np.ndarray:
        direction = frame[["direction_x", "direction_y", "direction_z"]].to_numpy()
        direction /= np.linalg.norm(direction, axis=1, keepdims=True)
        return np.rad2deg(np.arccos(np.clip(np.sum(direction * truth_direction, axis=1), -1, 1)))

    angle_base, angle_new = angle_error(old), angle_error(new)
    line_base, line_new, line_mid = map(line_error, (base_point, new_point, mid_point))
    parts = np.asarray(
        ["_".join(event_id.split("_")[1:-1]) for event_id in old.event_id],
        dtype=str,
    )
    if len(np.unique(parts)) < 2:
        raise ValueError("Part bootstrap requires at least two source parts")
    rng = np.random.default_rng(20261013)
    result = {
        "n": len(old),
        "excluded_prediction_rows_without_unique_truth": excluded_prediction_rows,
        "source_parts": len(np.unique(parts)),
        "rerun": int((new.selected_threshold.to_numpy() < 0.6).sum()),
        "angle_baseline_q50_q68_q95_deg": quantiles(angle_base),
        "angle_candidate_q50_q68_q95_deg": quantiles(angle_new),
        "line_baseline_q50_q68_q95_m": quantiles(line_base),
        "line_candidate_q50_q68_q95_m": quantiles(line_new),
        "line_midpoint_q50_q68_q95_m": quantiles(line_mid),
        "changes": {},
    }
    for label, candidate, baseline in (
        ("angle_candidate_minus_baseline_q68_deg", angle_new, angle_base),
        ("line_candidate_minus_baseline_q68_m", line_new, line_base),
        ("line_candidate_minus_baseline_q95_m", line_new, line_base),
        ("line_midpoint_minus_baseline_q68_m", line_mid, line_base),
        ("line_midpoint_minus_baseline_q95_m", line_mid, line_base),
        ("line_midpoint_minus_candidate_q95_m", line_mid, line_new),
    ):
        quantile = 0.95 if "q95" in label else 0.68
        result["changes"][label] = bootstrap_change(
            candidate, baseline, parts, quantile, rng, args.bootstrap
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
