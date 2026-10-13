#!/usr/bin/env python3
"""Paired independent-MC audit of a point-only checkpoint and hit-time gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def quantiles(values: np.ndarray) -> list[float]:
    return np.quantile(values, [0.5, 0.68, 0.95]).tolist()


def source_part_ci(
    new: np.ndarray,
    old: np.ndarray,
    parts: np.ndarray,
    quantile: float,
    trials: int,
) -> list[float]:
    rng = np.random.default_rng(20261013)
    groups = [np.flatnonzero(parts == part) for part in np.unique(parts)]
    changes = []
    for _ in range(trials):
        selected = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        changes.append(np.quantile(new[selected], quantile) - np.quantile(old[selected], quantile))
    return np.quantile(changes, [0.025, 0.975]).tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-p070", type=Path, required=True)
    parser.add_argument("--candidate-p070", type=Path, required=True)
    parser.add_argument("--baseline-gate", type=Path, required=True)
    parser.add_argument("--candidate-gate", type=Path, required=True)
    parser.add_argument("--truth-points", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=500)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    frames = {
        name: pd.read_csv(path)
        for name, path in (
            ("baseline_p070", args.baseline_p070),
            ("candidate_p070", args.candidate_p070),
            ("baseline_gate", args.baseline_gate),
            ("candidate_gate", args.candidate_gate),
        )
    }
    common_ids = frames["baseline_p070"].event_id
    if not all(frame.event_id.equals(common_ids) for frame in frames.values()):
        raise ValueError("Prediction files do not share the same fixed event cohort/order")
    with np.load(args.truth_points) as truth:
        ids = truth["event_ids"].astype(str)
        valid = truth["valid"].astype(bool)
        prime = truth["prime"]
        truth_points = truth["point_m"]
    _, inverse, counts = np.unique(ids, return_inverse=True, return_counts=True)
    valid &= counts[inverse] == 1
    eligible = pd.Index(ids[valid])
    prediction_unique = common_ids.map(common_ids.value_counts()).eq(1).to_numpy()
    selected = prediction_unique & common_ids.isin(eligible).to_numpy()
    excluded = int((~selected).sum())
    frames = {name: frame.loc[selected].copy() for name, frame in frames.items()}
    common_ids = frames["baseline_p070"].event_id.to_numpy(str)
    lookup = eligible.get_indexer(common_ids)
    if (lookup < 0).any():
        raise ValueError("Missing event truth")
    truth_rows = np.flatnonzero(valid)[lookup]
    theta, phi = np.deg2rad(prime[truth_rows, 0]), np.deg2rad(prime[truth_rows, 1])
    direction = np.column_stack(
        (np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta))
    )
    target = truth_points[truth_rows]
    parts = np.asarray(["_".join(event_id.split("_")[1:-1]) for event_id in common_ids])
    result = {
        "n": len(common_ids),
        "excluded_nonunique_or_invalid": excluded,
        "source_parts": len(np.unique(parts)),
        "gate_reruns": {},
        "methods": {},
        "changes": {},
    }
    errors = {}
    for name, frame in frames.items():
        pred_direction = frame[["direction_x", "direction_y", "direction_z"]].to_numpy()
        pred_direction /= np.linalg.norm(pred_direction, axis=1, keepdims=True)
        angle = np.rad2deg(np.arccos(np.clip(np.sum(pred_direction * direction, axis=1), -1, 1)))
        point = frame[["anchor_x_m", "anchor_y_m", "anchor_z_m"]].to_numpy()
        delta = point - target
        transverse = delta - np.sum(delta * direction, axis=1, keepdims=True) * direction
        line = np.linalg.norm(transverse, axis=1)
        errors[name] = {"angle": angle, "line": line}
        result["methods"][name] = {
            "angle_q50_q68_q95_deg": quantiles(angle),
            "line_q50_q68_q95_m": quantiles(line),
        }
    base_rerun = frames["baseline_gate"].selected_threshold.to_numpy() < 0.6
    cand_rerun = frames["candidate_gate"].selected_threshold.to_numpy() < 0.6
    result["gate_reruns"] = {
        "baseline": int(base_rerun.sum()),
        "candidate": int(cand_rerun.sum()),
        "different_decisions": int(np.count_nonzero(base_rerun != cand_rerun)),
    }
    for label, new_name, old_name in (
        ("point_only_p070_vs_baseline_p070", "candidate_p070", "baseline_p070"),
        ("point_only_gate_vs_baseline_gate", "candidate_gate", "baseline_gate"),
        ("point_only_gate_vs_baseline_p070", "candidate_gate", "baseline_p070"),
    ):
        result["changes"][label] = {}
        for metric, unit in (("angle", "deg"), ("line", "m")):
            for q in (0.68, 0.95):
                result["changes"][label][f"{metric}_q{round(q * 100)}_ci95_{unit}"] = (
                    source_part_ci(
                        errors[new_name][metric],
                        errors[old_name][metric],
                        parts,
                        q,
                        args.bootstrap,
                    )
                )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
