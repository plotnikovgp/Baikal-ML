#!/usr/bin/env python3
"""Paired MC angle quality for the fixed 2020 ID-masked and deduplicated test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def direction_error(frame: pd.DataFrame, truth: np.ndarray) -> np.ndarray:
    theta, phi = np.deg2rad(truth[:, 0]), np.deg2rad(truth[:, 1])
    target = np.column_stack(
        (np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta))
    )
    direction = frame[["direction_x", "direction_y", "direction_z"]].to_numpy()
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    return np.rad2deg(np.arccos(np.clip(np.sum(direction * target, axis=1), -1, 1)))


def q(values: np.ndarray) -> list[float]:
    return np.quantile(values, [0.5, 0.68, 0.95]).tolist()


def paired_interval(new: np.ndarray, old: np.ndarray) -> list[float]:
    rng = np.random.default_rng(20261013)
    delta = []
    for _ in range(500):
        selected = rng.integers(0, len(new), len(new))
        delta.append(np.quantile(new[selected], 0.68) - np.quantile(old[selected], 0.68))
    return np.quantile(delta, [0.025, 0.975]).tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--prepared-p050", type=Path, required=True)
    parser.add_argument("--prepared-p070", type=Path, required=True)
    parser.add_argument("--masks", type=Path, required=True)
    parser.add_argument("--anchor-targets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    old = pd.read_csv(args.baseline)
    new = pd.read_csv(args.candidate)
    if not old.event_id.equals(new.event_id):
        raise ValueError("Candidate event IDs/order differ")
    with h5py.File(args.prepared_p070) as h5:
        original_ids = np.asarray(h5["test/ev_ids/data"][:]).astype(str)
    if not np.array_equal(original_ids, old.event_id.to_numpy(dtype=str)):
        raise ValueError("ID mask cannot be aligned with baseline predictions")
    with np.load(args.masks) as masks:
        leak_free = np.asarray(masks["pred_test"], dtype=bool)
    if len(leak_free) != len(old):
        raise ValueError("ID mask length differs from predictions")
    old = old.loc[leak_free].drop_duplicates("event_id", keep="first")
    new = new.loc[leak_free].drop_duplicates("event_id", keep="first")
    if not old.event_id.equals(new.event_id):
        raise ValueError("Deduplicated paired events differ")
    indices = new.source_event_index.to_numpy(dtype=np.int64)
    with h5py.File(args.prepared_p050) as h5:
        ids = np.asarray(h5["test/ev_ids/data"])[indices].astype(str)
        truth = np.asarray(h5["test/prime_prty/data"])[indices]
    if not np.array_equal(ids, new.event_id.to_numpy(dtype=str)):
        raise ValueError("p>=0.50 HDF5 event mapping failed")
    with h5py.File(args.anchor_targets) as h5:
        anchors = h5["test"]
        original_indices = old.index.to_numpy(dtype=np.int64)
        target_ids = np.asarray(anchors["event_ids"])[original_indices].astype(str)
        if not np.array_equal(target_ids, new.event_id.to_numpy(dtype=str)):
            raise ValueError("Truth-line anchor IDs differ from selected events")
        line_valid = np.asarray(anchors["valid"])[original_indices].astype(bool)
        line_points = np.asarray(anchors["anchor_point_m"])[original_indices]
    old_error = direction_error(old, truth)
    new_error = direction_error(new, truth)
    theta, phi = np.deg2rad(truth[:, 0]), np.deg2rad(truth[:, 1])
    true_direction = np.column_stack(
        (np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta))
    )
    line_errors = {}
    for label, frame in (("baseline", old), ("candidate", new)):
        point = frame[["anchor_x_m", "anchor_y_m", "anchor_z_m"]].to_numpy()
        delta = point - line_points
        transverse = delta - np.sum(delta * true_direction, axis=1, keepdims=True) * true_direction
        line_errors[label] = np.linalg.norm(transverse, axis=1)
    particles = new.event_id.str.split("_", n=1).str[0].to_numpy()
    stages = {"all": np.ones(len(new), dtype=bool)}
    stages.update({name: particles == name for name in ("nue2", "nuatm", "muatm")})
    result = {"mask": "pred_test from masks_v2, then first exact event ID", "stages": {}}
    for name, selected in stages.items():
        old_subset, new_subset = old_error[selected], new_error[selected]
        result["stages"][name] = {
            "n": int(selected.sum()),
            "baseline_q50_q68_q95_deg": q(old_subset),
            "candidate_q50_q68_q95_deg": q(new_subset),
            "candidate_minus_baseline_q68_ci95_deg": paired_interval(new_subset, old_subset),
            "candidate_reruns": int((new.selected_threshold.to_numpy()[selected] < 0.6).sum()),
        }
        valid_stage = selected & line_valid
        line_old = line_errors["baseline"][valid_stage]
        line_new = line_errors["candidate"][valid_stage]
        result["stages"][name]["truth_line"] = {
            "n": int(valid_stage.sum()),
            "baseline_q50_q68_q95_m": q(line_old),
            "candidate_q50_q68_q95_m": q(line_new),
            "candidate_minus_baseline_q68_ci95_m": paired_interval(line_new, line_old),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
