#!/usr/bin/env python3
"""Compare two frozen direction checkpoints on the same ID-matched MC events."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def angle_error(frame: pd.DataFrame, truth: np.ndarray) -> np.ndarray:
    theta, phi = np.deg2rad(truth[:, 0]), np.deg2rad(truth[:, 1])
    target = np.column_stack(
        (np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta))
    )
    direction = frame[["direction_x", "direction_y", "direction_z"]].to_numpy(float)
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    return np.rad2deg(np.arccos(np.clip(np.sum(direction * target, axis=1), -1, 1)))


def quantiles(error: np.ndarray) -> dict[str, float]:
    return {f"q{int(100 * q)}": float(np.quantile(error, q)) for q in (0.5, 0.68, 0.95)}


def bootstrap(candidate: np.ndarray, baseline: np.ndarray, parts: np.ndarray) -> dict:
    rng = np.random.default_rng(20261014)
    groups = [np.flatnonzero(parts == part) for part in np.unique(parts)]
    result = {}
    for label, replicate in (
        ("event", lambda: rng.integers(0, len(candidate), len(candidate))),
        (
            "source_part",
            lambda: np.concatenate(
                [groups[index] for index in rng.integers(0, len(groups), len(groups))]
            ),
        ),
    ):
        deltas = np.empty((500, 2), dtype=float)
        for index in range(len(deltas)):
            sample = replicate()
            for column, q in enumerate((0.5, 0.68)):
                deltas[index, column] = np.quantile(candidate[sample], q) - np.quantile(
                    baseline[sample], q
                )
        result[label] = {
            name: np.quantile(deltas[:, column], [0.025, 0.975]).tolist()
            for column, name in enumerate(("q50_delta_ci95_deg", "q68_delta_ci95_deg"))
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with h5py.File(args.data) as h5:
        ids = np.asarray(h5["test/ev_ids/data"]).astype(str)
        truth = np.asarray(h5["test/prime_prty/data"])
    if len(np.unique(ids)) != len(ids):
        raise ValueError("Evaluation HDF5 has duplicate event IDs")
    errors = {}
    files = {"baseline": args.baseline, "candidate": args.candidate}
    if args.reference is not None:
        files["reference"] = args.reference
    for name, path in files.items():
        frame = pd.read_csv(path).set_index("event_id", verify_integrity=True)
        if not np.array_equal(frame.index.to_numpy(), ids):
            raise ValueError(f"Prediction IDs/order differ from HDF5: {path}")
        errors[name] = angle_error(frame, truth)
    baseline, candidate = errors["baseline"], errors["candidate"]
    parts = np.asarray([event_id.rsplit("_", 1)[0] for event_id in ids])
    report = {
        "inputs": {name: str(path) for name, path in files.items()} | {"data": str(args.data)},
        "n": len(ids),
        "source_parts": int(len(np.unique(parts))),
        "direction_error_deg": {name: quantiles(value) for name, value in errors.items()},
        "candidate_minus_baseline_deg": {
            name: quantiles(candidate)[name] - quantiles(baseline)[name]
            for name in ("q50", "q68", "q95")
        },
        "candidate_better_event_fraction": float(np.mean(candidate < baseline)),
        "paired_bootstrap": bootstrap(candidate, baseline, parts),
        "note": "B is independent by generated primaries from 2020 training, but has been used for earlier model decisions",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
