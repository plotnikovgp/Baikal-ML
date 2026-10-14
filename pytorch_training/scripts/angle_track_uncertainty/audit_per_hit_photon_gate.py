#!/usr/bin/env python3
"""Pair an eventwise all-or-nothing photon gate with an individual-hit gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from evaluate_release_holdout import line_errors, quantiles, source_part_intervals


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--truth-points", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    baseline = pd.read_csv(args.baseline)
    candidate = pd.read_csv(args.candidate)
    if (
        not baseline.event_id.is_unique
        or not candidate.event_id.is_unique
        or not baseline.event_id.equals(candidate.event_id)
        or not np.array_equal(baseline.source_event_index, candidate.source_event_index)
    ):
        raise ValueError("Prediction event IDs/order do not match exactly")
    if (candidate.n_hits.to_numpy() < baseline.n_hits.to_numpy()).any():
        raise ValueError("The individual-hit gate unexpectedly removed baseline hits")

    ids = baseline.event_id.to_numpy(str)
    source_indices = baseline.source_event_index.to_numpy(np.int64)
    with h5py.File(args.data) as h5:
        source_ids = np.asarray(h5["test/ev_ids/data"][source_indices]).astype(str)
        if not np.array_equal(ids, source_ids):
            raise ValueError("Prediction IDs do not match selected input")
        h5_prime = np.asarray(h5["test/prime_prty/data"][source_indices])
    with np.load(args.truth_points) as truth:
        truth_ids = truth["event_ids"].astype(str)
        if len(np.unique(truth_ids)) != len(truth_ids):
            raise ValueError("Duplicate truth-point event IDs")
        position = pd.Index(truth_ids).get_indexer(ids)
        if (position < 0).any() or not truth["valid"][position].all():
            raise ValueError("Missing valid truth line")
        truth_point = np.asarray(truth["point_m"][position], dtype=float)
        prime = np.asarray(truth["prime"][position], dtype=float)
    if not np.allclose(prime[:, :2], h5_prime[:, :2], atol=1e-4, rtol=0):
        raise ValueError("Truth line and input source primaries differ")

    theta, phi = np.deg2rad(prime[:, 0]), np.deg2rad(prime[:, 1])
    true_direction = np.column_stack(
        (np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta))
    )
    old_angle, old_line = line_errors(baseline, true_direction, truth_point)
    new_angle, new_line = line_errors(candidate, true_direction, truth_point)
    old_rerun = baseline.selected_threshold.to_numpy(float) < 0.6
    new_rerun = candidate.selected_threshold.to_numpy(float) < 0.6
    if (old_rerun & ~new_rerun).any():
        raise ValueError("The individual-hit gate removed old gate decisions")
    recovered = new_rerun & ~old_rerun
    if np.count_nonzero(recovered) != np.count_nonzero(
        candidate.n_hits.to_numpy() != baseline.n_hits.to_numpy()
    ):
        raise ValueError("Hit-count and rerun change counts differ")
    for column in ("angle_r68_deg", "angle_r95_deg", "anchor_r68_m", "anchor_r95_m"):
        if candidate.loc[new_rerun, column].notna().any():
            raise ValueError("Changed-input rows must not carry calibrated radii")

    def describe(use: np.ndarray) -> dict:
        return {
            "n": int(use.sum()),
            "baseline_angle_deg": quantiles(old_angle[use]),
            "candidate_angle_deg": quantiles(new_angle[use]),
            "baseline_line_m": quantiles(old_line[use]),
            "candidate_line_m": quantiles(new_line[use]),
            "eventwise_candidate_better_angle": float(np.mean(new_angle[use] < old_angle[use])),
        }

    parts = np.asarray([event_id.rsplit("_", 1)[0] for event_id in ids])
    result = {
        "inputs": {key: str(value) for key, value in vars(args).items() if key != "output"},
        "source_parts": int(len(np.unique(parts))),
        "baseline_reruns": int(old_rerun.sum()),
        "candidate_reruns": int(new_rerun.sum()),
        "newly_recovered": int(recovered.sum()),
        "all": describe(np.ones(len(ids), dtype=bool)),
        "newly_recovered_rows": describe(recovered),
        "source_part_bootstrap_candidate_minus_baseline_ci95": source_part_intervals(
            {"angle": old_angle, "line": old_line},
            {"angle": new_angle, "line": new_line},
            parts,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
