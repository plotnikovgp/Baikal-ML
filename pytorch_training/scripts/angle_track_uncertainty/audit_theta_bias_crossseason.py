#!/usr/bin/env python3
"""Predeclared A->B check of a small zenith-bias correction for frozen v3.

This is a diagnostic, not a bundle edit. Fit on 2019-A only, report both
seasons and never choose the correction using 2019-B metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def load(csv_path: Path, h5_path: Path) -> dict[str, np.ndarray]:
    frame = pd.read_csv(csv_path)
    index = frame.source_event_index.to_numpy(dtype=np.int64)
    with h5py.File(h5_path) as h5:
        ids = h5["test/ev_ids/data"][:]
        truth = h5["test/prime_prty/data"][:]
    if np.any(index < 0) or np.any(index >= len(ids)):
        raise ValueError("Prediction source index out of bounds")
    if len(np.unique(index)) != len(index):
        raise ValueError("Duplicate source indices")
    if not np.array_equal(frame.event_id.to_numpy(dtype="S64"), ids[index]):
        raise ValueError("Prediction and HDF5 IDs mismatch")
    return {
        "truth": truth[index, :2].astype(np.float64),
        "primary": truth[index, :3].copy(),
        "theta": frame.theta_deg.to_numpy(dtype=np.float64),
        "phi": frame.phi_deg.to_numpy(dtype=np.float64),
        "hits": frame.n_hits.to_numpy(dtype=np.int64),
        "changed": (frame.selected_threshold.to_numpy() < 0.7),
    }


def vectors(theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    t, p = np.deg2rad(theta), np.deg2rad(phi)
    return np.column_stack([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)])


def quantiles(values: np.ndarray) -> tuple[float, float]:
    return tuple(np.quantile(values, [0.5, 0.68]))


def report(rows: dict[str, np.ndarray], coef: np.ndarray) -> dict:
    truth = rows["truth"]
    theta = rows["theta"]
    x = np.column_stack(
        [np.ones(len(theta)), np.cos(np.deg2rad(theta)), np.cos(np.deg2rad(theta)) ** 2]
    )
    delta = np.clip(x @ coef, -2.0, 2.0)
    before = np.rad2deg(
        np.arccos(
            np.clip(
                np.sum(vectors(theta, rows["phi"]) * vectors(truth[:, 0], truth[:, 1]), axis=1),
                -1,
                1,
            )
        )
    )
    after = np.rad2deg(
        np.arccos(
            np.clip(
                np.sum(
                    vectors(theta + delta, rows["phi"]) * vectors(truth[:, 0], truth[:, 1]), axis=1
                ),
                -1,
                1,
            )
        )
    )
    result = {}
    for scope, mask in [
        ("all", np.ones(len(theta), dtype=bool)),
        ("hits>=20", rows["hits"] >= 20),
        ("hits>=35", rows["hits"] >= 35),
        ("unchanged_p070", ~rows["changed"]),
        ("changed_gate", rows["changed"]),
    ]:
        if not mask.any():
            continue
        result[scope] = {
            "n": int(mask.sum()),
            "before_q50_q68": quantiles(before[mask]),
            "after_q50_q68": quantiles(after[mask]),
            "median_theta_residual_deg": float(np.median(truth[mask, 0] - theta[mask])),
        }
    result["theta_bins"] = []
    for lo, hi in zip(range(0, 180, 20), range(20, 200, 20)):
        mask = (theta >= lo) & (theta < hi)
        if mask.sum() < 50:
            continue
        residual = truth[mask, 0] - theta[mask]
        result["theta_bins"].append(
            {
                "min_pred_theta_deg": lo,
                "max_pred_theta_deg": hi,
                "n": int(mask.sum()),
                "mean_residual_deg": float(np.mean(residual)),
                "median_residual_deg": float(np.median(residual)),
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("a_csv", "a_h5", "b_csv", "b_h5"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    a = load(args.a_csv, args.a_h5)
    b = load(args.b_csv, args.b_h5)

    def keys(rows: dict[str, np.ndarray]) -> np.ndarray:
        return np.ascontiguousarray(rows["primary"]).view("V12").ravel()

    if np.intersect1d(keys(a), keys(b)).size:
        raise ValueError("2019-A and 2019-B share generated primaries")
    # Fixed low-dimensional model; regularize slopes and cap the deployed shift.
    t = a["theta"]
    c = np.cos(np.deg2rad(t))
    x = np.column_stack([np.ones(len(c)), c, c**2])
    y = a["truth"][:, 0] - t
    ridge = np.diag([0.0, 1000.0, 1000.0])
    coef = np.linalg.solve(x.T @ x + ridge, x.T @ y)
    result = {
        "method": "A-only ridge quadratic in cos(predicted theta), clipped to +/-2 degrees",
        "coef": coef.tolist(),
        "inputs": {name: str(getattr(args, name)) for name in ("a_csv", "a_h5", "b_csv", "b_h5")},
        "2019-A": report(a, coef),
        "2019-B": report(b, coef),
    }
    output = json.dumps(result, indent=2)
    print(output)
    if args.output is not None:
        if args.output.exists():
            raise FileExistsError(f"Refusing to overwrite {args.output}")
        args.output.write_text(output + "\n")


if __name__ == "__main__":
    main()
