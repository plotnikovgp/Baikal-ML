#!/usr/bin/env python3
"""Test a fast hit-time/geometry direction estimate against one NN checkpoint.

Only measured hit position and time are used by the physical estimate. MC
truth is read for evaluation, never to choose an event's estimate or sign.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ALPHAS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5)


def unit(vector: np.ndarray) -> np.ndarray:
    size = np.linalg.norm(vector, axis=-1, keepdims=True)
    return vector / np.maximum(size, 1e-12)


def angle_deg(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return np.rad2deg(np.arccos(np.clip(np.sum(first * second, axis=-1), -1, 1)))


def fit_event(hits: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    position = hits[:, 2:5].astype(np.float64)
    time = hits[:, 1].astype(np.float64)
    position = position - position.mean(axis=0)
    time = time - time.mean()
    cross = position.T @ time
    covariance = position.T @ position
    regularization = 0.02 * np.trace(covariance) / 3 + 1e-6
    gradient = np.linalg.solve(covariance + np.eye(3) * regularization, cross)
    time_cov_direction = unit(cross)
    gradient_direction = unit(gradient)
    residual = time - position @ gradient
    total = np.sum(time**2)
    r_squared = 1 - np.sum(residual**2) / max(total, 1e-12)
    return time_cov_direction, gradient_direction, float(r_squared), float(np.linalg.norm(gradient))


def describe(values: np.ndarray) -> dict:
    return {
        "count": int(len(values)),
        "q50": float(np.quantile(values, 0.5)),
        "q68": float(np.quantile(values, 0.68)),
        "q90": float(np.quantile(values, 0.9)),
    }


def evaluate_group(
    selected: np.ndarray,
    nn: np.ndarray,
    fitted: dict[str, np.ndarray],
    truth: np.ndarray | None,
) -> dict:
    report = {"count": int(selected.sum())}
    if not selected.any():
        return report
    nn_selected = nn[selected]
    if truth is None:
        report["nn_theta_deg"] = describe(np.rad2deg(np.arccos(np.clip(nn_selected[:, 2], -1, 1))))
    else:
        truth_selected = truth[selected]
        report["nn_angle_error_deg"] = describe(angle_deg(nn_selected, truth_selected))
    for name, estimate in fitted.items():
        fit_selected = estimate[selected]
        method = {"change_from_nn_deg": describe(angle_deg(nn_selected, fit_selected))}
        if truth is not None:
            method["fit_angle_error_deg"] = describe(angle_deg(fit_selected, truth[selected]))
        method["fusions"] = {}
        for alpha in ALPHAS:
            fused = unit((1 - alpha) * nn_selected + alpha * fit_selected)
            if truth is None:
                method["fusions"][str(alpha)] = {
                    "theta_deg": describe(np.rad2deg(np.arccos(np.clip(fused[:, 2], -1, 1)))),
                    "change_from_nn_deg": describe(angle_deg(fused, nn_selected)),
                }
            else:
                method["fusions"][str(alpha)] = describe(angle_deg(fused, truth[selected]))
        report[name] = method
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--anchors", type=Path)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--max-events", type=int)
    args = parser.parse_args()
    predictions = pd.read_csv(args.predictions)
    if args.max_events is not None:
        predictions = predictions.iloc[: args.max_events].copy()
    if "source_event_index" not in predictions:
        raise ValueError("Predictions must contain source_event_index")
    source_index = predictions.source_event_index.to_numpy(dtype=np.int64)
    if (np.diff(source_index) <= 0).any():
        raise ValueError("source_event_index must be strictly increasing")
    nn = unit(predictions[["direction_x", "direction_y", "direction_z"]].to_numpy())
    n = len(predictions)
    time_cov = np.empty((n, 3), dtype=np.float32)
    gradient = np.empty((n, 3), dtype=np.float32)
    r_squared = np.empty(n, dtype=np.float32)
    slope = np.empty(n, dtype=np.float32)
    truth = None
    with h5py.File(args.data, "r") as h5:
        group = h5[args.split]
        ids = np.char.decode(group["ev_ids/data"][source_index].astype("S"), "utf-8")
        if not np.array_equal(ids, predictions.event_id.to_numpy()):
            raise ValueError("Prediction IDs do not match the HDF5 source rows")
        starts = group["ev_starts/data"]
        hits = group["data/data"]
        norm_mean = np.asarray(h5["norm_param/mean"])
        norm_std = np.asarray(h5["norm_param/std"])
        for i, index in enumerate(source_index):
            raw = np.asarray(hits[int(starts[index]) : int(starts[index + 1])])
            raw = raw * norm_std + norm_mean
            time_cov[i], gradient[i], r_squared[i], slope[i] = fit_event(raw)
        if "prime_prty/data" in group:
            prime = group["prime_prty/data"][source_index, :2]
            theta, phi = np.deg2rad(prime[:, 0]), np.deg2rad(prime[:, 1])
            truth = np.stack(
                (np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)),
                axis=1,
            )
    valid_track = None
    if args.anchors:
        with h5py.File(args.anchors, "r") as h5:
            group = h5[args.split]
            anchor_ids = group["event_ids"][source_index]
            if not np.array_equal(anchor_ids, ids.astype("S")):
                raise ValueError("Anchor IDs do not match the HDF5 source rows")
            valid_track = group["valid"][source_index].astype(bool)
    particles = predictions.event_id.str.split("_", n=1).str[0].to_numpy()
    fitted = {"time_cov": time_cov, "time_gradient": gradient}
    report = {
        "predictions": str(args.predictions),
        "data": str(args.data),
        "split": args.split,
        "r_squared": describe(r_squared),
        "slope_ns_per_m": describe(slope),
        "groups": {},
    }
    for particle in np.unique(particles):
        selected = particles == particle
        report["groups"][str(particle)] = evaluate_group(selected, nn, fitted, truth)
        if particle == "muatm" and valid_track is not None:
            report["groups"]["muatm_single"] = evaluate_group(
                selected & valid_track, nn, fitted, truth
            )
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    args.output_prefix.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    arrays = dict(
        event_id=ids.astype(str),
        source_event_index=source_index,
        nn=nn.astype(np.float32),
        time_cov=time_cov,
        time_gradient=gradient,
        r_squared=r_squared,
        slope_ns_per_m=slope,
    )
    if truth is not None:
        arrays["truth"] = truth.astype(np.float32)
    if valid_track is not None:
        arrays["valid_track"] = valid_track
    np.savez_compressed(args.output_prefix.with_suffix(".npz"), **arrays)
    print(
        json.dumps(
            {"output": str(args.output_prefix), "events": n, "groups": list(report["groups"])}
        )
    )


if __name__ == "__main__":
    main()
