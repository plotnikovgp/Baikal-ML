#!/usr/bin/env python3
"""Test a no-fit, time-oriented hit-cloud PCA axis against a frozen NN.

This is a physics-inspired feasibility diagnostic, not a reconstruction
candidate. The oracle-sign metric uses MC truth and is never deployable.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np


def error_degrees(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return np.rad2deg(np.arccos(np.clip(np.sum(pred * truth, axis=1), -1, 1)))


def summarize(mask: np.ndarray, errors: dict[str, np.ndarray], sign_correct: np.ndarray) -> dict:
    return {
        "n": int(mask.sum()),
        "q50_q68_deg": {
            name: np.quantile(value[mask], [0.5, 0.68]).tolist() for name, value in errors.items()
        },
        "time_sign_correct_fraction": float(np.mean(sign_correct[mask])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    with h5py.File(args.data) as h5:
        group = h5["test"]
        ids = group["ev_ids/data"][:].astype("U")
        starts = group["ev_starts/data"][:].astype(np.int64)
        x = group["data/data"][:].astype(np.float64)
        prime = group["prime_prty/data"][:, :2].astype(np.float64)
        mean = h5["norm_param/mean"][:].astype(np.float64)
        std = h5["norm_param/std"][:].astype(np.float64)
    if len(np.unique(ids)) != len(ids) or np.any(np.diff(starts) < 8):
        raise ValueError("Duplicate IDs or event below the 8-hit cut")

    predictions = {}
    with args.predictions.open(newline="") as stream:
        for row in csv.DictReader(stream):
            event_id = row["event_id"]
            if event_id in predictions:
                raise ValueError(f"Duplicate prediction ID: {event_id}")
            predictions[event_id] = np.asarray(
                [float(row[f"direction_{axis}"]) for axis in "xyz"], dtype=np.float64
            )
    if set(predictions) != set(ids):
        raise ValueError("Prediction and HDF5 event sets differ")
    nn = np.stack([predictions[event_id] for event_id in ids])
    nn /= np.linalg.norm(nn, axis=1, keepdims=True)
    theta, phi = np.deg2rad(prime[:, 0]), np.deg2rad(prime[:, 1])
    truth = np.column_stack(
        (np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta))
    )

    xyz = x[:, 2:5] * std[2:5] + mean[2:5]
    times = x[:, 1] * std[1] + mean[1]
    axes = np.zeros((len(ids), 3), dtype=np.float64)
    linearity = np.zeros(len(ids), dtype=np.float64)
    counts = np.diff(starts)
    for index, (low, high) in enumerate(zip(starts[:-1], starts[1:])):
        positions = xyz[low:high]
        centered = positions - positions.mean(axis=0)
        eigenvalues, eigenvectors = np.linalg.eigh(centered.T @ centered)
        axis = eigenvectors[:, -1]
        linearity[index] = eigenvalues[-1] / max(eigenvalues.sum(), 1e-9)
        if np.dot(centered @ axis, times[low:high] - times[low:high].mean()) < 0:
            axis = -axis
        axes[index] = axis
    nn_oriented_axes = np.where((np.sum(axes * nn, axis=1) < 0)[:, None], -axes, axes)
    errors = {
        "frozen_nn": error_degrees(nn, truth),
        "pca_time_sign": error_degrees(axes, truth),
        "pca_nn_sign": error_degrees(nn_oriented_axes, truth),
    }
    errors["pca_oracle_sign"] = np.minimum(errors["pca_time_sign"], 180 - errors["pca_time_sign"])
    sign_correct = np.sum(axes * truth, axis=1) > 0
    cohorts = {
        "all": counts >= 8,
        "hits_ge_20": counts >= 20,
        "hits_ge_35": counts >= 35,
        "linearity_ge_0p9": linearity >= 0.9,
        "hits_ge_20_linearity_ge_0p9": (counts >= 20) & (linearity >= 0.9),
    }
    output = {
        "data": str(args.data),
        "predictions": str(args.predictions),
        "method": "top eigenvector of selected-hit XYZ covariance; sign from positive hit-time correlation",
        "linearity_q50_q90": np.quantile(linearity, [0.5, 0.9]).tolist(),
        "cohorts": {name: summarize(mask, errors, sign_correct) for name, mask in cohorts.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
