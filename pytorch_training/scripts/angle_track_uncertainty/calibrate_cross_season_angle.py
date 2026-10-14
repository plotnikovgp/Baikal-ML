#!/usr/bin/env python3
"""Conservatively recalibrate frozen angle radii across 2020 and 2019 MC.

The 2019-A sample sets only multiplicative angle-radius factors. 2019-B and
the 2020 test split are diagnostics and never influence the factors. Point
uncertainty remains calibrated only on 2020 MC because no matched 2019 point
truth is supplied here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch


def read_angles(prediction_path: Path, data_path: Path):
    predictions = pd.read_csv(prediction_path)
    with h5py.File(data_path) as h5:
        group = h5["test"]
        ids = group["ev_ids/data"][:].astype("U")
        prime = np.asarray(group["prime_prty/data"][:, :3], dtype=np.float32)
    if not np.array_equal(predictions.event_id.to_numpy(), ids):
        raise ValueError(f"Prediction/HDF5 event alignment failed: {prediction_path}")
    theta, phi = np.deg2rad(prime[:, 0]), np.deg2rad(prime[:, 1])
    truth = np.column_stack(
        (np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta))
    )
    predicted = predictions[["direction_x", "direction_y", "direction_z"]].to_numpy()
    predicted /= np.linalg.norm(predicted, axis=1, keepdims=True)
    error = np.rad2deg(
        np.arctan2(
            np.linalg.norm(np.cross(truth, predicted), axis=1),
            np.einsum("ij,ij->i", truth, predicted),
        )
    )
    sigma = predictions.sigma_angle_deg.to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(error)) or not np.all(np.isfinite(sigma) & (sigma > 0)):
        raise ValueError("Invalid errors or predicted angle scales")
    return ids, prime, error, sigma


def coverage(error: np.ndarray, sigma: np.ndarray, factors: dict) -> dict:
    return {
        "events": len(error),
        "angle_q50_deg": float(np.quantile(error, 0.5)),
        "angle_q68_deg": float(np.quantile(error, 0.68)),
        "angle_q95_deg": float(np.quantile(error, 0.95)),
        "coverage68": float(np.mean(error <= sigma * factors["68"])),
        "coverage95": float(np.mean(error <= sigma * factors["95"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-bundle", type=Path, required=True)
    parser.add_argument("--calibration-predictions", type=Path, required=True)
    parser.add_argument("--calibration-data", type=Path, required=True)
    parser.add_argument("--holdout-predictions", type=Path, required=True)
    parser.add_argument("--holdout-data", type=Path, required=True)
    parser.add_argument(
        "--2020-test-predictions", dest="test_2020_predictions", type=Path, required=True
    )
    parser.add_argument("--output-bundle", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()
    if args.output_bundle.exists() or args.output_report.exists():
        raise FileExistsError("Refusing to overwrite a bundle or report")

    ids_a, prime_a, error_a, sigma_a = read_angles(
        args.calibration_predictions, args.calibration_data
    )
    ids_b, prime_b, error_b, sigma_b = read_angles(args.holdout_predictions, args.holdout_data)
    if np.intersect1d(ids_a, ids_b).size:
        raise ValueError("2019 calibration and holdout have shared event IDs")
    primary_a = {row.tobytes() for row in prime_a}
    primary_b = {row.tobytes() for row in prime_b}
    if primary_a & primary_b:
        raise ValueError("2019 calibration and holdout have shared generated primaries")

    bundle = torch.load(args.base_bundle, map_location="cpu", weights_only=True)
    if (
        bundle.get("format_version") != 2
        or bundle.get("model_type") != "photon_corrected_uncertainty"
    ):
        raise ValueError("Expected the frozen photon-corrected uncertainty bundle")
    old = bundle["calibration"]["angle"]
    candidate = {
        "68": float(np.quantile(error_a / sigma_a, 0.68)),
        "95": float(np.quantile(error_a / sigma_a, 0.95)),
    }
    calibrated = {key: max(float(old[key]), candidate[key]) for key in ("68", "95")}
    bundle["calibration"]["angle"] = calibrated
    bundle["cross_season_calibration"] = {
        "method": "maximum of 2020-validation and independent 2019-A angle factors",
        "2019_calibration_predictions": str(args.calibration_predictions),
        "2019_holdout_predictions": str(args.holdout_predictions),
        "point_calibration_scope": "2020 MC only; not validated across seasons",
    }

    with np.load(args.test_2020_predictions) as npz:
        error_2020 = np.asarray(npz["angle"], dtype=np.float64)
        sigma_2020 = np.asarray(npz["sigma_angle"], dtype=np.float64)
    report = {
        "old_angle_factors": old,
        "new_angle_factors": calibrated,
        "2019_a_calibration": coverage(error_a, sigma_a, calibrated),
        "2019_b_holdout_old": coverage(error_b, sigma_b, old),
        "2019_b_holdout_new": coverage(error_b, sigma_b, calibrated),
        "2020_test_old": coverage(error_2020, sigma_2020, old),
        "2020_test_new": coverage(error_2020, sigma_2020, calibrated),
        "2019_calibration_and_holdout_primary_overlap": 0,
        "point_uncertainty_caveat": "2020 MC calibration only; no 2019 matched anchor truth",
    }
    args.output_bundle.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, args.output_bundle)
    args.output_report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
