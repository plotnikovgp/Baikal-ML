#!/usr/bin/env python3
"""Conservatively recalibrate frozen point-to-track radii across MC seasons.

Only independent 2019-A truth lines influence the factors, together with
the pre-existing 2020-validation calibration. 2019-B and the 2020 test are
held-out diagnostics. Direction, point, and uncertainty-network weights are
never changed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from audit_2019_point_uncertainty import load_line_error
from calibrate_cross_season_angle import read_angles


def coverage(error: np.ndarray, sigma: np.ndarray, factors: dict) -> dict:
    return {
        "events": int(len(error)),
        "line_q50_m": float(np.quantile(error, 0.5)),
        "line_q68_m": float(np.quantile(error, 0.68)),
        "line_q95_m": float(np.quantile(error, 0.95)),
        "coverage68": float(np.mean(error <= sigma * factors["68"])),
        "coverage95": float(np.mean(error <= sigma * factors["95"])),
    }


def check_truth_primaries(ids: np.ndarray, prime: np.ndarray, truth_path: Path) -> None:
    """Ensure the point source and selected-hit HDF5 describe the same primaries."""
    with np.load(truth_path) as truth:
        source_ids = truth["event_ids"].astype(str)
        source_prime = np.asarray(truth["prime"][:, :3], dtype=np.float32)
    order = np.argsort(source_ids)
    position = np.searchsorted(source_ids[order], ids)
    if (position >= len(order)).any():
        raise ValueError(f"Missing truth event IDs in {truth_path}")
    source_rows = order[position]
    if not np.array_equal(source_ids[source_rows], ids) or not np.array_equal(
        source_prime[source_rows], prime
    ):
        raise ValueError(f"Selected-hit and truth-point primaries disagree: {truth_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-bundle", type=Path, required=True)
    parser.add_argument("--calibration-predictions", type=Path, required=True)
    parser.add_argument("--calibration-data", type=Path, required=True)
    parser.add_argument("--calibration-truth-points", type=Path, required=True)
    parser.add_argument("--holdout-predictions", type=Path, required=True)
    parser.add_argument("--holdout-data", type=Path, required=True)
    parser.add_argument("--holdout-truth-points", type=Path, required=True)
    parser.add_argument(
        "--2020-test-predictions", dest="test_2020_predictions", type=Path, required=True
    )
    parser.add_argument("--output-bundle", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()
    if args.output_bundle.exists() or args.output_report.exists():
        raise FileExistsError("Refusing to overwrite a bundle or report")

    ids_a, prime_a, _, _ = read_angles(args.calibration_predictions, args.calibration_data)
    ids_b, prime_b, _, _ = read_angles(args.holdout_predictions, args.holdout_data)
    if np.intersect1d(ids_a, ids_b).size:
        raise ValueError("Calibration and holdout share event IDs")
    if {row.tobytes() for row in prime_a} & {row.tobytes() for row in prime_b}:
        raise ValueError("Calibration and holdout share generated primaries")
    check_truth_primaries(ids_a, prime_a, args.calibration_truth_points)
    check_truth_primaries(ids_b, prime_b, args.holdout_truth_points)

    pred_a, error_a, sigma_a, excluded_a = load_line_error(
        args.calibration_predictions, args.calibration_truth_points
    )
    pred_b, error_b, sigma_b, excluded_b = load_line_error(
        args.holdout_predictions, args.holdout_truth_points
    )
    if (
        excluded_a
        or excluded_b
        or not np.array_equal(pred_a.event_id, ids_a)
        or not np.array_equal(pred_b.event_id, ids_b)
    ):
        raise ValueError("2019 selected predictions lack unique valid truth lines")

    bundle = torch.load(args.base_bundle, map_location="cpu", weights_only=True)
    if (
        bundle.get("format_version") != 2
        or bundle.get("model_type") != "photon_corrected_uncertainty"
    ):
        raise ValueError("Expected the frozen photon-corrected uncertainty bundle")
    old = bundle["calibration"]["line"]
    candidate = {
        "68": float(np.quantile(error_a / sigma_a, 0.68)),
        "95": float(np.quantile(error_a / sigma_a, 0.95)),
    }
    calibrated = {key: max(float(old[key]), candidate[key]) for key in ("68", "95")}
    bundle["calibration"]["line"] = calibrated
    bundle["cross_season_calibration"]["point_calibration_scope"] = (
        "maximum of 2020-validation and independent 2019-A line factors"
    )
    bundle["cross_season_calibration"]["2019_point_calibration_truth"] = str(
        args.calibration_truth_points
    )
    bundle["cross_season_calibration"]["2019_point_holdout_truth"] = str(args.holdout_truth_points)

    with np.load(args.test_2020_predictions) as npz:
        valid = np.asarray(npz["valid"], dtype=bool)
        error_2020 = np.asarray(npz["line"], dtype=np.float64)[valid]
        sigma_2020 = np.asarray(npz["sigma_line"], dtype=np.float64)[valid]
    report = {
        "old_line_factors": old,
        "new_line_factors": calibrated,
        "angle_factors_unchanged": bundle["calibration"]["angle"],
        "2019_a_calibration": coverage(error_a, sigma_a, calibrated),
        "2019_b_holdout_old": coverage(error_b, sigma_b, old),
        "2019_b_holdout_new": coverage(error_b, sigma_b, calibrated),
        "2020_test_old": coverage(error_2020, sigma_2020, old),
        "2020_test_new": coverage(error_2020, sigma_2020, calibrated),
        "2019_calibration_and_holdout_primary_overlap": 0,
        "line_definition": "transverse distance from predicted point to true infinite track line",
    }
    args.output_bundle.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, args.output_bundle)
    args.output_report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
