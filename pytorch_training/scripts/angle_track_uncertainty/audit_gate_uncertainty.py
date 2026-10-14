#!/usr/bin/env python3
"""Audit A-fitted conditional photon-gate radii on disjoint B and 2020 MC.

This script reports candidate calibration factors; it does not alter a model
bundle. Only 2019-A events whose input changes under the p>=0.50 gate are
used to fit the candidate factors.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from audit_2019_point_uncertainty import load_line_error
from audit_primary_overlap import primary_keys
from audit_timegate_mc_quality import direction_error, novel_primary_mask
from calibrate_cross_season_angle import read_angles
from calibrate_cross_season_track import check_truth_primaries


def sample_2019(predictions: Path, selected: Path, truth_points: Path) -> dict:
    frame = pd.read_csv(predictions)
    ids, prime, angle_error, angle_sigma = read_angles(predictions, selected)
    check_truth_primaries(ids, prime, truth_points)
    line_frame, line_error, line_sigma, excluded = load_line_error(predictions, truth_points)
    if excluded or not np.array_equal(line_frame.event_id.to_numpy(), ids):
        raise ValueError("2019 truth line missing or out of order")
    return {
        "ids": ids,
        "prime": prime,
        "gated": frame.selected_threshold.to_numpy() < 0.6,
        "hits": frame.n_hits.to_numpy(),
        "angle_error": angle_error,
        "angle_sigma": angle_sigma,
        "line_error": line_error,
        "line_sigma": line_sigma,
        "line_valid": np.ones(len(ids), dtype=bool),
    }


def sample_2020(
    predictions: Path,
    selected_p050: Path,
    selected_p070: Path,
    masks: Path,
    anchor_targets: Path,
    train_predicted: Path,
    train_gt: Path,
) -> dict:
    frame = pd.read_csv(predictions)
    with h5py.File(selected_p070) as h5:
        original_ids = h5["test/ev_ids/data"][:].astype("U")
    if not np.array_equal(frame.event_id.to_numpy(), original_ids):
        raise ValueError("2020 p>=0.70 event alignment failed")
    with np.load(masks) as saved:
        keep = saved["pred_test"].astype(bool)
    if len(keep) != len(frame):
        raise ValueError("2020 test mask length differs")
    frame = frame.loc[keep].drop_duplicates("event_id", keep="first")
    original_index = frame.index.to_numpy(np.int64)
    source_index = frame.source_event_index.to_numpy(np.int64)
    with h5py.File(selected_p050) as h5:
        ids = h5["test/ev_ids/data"][:][source_index].astype("U")
        prime = h5["test/prime_prty/data"][:][source_index]
    if not np.array_equal(ids, frame.event_id.to_numpy()):
        raise ValueError("2020 p>=0.50 truth alignment failed")
    with h5py.File(anchor_targets) as h5:
        target = h5["test"]
        anchor_ids = target["event_ids"][:][original_index].astype("U")
        line_valid = target["valid"][:][original_index].astype(bool)
        truth_points = target["anchor_point_m"][:][original_index]
    if not np.array_equal(ids, anchor_ids):
        raise ValueError("2020 truth-line target alignment failed")
    theta, phi = np.deg2rad(prime[:, 0]), np.deg2rad(prime[:, 1])
    truth = np.column_stack(
        (np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta))
    )
    delta = frame[["anchor_x_m", "anchor_y_m", "anchor_z_m"]].to_numpy() - truth_points
    transverse = delta - np.sum(delta * truth, axis=1, keepdims=True) * truth
    with h5py.File(train_predicted) as h5:
        train_pred = h5["train/prime_prty/data"][:]
    with h5py.File(train_gt) as h5:
        train_gt_prime = h5["train/prime_prty/data"][:]
    novel, train_primary_count = novel_primary_mask(prime, train_pred, train_gt_prime)
    return {
        "ids": ids,
        "prime": prime,
        "gated": frame.selected_threshold.to_numpy() < 0.6,
        "hits": frame.n_hits.to_numpy(),
        "angle_error": direction_error(frame, prime),
        "angle_sigma": frame.sigma_angle_deg.to_numpy(),
        "line_error": np.linalg.norm(transverse, axis=1),
        "line_sigma": frame.sigma_anchor_m.to_numpy(),
        "line_valid": line_valid,
        "novel_primary": novel,
        "train_primary_count": train_primary_count,
    }


def summarize(sample: dict, selected: np.ndarray, factors: dict) -> dict:
    result = {"events": int(selected.sum())}
    for metric, sigma_name, valid_name in (
        ("angle", "angle_sigma", None),
        ("line", "line_sigma", "line_valid"),
    ):
        use = selected & (sample[valid_name] if valid_name else True)
        error = sample[f"{metric}_error"][use]
        sigma = sample[sigma_name][use]
        if not len(error):
            result[metric] = None
            continue
        if not np.all(np.isfinite(error) & np.isfinite(sigma) & (sigma > 0)):
            raise ValueError(f"Non-finite {metric} errors or scales")
        ratio = error / sigma
        result[metric] = {
            "valid_events": int(len(error)),
            "q50": float(np.quantile(error, 0.5)),
            "q68": float(np.quantile(error, 0.68)),
            "required_factor68": float(np.quantile(ratio, 0.68)),
            "required_factor95": float(np.quantile(ratio, 0.95)),
            "coverage68": float(np.mean(ratio <= factors[metric]["68"])),
            "coverage95": float(np.mean(ratio <= factors[metric]["95"])),
        }
    return result


def decile_report(sample: dict, selected: np.ndarray, factors: dict) -> dict:
    """Check whether marginal factors hide risk-dependent undercoverage."""
    report = {}
    for metric, sigma_name, valid_name in (
        ("angle", "angle_sigma", None),
        ("line", "line_sigma", "line_valid"),
    ):
        use = selected & (sample[valid_name] if valid_name else True)
        positions = np.flatnonzero(use)
        order = positions[np.argsort(sample[sigma_name][positions])]
        rows = []
        for part in np.array_split(order, 10):
            if not len(part):
                continue
            error = sample[f"{metric}_error"][part]
            sigma = sample[sigma_name][part]
            rows.append(
                {
                    "n": int(len(part)),
                    "median_sigma": float(np.median(sigma)),
                    "error_q50": float(np.median(error)),
                    "coverage68": float(np.mean(error <= sigma * factors[metric]["68"])),
                    "coverage95": float(np.mean(error <= sigma * factors[metric]["95"])),
                }
            )
        report[metric] = rows
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--a-predictions", type=Path, required=True)
    parser.add_argument("--a-selected", type=Path, required=True)
    parser.add_argument("--a-truth-points", type=Path, required=True)
    parser.add_argument("--b-predictions", type=Path, required=True)
    parser.add_argument("--b-selected", type=Path, required=True)
    parser.add_argument("--b-truth-points", type=Path, required=True)
    parser.add_argument("--mc-predictions", type=Path, required=True)
    parser.add_argument("--mc-selected-p050", type=Path, required=True)
    parser.add_argument("--mc-selected-p070", type=Path, required=True)
    parser.add_argument("--mc-masks", type=Path, required=True)
    parser.add_argument("--mc-anchor-targets", type=Path, required=True)
    parser.add_argument("--train-predicted", type=Path, required=True)
    parser.add_argument("--train-gt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    a = sample_2019(args.a_predictions, args.a_selected, args.a_truth_points)
    b = sample_2019(args.b_predictions, args.b_selected, args.b_truth_points)
    if np.intersect1d(a["ids"], b["ids"]).size:
        raise ValueError("2019-A and B event IDs overlap")
    if np.intersect1d(primary_keys(a["prime"]), primary_keys(b["prime"])).size:
        raise ValueError("2019-A and B generated primaries overlap")
    bundle = torch.load(args.bundle, map_location="cpu", weights_only=True)
    if bundle.get("model_type") != "photon_corrected_uncertainty":
        raise ValueError("Expected photon-corrected uncertainty bundle")
    old = {key: bundle["calibration"][key] for key in ("angle", "line")}
    fitted = {}
    for metric in ("angle", "line"):
        error = a[f"{metric}_error"][a["gated"]]
        sigma = a[f"{metric}_sigma"][a["gated"]]
        fitted[metric] = {
            "68": float(np.quantile(error / sigma, 0.68)),
            "95": float(np.quantile(error / sigma, 0.95)),
        }
    conservative = {
        metric: {level: max(old[metric][level], fitted[metric][level]) for level in ("68", "95")}
        for metric in ("angle", "line")
    }
    mc = sample_2020(
        args.mc_predictions,
        args.mc_selected_p050,
        args.mc_selected_p070,
        args.mc_masks,
        args.mc_anchor_targets,
        args.train_predicted,
        args.train_gt,
    )
    particle = np.char.partition(mc["ids"].astype(str), "_")[:, 0]
    stages = {
        "2019_a_gated_fit": (a, a["gated"]),
        "2019_b_gated_holdout": (b, b["gated"]),
        "2019_b_unchanged": (b, ~b["gated"]),
    }
    for name in ("nue2", "nuatm", "muatm"):
        stages[f"2020_{name}_gated"] = (mc, mc["gated"] & (particle == name))
        stages[f"2020_{name}_gated_primary_disjoint"] = (
            mc,
            mc["gated"] & (particle == name) & mc["novel_primary"],
        )
    result = {
        "calibration_source": "2019-A gated events only; B and 2020 test are validation",
        "selection": "frozen classifier high p>=0.70 2s8h, conditional low p>=0.50 only if theta<90 and all newly added photon residuals <=50ns",
        "old_p070_factors": old,
        "a_fitted_gated_factors": fitted,
        "conservative_gated_factors": conservative,
        "2019_a_b_event_and_primary_overlap": 0,
        "2020_unique_train_primaries": mc["train_primary_count"],
        "stages": {},
    }
    for name, (sample, mask) in stages.items():
        result["stages"][name] = {
            "old_factors": summarize(sample, mask, old),
            "a_fitted_factors": summarize(sample, mask, fitted),
            "conservative_factors": summarize(sample, mask, conservative),
        }
    result["conditional_deciles_with_old_factors"] = {
        name: decile_report(*stages[name], old)
        for name in (
            "2019_b_gated_holdout",
            "2019_b_unchanged",
            "2020_nue2_gated",
            "2020_nuatm_gated",
        )
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
