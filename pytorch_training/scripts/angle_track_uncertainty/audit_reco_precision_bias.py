#!/usr/bin/env python3
"""Check whether a low-order angular bias explains the high-quality reco gap.

This is a diagnostic on the paired 2020 reco cohort, not an independent
model-selection result. Source parts A-C fit one constant theta/phi offset;
parts D-E only test that predeclared offset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from analyze_reco_central_gap import root_ten_cut


def wrap_deg(delta: np.ndarray) -> np.ndarray:
    return (delta + 180) % 360 - 180


def unit(theta_deg: np.ndarray, phi_deg: np.ndarray) -> np.ndarray:
    theta, phi = np.deg2rad(theta_deg), np.deg2rad(phi_deg)
    return np.column_stack(
        (np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta))
    )


def angular_error(theta: np.ndarray, phi: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return np.rad2deg(np.arccos(np.clip(np.sum(unit(theta, phi) * truth, axis=1), -1, 1)))


def quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        "q50": float(np.quantile(values, 0.5)),
        "q68": float(np.quantile(values, 0.68)),
        "q90": float(np.quantile(values, 0.9)),
    }


def signed_summary(theta: np.ndarray, phi: np.ndarray, truth_angles: np.ndarray) -> dict:
    dtheta = theta - truth_angles[:, 0]
    dphi = wrap_deg(phi - truth_angles[:, 1])
    return {
        name: {
            "mean_deg": float(np.mean(values)),
            "median_deg": float(np.median(values)),
            "std_deg": float(np.std(values)),
            "abs": quantiles(np.abs(values)),
        }
        for name, values in (("theta", dtheta), ("phi", dphi))
    }


def score(mask: np.ndarray, nn: np.ndarray, reco: np.ndarray) -> dict:
    if mask.sum() < 30:
        return {"n": int(mask.sum())}
    return {
        "n": int(mask.sum()),
        "nn": quantiles(nn[mask]),
        "reco": quantiles(reco[mask]),
        "nn_better_fraction": float(np.mean(nn[mask] < reco[mask])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired", type=Path, required=True)
    parser.add_argument("--root-fields", type=Path, required=True)
    parser.add_argument("--fair-source", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    with np.load(args.paired) as pair:
        ids = pair["event_ids"].astype("U")
        index = pair["fair_index"].astype(np.int64)
        reco_error = pair["standard_reco_direction_error_deg"].astype(float)
        gt_span = pair["gt_span_m"].astype(float)
    if len(np.unique(ids)) != len(ids):
        raise ValueError("Duplicate paired IDs")
    root = pd.read_csv(args.root_fields).set_index("event_id", verify_integrity=True).reindex(ids)
    gate = pd.read_csv(args.gate).set_index("event_id", verify_integrity=True).reindex(ids)
    with h5py.File(args.fair_source) as source:
        if not np.array_equal(source["test/ev_ids/data"][:][index].astype("U"), ids):
            raise ValueError("Truth and prediction event IDs do not align")
        truth_angles = np.asarray(source["test/prime_prty/data"][:][index, :2], dtype=float)
    available = np.isfinite(gate["direction_x"].to_numpy(float))
    high = (
        root_ten_cut(root)
        & available
        & (root["fNHits"].to_numpy(float) >= 20)
        & (root["fNTriplets"].to_numpy(float) >= 12)
    )
    if int(high.sum()) != 5679:
        raise ValueError(f"Expected 5,679 paired high-quality events; got {high.sum()}")
    vectors = gate[["direction_x", "direction_y", "direction_z"]].to_numpy(float)
    norms = np.linalg.norm(vectors, axis=1)
    cosine = np.divide(vectors[:, 2], norms, out=np.full(len(ids), np.nan), where=norms > 0)
    theta = np.rad2deg(np.arccos(np.clip(cosine, -1, 1)))
    phi = np.rad2deg(np.arctan2(vectors[:, 1], vectors[:, 0]))
    truth = unit(truth_angles[:, 0], truth_angles[:, 1])
    nn_error = angular_error(theta[high], phi[high], truth[high])
    if not np.isfinite(nn_error).all():
        raise ValueError("Non-finite NN angle")
    nn_all = np.full(len(ids), np.nan)
    nn_all[high] = nn_error
    part = np.asarray([value.split("_")[1] for value in ids])
    fit = high & np.isin(part, ["A", "B", "C"])
    holdout = high & np.isin(part, ["D", "E"])
    correction = {
        "theta_deg": float(np.median(theta[fit] - truth_angles[fit, 0])),
        "phi_deg": float(np.median(wrap_deg(phi[fit] - truth_angles[fit, 1]))),
    }
    corrected = angular_error(
        theta[holdout] - correction["theta_deg"],
        phi[holdout] - correction["phi_deg"],
        truth[holdout],
    )
    before = nn_all[holdout]
    count = int(holdout.sum())
    rng = np.random.default_rng(20261014)
    bootstrap = []
    for _ in range(1000):
        sampled = rng.integers(0, count, count)
        bootstrap.append(np.median(corrected[sampled]) - np.median(before[sampled]))
    report = {
        "inputs": {key: str(value) for key, value in vars(args).items() if key != "output"},
        "note": "2020 reco cohort is descriptive and may overlap training primaries",
        "high_quality": score(high, nn_all, reco_error),
        "signed_high_quality_nn": signed_summary(theta[high], phi[high], truth_angles[high]),
        "signed_high_quality_reco": signed_summary(
            root["fThetaRec"].to_numpy(float)[high],
            root["fPhiRec"].to_numpy(float)[high],
            truth_angles[high],
        ),
        "constant_correction_fit_parts_abc": correction,
        "constant_correction_holdout_de": {
            "n": count,
            "before": quantiles(before),
            "after": quantiles(corrected),
            "standard_reco": quantiles(reco_error[holdout]),
            "q50_change_bootstrap_95": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
        },
        "truth_theta_bins": {},
        "truth_phi_bins": {},
        "high_quality_strata": {},
    }
    for name, values, edges in (
        ("truth_theta_bins", truth_angles[:, 0], [0, 30, 60, 90, 120, 150, 180.001]),
        ("truth_phi_bins", truth_angles[:, 1] % 360, [0, 45, 90, 135, 180, 225, 270, 315, 360.001]),
    ):
        for low, high_edge in zip(edges[:-1], edges[1:]):
            mask = high & (values >= low) & (values < high_edge)
            report[name][f"{low:g}-{high_edge:g}"] = score(mask, nn_all, reco_error)
    selected_hits = gate["n_hits"].to_numpy(float)
    for name, values, edges in (
        ("reco_hits", root["fNHits"].to_numpy(float), [20, 30, 50, np.inf]),
        ("reco_triplets", root["fNTriplets"].to_numpy(float), [12, 20, 40, np.inf]),
        ("selected_hits", selected_hits, [0, 20, 35, 60, np.inf]),
        (
            "reco_minus_selected_hits",
            root["fNHits"].to_numpy(float) - selected_hits,
            [-np.inf, -10, -3, 3, 10, np.inf],
        ),
        ("gt_span_m", gt_span, [0, 150, 300, 450, np.inf]),
        (
            "reco_theta_error_estimate_deg",
            root["fThetaErr"].to_numpy(float),
            [0, 0.3, 0.6, 1.0, np.inf],
        ),
    ):
        report["high_quality_strata"][name] = {}
        for low, high_edge in zip(edges[:-1], edges[1:]):
            mask = high & (values >= low) & (values < high_edge)
            report["high_quality_strata"][name][f"{low:g}-{high_edge:g}"] = score(
                mask, nn_all, reco_error
            )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
