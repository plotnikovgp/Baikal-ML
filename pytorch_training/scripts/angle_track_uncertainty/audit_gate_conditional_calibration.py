#!/usr/bin/env python3
"""Fit a five-bin gate uncertainty map on 2019-A and audit untouched holdouts.

This is a diagnostic only: no bundle, prediction, or calibrated output radius
is changed. The 2020 primary-disjoint and muatm rows are safety checks, not
fit data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from audit_gate_uncertainty import sample_2019, sample_2020
from audit_primary_overlap import primary_keys


def fit_factors(sample: dict, metric: str) -> dict:
    use = sample["gated"] & sample["line_valid"] if metric == "line" else sample["gated"]
    sigma = sample[f"{metric}_sigma"][use]
    error = sample[f"{metric}_error"][use]
    if not np.isfinite(sigma).all() or not np.isfinite(error).all() or np.any(sigma <= 0):
        raise ValueError(f"Invalid {metric} fit data")
    edges = np.quantile(sigma, [0.2, 0.4, 0.6, 0.8])
    if np.any(np.diff(edges) <= 0):
        raise ValueError("Sigma quantile edges are not strictly increasing")
    labels = np.searchsorted(edges, sigma, side="right")
    factors = {}
    for level, quantile in (("68", 0.68), ("95", 0.95)):
        factors[level] = [
            float(np.quantile((error / sigma)[labels == bin_index], quantile))
            for bin_index in range(5)
        ]
    return {
        "sigma_edges": edges.tolist(),
        "fit_count_per_bin": np.bincount(labels, minlength=5).tolist(),
        "factors": factors,
    }


def wilson_interval(successes: int, total: int, z: float = 1.96) -> list[float]:
    if total <= 0:
        return [float("nan"), float("nan")]
    p = successes / total
    denom = 1 + z * z / total
    midpoint = (p + z * z / (2 * total)) / denom
    halfwidth = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return [float(midpoint - halfwidth), float(midpoint + halfwidth)]


def coverage(error: np.ndarray, radius: np.ndarray) -> dict:
    hits = int(np.sum(error <= radius))
    return {
        "n": int(len(error)),
        "fraction": hits / len(error),
        "wilson95": wilson_interval(hits, len(error)),
        "median_radius": float(np.median(radius)),
    }


def evaluate(sample: dict, use: np.ndarray, fitted: dict, old: dict) -> dict:
    output = {"events": int(use.sum())}
    for metric in ("angle", "line"):
        valid = use & sample["line_valid"] if metric == "line" else use
        sigma = sample[f"{metric}_sigma"][valid]
        error = sample[f"{metric}_error"][valid]
        if not len(error):
            output[metric] = None
            continue
        if not np.isfinite(sigma).all() or not np.isfinite(error).all() or np.any(sigma <= 0):
            raise ValueError(f"Invalid {metric} evaluation data")
        fit = fitted[metric]
        labels = np.searchsorted(fit["sigma_edges"], sigma, side="right")
        report = {"valid_events": int(len(error)), "levels": {}}
        for level in ("68", "95"):
            old_radius = sigma * old[metric][level]
            mapped = sigma * np.asarray(fit["factors"][level])[labels]
            conservative = np.maximum(mapped, old_radius)
            report["levels"][level] = {
                "old": coverage(error, old_radius),
                "a_fitted_five_bin": coverage(error, mapped),
                "max_old_and_fitted": coverage(error, conservative),
            }
        order = np.argsort(sigma)
        report["sigma_deciles"] = []
        for decile_index, positions in enumerate(np.array_split(order, 10), start=1):
            local = {
                "index": decile_index,
                "n": int(len(positions)),
                "median_sigma": float(np.median(sigma[positions])),
            }
            for level in ("68", "95"):
                old_radius = sigma[positions] * old[metric][level]
                mapped = sigma[positions] * np.asarray(fit["factors"][level])[labels[positions]]
                local[f"old_coverage{level}"] = float(np.mean(error[positions] <= old_radius))
                local[f"fitted_coverage{level}"] = float(np.mean(error[positions] <= mapped))
            report["sigma_deciles"].append(local)
        output[metric] = report
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "bundle",
        "a_predictions",
        "a_selected",
        "a_truth_points",
        "b_predictions",
        "b_selected",
        "b_truth_points",
        "mc_predictions",
        "mc_selected_p050",
        "mc_selected_p070",
        "mc_masks",
        "mc_anchor_targets",
        "train_predicted",
        "train_gt",
        "output",
    ):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    a = sample_2019(args.a_predictions, args.a_selected, args.a_truth_points)
    b = sample_2019(args.b_predictions, args.b_selected, args.b_truth_points)
    if set(a["ids"]) & set(b["ids"]):
        raise ValueError("2019-A/B event IDs overlap")
    if np.intersect1d(primary_keys(a["prime"]), primary_keys(b["prime"])).size:
        raise ValueError("2019-A/B generated primaries overlap")
    bundle = torch.load(args.bundle, map_location="cpu", weights_only=True)
    if bundle.get("model_type") != "photon_corrected_uncertainty":
        raise ValueError("Unexpected bundle model type")
    old = {key: bundle["calibration"][key] for key in ("angle", "line")}
    fitted = {metric: fit_factors(a, metric) for metric in ("angle", "line")}
    mc = sample_2020(
        args.mc_predictions,
        args.mc_selected_p050,
        args.mc_selected_p070,
        args.mc_masks,
        args.mc_anchor_targets,
        args.train_predicted,
        args.train_gt,
    )
    if mc["train_primary_count"] != 3_022_664:
        raise ValueError("Training-primary baseline differs from the established 2020 audit")
    particle = np.char.partition(mc["ids"].astype(str), "_")[:, 0]
    groups = {
        "2019_a_gated_fit": (a, a["gated"]),
        "2019_b_gated_holdout": (b, b["gated"]),
        "2020_nue2_gated": (mc, mc["gated"] & (particle == "nue2")),
        "2020_nue2_gated_primary_disjoint": (
            mc,
            mc["gated"] & (particle == "nue2") & mc["novel_primary"],
        ),
        "2020_nuatm_gated_primary_disjoint": (
            mc,
            mc["gated"] & (particle == "nuatm") & mc["novel_primary"],
        ),
        "2020_muatm_gated": (mc, mc["gated"] & (particle == "muatm")),
    }
    if groups["2020_nue2_gated_primary_disjoint"][1].sum() != 225:
        raise ValueError("Primary-disjoint gated NUE2 cohort differs from prior audit")
    result = {
        "note": "Diagnostic only; fit 2019-A gated five equal-count sigma bins, hold out all B/2020; no bundle changes",
        "sources": {name: str(value) for name, value in vars(args).items() if name != "output"},
        "old_p070_factors": old,
        "a_fitted_five_bin": fitted,
        "groups": {
            name: evaluate(sample, use, fitted, old) for name, (sample, use) in groups.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "fit_n": result["groups"]["2019_a_gated_fit"]["events"],
                "b_n": result["groups"]["2019_b_gated_holdout"]["events"],
                "2020_disjoint_nue2_n": result["groups"]["2020_nue2_gated_primary_disjoint"][
                    "events"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
