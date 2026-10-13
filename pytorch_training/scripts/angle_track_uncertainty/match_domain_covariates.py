#!/usr/bin/env python3
"""Check how much theta-domain shift is explained by observed input features.

The comparison reweights muatm MC to the experimental feature composition.
It does not assume experimental truth and never uses predicted theta for
weight construction. Empty MC cells are explicitly reported as unsupported.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_EDGES = {
    "n_hits": np.asarray([8, 10, 12, 16, 24, 40, np.inf]),
    "n_strings": np.asarray([2, 3, 4, 6, 9, np.inf]),
}
EXTRA_SETS = (
    (),
    ("z_std_m",),
    ("time_std_ns",),
    ("p_mean",),
    ("time_mean_ns",),
    ("z_std_m", "time_std_ns"),
    ("z_std_m", "time_std_ns", "p_mean"),
)


def join_features(features: Path, predictions: Path, *, deduplicate: bool) -> pd.DataFrame:
    feat = pd.read_csv(features).set_index("event_id", verify_integrity=True)
    pred = pd.read_csv(predictions)
    if deduplicate:
        pred = pred.drop_duplicates("event_id", keep="first")
    pred = pred.set_index("event_id", verify_integrity=True)
    if len(feat) != len(pred) or not feat.index.isin(pred.index).all():
        raise ValueError("Input summary and prediction event IDs differ")
    result = feat.join(pred[["theta_deg"]])
    if result.isna().any().any():
        raise ValueError("Missing values after feature/prediction join")
    return result


def bins(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.clip(np.searchsorted(edges, values, side="right") - 1, 0, len(edges) - 2)


def measure(
    exp: pd.DataFrame,
    mc: pd.DataFrame,
    extra: tuple[str, ...],
    *,
    return_weights: bool = False,
) -> dict | tuple[dict, np.ndarray]:
    columns = ("n_hits", "n_strings", *extra)
    edges = dict(BASE_EDGES)
    for name in extra:
        cuts = np.quantile(exp[name], [1 / 3, 2 / 3])
        edges[name] = np.asarray([-np.inf, *cuts, np.inf])
    shape = tuple(len(edges[name]) - 1 for name in columns)
    exp_cell = np.ravel_multi_index(
        tuple(bins(exp[name].to_numpy(), edges[name]) for name in columns), shape
    )
    mc_cell = np.ravel_multi_index(
        tuple(bins(mc[name].to_numpy(), edges[name]) for name in columns), shape
    )
    n_cells = int(np.prod(shape))
    exp_count = np.bincount(exp_cell, minlength=n_cells)
    mc_count = np.bincount(mc_cell, minlength=n_cells)
    supported = mc_count > 0
    exp_supported = supported[exp_cell]
    factor = np.zeros(n_cells, dtype=np.float64)
    factor[supported] = exp_count[supported] / mc_count[supported]
    weight = factor[mc_cell]
    if weight.sum() == 0:
        raise ValueError("No common input support")
    weight /= weight.sum()
    theta_exp = exp.theta_deg.to_numpy()
    theta_mc = mc.theta_deg.to_numpy()
    result = {
        "columns": list(columns),
        "cells": n_cells,
        "exp_unsupported_events": int((~exp_supported).sum()),
        "exp_unsupported_fraction": float((~exp_supported).mean()),
        "mc_effective_sample_size": float(1.0 / np.square(weight).sum()),
        "theta_w1_full_exp_deg": float(wasserstein_distance(theta_exp, theta_mc, v_weights=weight)),
        "theta_w1_supported_exp_deg": float(
            wasserstein_distance(theta_exp[exp_supported], theta_mc, v_weights=weight)
        ),
        "bin_edges": {name: [float(value) for value in edges[name][1:-1]] for name in extra},
    }
    return (result, weight) if return_weights else result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-features", type=Path, required=True)
    parser.add_argument("--mc-features", type=Path, required=True)
    parser.add_argument("--exp-predictions", type=Path, required=True)
    parser.add_argument("--mc-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=200)
    args = parser.parse_args()
    if args.bootstrap < 1:
        raise ValueError("Bootstrap count must be positive")
    exp = join_features(args.exp_features, args.exp_predictions, deduplicate=False)
    mc = join_features(args.mc_features, args.mc_predictions, deduplicate=True)
    report = {"exp_events": len(exp), "muatm_mc_events": len(mc), "matchings": {}}
    for extra in EXTRA_SETS:
        label = "hit_string" + ("_" + "_".join(extra) if extra else "")
        report["matchings"][label] = measure(exp, mc, extra)
    rng = np.random.default_rng(731)
    changes = np.empty(args.bootstrap)
    for i in range(args.bootstrap):
        exp_sample = exp.iloc[rng.integers(len(exp), size=len(exp))]
        mc_sample = mc.iloc[rng.integers(len(mc), size=len(mc))]
        base = measure(exp_sample, mc_sample, ())
        z_match = measure(exp_sample, mc_sample, ("z_std_m",))
        changes[i] = z_match["theta_w1_full_exp_deg"] - base["theta_w1_full_exp_deg"]
    report["z_match_minus_hit_string_w1_ci95_deg"] = np.quantile(changes, [0.025, 0.975]).tolist()
    z_cuts = np.quantile(exp.z_std_m, [1 / 3, 2 / 3])
    report["theta_w1_within_z_tertiles"] = []
    exp_z_bin = np.searchsorted(z_cuts, exp.z_std_m, side="right")
    mc_z_bin = np.searchsorted(z_cuts, mc.z_std_m, side="right")
    for z_bin in range(3):
        e = exp.iloc[np.flatnonzero(exp_z_bin == z_bin)]
        m = mc.iloc[np.flatnonzero(mc_z_bin == z_bin)]
        conditional = measure(e, m, ())
        report["theta_w1_within_z_tertiles"].append(
            {"z_bin": z_bin, "exp_events": len(e), "mc_events": len(m), **conditional}
        )
    _, base_weights = measure(exp, mc, (), return_weights=True)
    _, z_weights = measure(exp, mc, ("z_std_m",), return_weights=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    edges = np.arange(70, 186, 5)
    ax.hist(
        exp.theta_deg,
        bins=edges,
        density=True,
        histtype="step",
        linewidth=2,
        label="Experiment",
    )
    ax.hist(
        mc.theta_deg,
        bins=edges,
        weights=base_weights,
        density=True,
        histtype="step",
        linewidth=1.8,
        label="muatm MC, hits/strings matched",
    )
    ax.hist(
        mc.theta_deg,
        bins=edges,
        weights=z_weights,
        density=True,
        histtype="step",
        linewidth=1.8,
        label="muatm MC, plus z-spread matched",
    )
    ax.set(
        xlabel="Predicted θ [deg]",
        ylabel="Density / 5° bin",
        title="Input-composition effect on experimental/MC θ comparison",
    )
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output.with_suffix(".png"), dpi=180)
    plt.close(fig)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
