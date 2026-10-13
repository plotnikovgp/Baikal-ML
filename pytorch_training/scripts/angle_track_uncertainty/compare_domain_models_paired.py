#!/usr/bin/env python3
"""Compare two checkpoints' theta-domain shift on identical selected events."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from analyze_domain_predictions import matching_weights
from match_domain_covariates import measure
from scipy.stats import wasserstein_distance


def read(path: Path, deduplicate: bool) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if deduplicate:
        frame = frame.drop_duplicates("event_id", keep="first")
    return frame.set_index("event_id", verify_integrity=True)


def aligned(before_path: Path, after_path: Path, deduplicate: bool):
    before = read(before_path, deduplicate)
    after = read(after_path, deduplicate)
    if len(before) != len(after) or not before.index.isin(after.index).all():
        raise ValueError(f"Before/after event IDs differ: {before_path} {after_path}")
    after = after.reindex(before.index)
    if not (
        np.array_equal(before.n_hits.to_numpy(), after.n_hits.to_numpy())
        and np.array_equal(before.n_strings.to_numpy(), after.n_strings.to_numpy())
    ):
        raise ValueError("Before/after selected hit and string counts differ")
    return before.reset_index(), after.reset_index()


def w1(exp: pd.DataFrame, mc: pd.DataFrame, weights: np.ndarray) -> float:
    return float(
        wasserstein_distance(exp.theta_deg.to_numpy(), mc.theta_deg.to_numpy(), v_weights=weights)
    )


def with_z_spread(predictions: pd.DataFrame, features_path: Path) -> pd.DataFrame:
    features = pd.read_csv(features_path).set_index("event_id", verify_integrity=True)
    if len(features) != len(predictions) or not predictions.event_id.isin(features.index).all():
        raise ValueError(f"Input features do not cover prediction IDs: {features_path}")
    aligned_features = features.reindex(predictions.event_id)
    for name in ("n_hits", "n_strings"):
        if not np.array_equal(predictions[name].to_numpy(), aligned_features[name].to_numpy()):
            raise ValueError(f"Input features disagree on {name}: {features_path}")
    if not np.isfinite(aligned_features.z_std_m).all():
        raise ValueError(f"Non-finite vertical hit spread: {features_path}")
    return predictions.assign(z_std_m=aligned_features.z_std_m.to_numpy())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before-exp", type=Path, required=True)
    parser.add_argument("--after-exp", type=Path, required=True)
    parser.add_argument("--before-mc", type=Path, required=True)
    parser.add_argument("--after-mc", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--exp-input-features", type=Path, default=None)
    parser.add_argument("--mc-input-features", type=Path, default=None)
    parser.add_argument("--bootstrap", type=int, default=500)
    args = parser.parse_args()
    if (args.exp_input_features is None) != (args.mc_input_features is None):
        raise ValueError("Pass both experimental and MC input-feature files")
    if args.bootstrap < 1:
        raise ValueError("Bootstrap count must be positive")
    exp_old, exp_new = aligned(args.before_exp, args.after_exp, False)
    mc_old, mc_new = aligned(args.before_mc, args.after_mc, True)
    weights, support = matching_weights(exp_old, mc_old)
    z_weights = None
    z_support = None
    if args.exp_input_features is not None:
        exp_old = with_z_spread(exp_old, args.exp_input_features)
        mc_old = with_z_spread(mc_old, args.mc_input_features)
        z_support, z_weights = measure(exp_old, mc_old, ("z_std_m",), return_weights=True)
    exp_components = ("direction_x", "direction_y", "direction_z")
    old_direction = exp_old[list(exp_components)].to_numpy()
    new_direction = exp_new[list(exp_components)].to_numpy()
    direction_change = np.rad2deg(
        np.arccos(np.clip(np.sum(old_direction * new_direction, axis=1), -1, 1))
    )
    rng = np.random.default_rng(83)
    changes = np.empty(args.bootstrap)
    z_changes = np.empty(args.bootstrap) if z_weights is not None else None
    for i in range(args.bootstrap):
        ei = rng.integers(len(exp_old), size=len(exp_old))
        mi = rng.integers(len(mc_old), size=len(mc_old))
        e0, e1 = exp_old.iloc[ei], exp_new.iloc[ei]
        m0, m1 = mc_old.iloc[mi], mc_new.iloc[mi]
        sampled_weights, _ = matching_weights(e0, m0)
        changes[i] = w1(e1, m1, sampled_weights) - w1(e0, m0, sampled_weights)
        if z_changes is not None:
            _, sampled_z_weights = measure(e0, m0, ("z_std_m",), return_weights=True)
            z_changes[i] = w1(e1, m1, sampled_z_weights) - w1(e0, m0, sampled_z_weights)
    report = {
        "experimental_events": len(exp_old),
        "unique_muatm_mc_events": len(mc_old),
        "matched_theta_w1_before_deg": w1(exp_old, mc_old, weights),
        "matched_theta_w1_after_deg": w1(exp_new, mc_new, weights),
        "w1_delta_after_minus_before_ci95_deg": np.quantile(changes, [0.025, 0.975]).tolist(),
        "experimental_direction_change_q50_q68_deg": np.quantile(
            direction_change, [0.5, 0.68]
        ).tolist(),
        "experiment_theta_q10_q50_q90_before_deg": np.quantile(
            exp_old.theta_deg, [0.1, 0.5, 0.9]
        ).tolist(),
        "experiment_theta_q10_q50_q90_after_deg": np.quantile(
            exp_new.theta_deg, [0.1, 0.5, 0.9]
        ).tolist(),
        "matching_support": support,
        "warning": "Experimental data have no angle truth; W1 is distributional only.",
    }
    if z_weights is not None:
        report.update(
            zspread_matched_theta_w1_before_deg=w1(exp_old, mc_old, z_weights),
            zspread_matched_theta_w1_after_deg=w1(exp_new, mc_new, z_weights),
            zspread_w1_delta_after_minus_before_ci95_deg=np.quantile(
                z_changes, [0.025, 0.975]
            ).tolist(),
            zspread_matching_support={
                key: z_support[key]
                for key in (
                    "exp_unsupported_events",
                    "exp_unsupported_fraction",
                    "mc_effective_sample_size",
                    "bin_edges",
                )
            },
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "paired_domain.json").write_text(json.dumps(report, indent=2) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    edges = np.arange(70, 186, 5)
    for axis, exp, mc, title in (
        (axes[0], exp_old, mc_old, "Before"),
        (axes[1], exp_new, mc_new, "After"),
    ):
        axis.hist(
            exp.theta_deg,
            bins=edges,
            weights=np.ones(len(exp)) / len(exp),
            histtype="step",
            linewidth=1.8,
            label="Experiment",
        )
        axis.hist(
            mc.theta_deg,
            bins=edges,
            weights=weights,
            histtype="step",
            linewidth=1.8,
            label="muatm MC, hit/string matched",
        )
        if z_weights is not None:
            axis.hist(
                mc.theta_deg,
                bins=edges,
                weights=z_weights,
                histtype="step",
                linewidth=1.8,
                label="muatm MC, plus z-spread matched",
            )
        axis.set_title(title)
        axis.set_xlabel("Predicted θ [deg]")
        axis.set_ylabel("Fraction / 5° bin")
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output_dir / "paired_domain.png", dpi=180)
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
