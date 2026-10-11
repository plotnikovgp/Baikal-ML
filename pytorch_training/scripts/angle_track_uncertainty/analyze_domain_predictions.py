#!/usr/bin/env python3
"""Audit angle-prediction shift between experimental data and muatm MC.

Both inputs must have used the same signal/noise checkpoint, probability cut,
and 2-string/8-hit selection. Matching weights use only observed hit/string
counts, never the predicted angles.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance


def load_predictions(paths: list[Path], *, deduplicate: bool = False) -> pd.DataFrame:
    frames = [pd.read_csv(path).assign(source_split=path.stem) for path in paths]
    result = pd.concat(frames, ignore_index=True)
    duplicates = int(result.event_id.duplicated().sum())
    if duplicates and not deduplicate:
        raise ValueError("Repeated event IDs across supplied prediction files")
    if deduplicate:
        result = result.drop_duplicates("event_id", keep="first").copy()
    result.attrs["duplicate_rows_removed"] = duplicates if deduplicate else 0
    for field in ("theta_deg", "phi_deg", "n_hits", "n_strings"):
        if not np.isfinite(result[field]).all():
            raise ValueError(f"Non-finite {field} values in predictions")
    return result


def attach_mc_track_validity(mc: pd.DataFrame, anchor_path: Path) -> tuple[pd.DataFrame, dict]:
    validity: dict[str, bool] = {}
    conflicts = 0
    with h5py.File(anchor_path, "r") as h5:
        for split in ("train", "val", "test"):
            event_ids = h5[f"{split}/event_ids"][:]
            valid = h5[f"{split}/valid"][:]
            for event_id, is_valid in zip(event_ids, valid):
                key = event_id.decode() if isinstance(event_id, bytes) else str(event_id)
                value = bool(is_valid)
                if key in validity and validity[key] != value:
                    conflicts += 1
                else:
                    validity[key] = value
    mc = mc.copy()
    matched = mc.event_id.map(validity)
    mc["single_track_known"] = matched.fillna(False).astype(bool)
    mc["validity_known"] = matched.notna()
    return mc, {
        "anchor_event_ids": len(validity),
        "conflicting_duplicate_ids": conflicts,
        "mc_matched_count": int(mc.validity_known.sum()),
        "mc_unmatched_count": int((~mc.validity_known).sum()),
        "mc_single_track_count": int(mc.single_track_known.sum()),
    }


def matching_weights(exp: pd.DataFrame, mc: pd.DataFrame) -> tuple[np.ndarray, dict]:
    hit_edges = np.array([8, 10, 12, 16, 24, 40, np.inf])
    string_edges = np.array([2, 3, 4, 6, 9, np.inf])

    def cells(frame: pd.DataFrame) -> np.ndarray:
        hit_bin = np.searchsorted(hit_edges, frame.n_hits.to_numpy(), side="right") - 1
        string_bin = np.searchsorted(string_edges, frame.n_strings.to_numpy(), side="right") - 1
        hit_bin = np.clip(hit_bin, 0, len(hit_edges) - 2)
        string_bin = np.clip(string_bin, 0, len(string_edges) - 2)
        return hit_bin * (len(string_edges) - 1) + string_bin

    exp_cell, mc_cell = cells(exp), cells(mc)
    n_cells = (len(hit_edges) - 1) * (len(string_edges) - 1)
    exp_count = np.bincount(exp_cell, minlength=n_cells)
    mc_count = np.bincount(mc_cell, minlength=n_cells)
    unsupported = int(exp_count[mc_count == 0].sum())
    factor = np.zeros(n_cells, dtype=np.float64)
    supported = mc_count > 0
    factor[supported] = exp_count[supported] / mc_count[supported]
    weights = factor[mc_cell]
    if weights.sum() == 0:
        raise ValueError("No shared hit/string-count support")
    weights /= weights.sum()
    ess = 1.0 / np.square(weights).sum()
    return weights, {
        "exp_outside_mc_support_count": unsupported,
        "exp_outside_mc_support_fraction": unsupported / len(exp),
        "mc_effective_sample_size": float(ess),
        "mc_nonzero_weight_count": int(np.count_nonzero(weights)),
        "hit_edges": hit_edges[:-1].astype(int).tolist(),
        "string_edges": string_edges[:-1].astype(int).tolist(),
    }


def weighted_ks(x: np.ndarray, y: np.ndarray, y_weights: np.ndarray) -> float:
    x_sorted = np.sort(x)
    y_order = np.argsort(y)
    y_sorted = y[y_order]
    y_cdf = np.cumsum(y_weights[y_order])
    values = np.sort(np.concatenate((x_sorted, y_sorted)))
    x_fraction = np.searchsorted(x_sorted, values, side="right") / len(x_sorted)
    y_positions = np.searchsorted(y_sorted, values, side="right") - 1
    y_fraction = np.where(y_positions >= 0, y_cdf[np.maximum(y_positions, 0)], 0.0)
    return float(np.max(np.abs(x_fraction - y_fraction)))


def weighted_quantile(values: np.ndarray, weights: np.ndarray, fraction: float) -> float:
    order = np.argsort(values)
    sorted_values = values[order]
    cumulative = np.cumsum(weights[order])
    return float(sorted_values[min(np.searchsorted(cumulative, fraction), len(values) - 1)])


def comparison(exp: pd.DataFrame, mc: pd.DataFrame, weights: np.ndarray | None) -> dict:
    x, y = exp.theta_deg.to_numpy(), mc.theta_deg.to_numpy()
    if weights is None:
        weights = np.full(len(y), 1.0 / len(y))
    return {
        "theta_wasserstein_deg": float(wasserstein_distance(x, y, v_weights=weights)),
        "theta_ks": weighted_ks(x, y, weights),
        "theta_gt90_exp": float(np.mean(x > 90)),
        "theta_gt90_mc": float(np.sum(weights[y > 90])),
        "theta_median_exp_deg": float(np.median(x)),
        "theta_median_mc_deg": weighted_quantile(y, weights, 0.5),
    }


def describe(frame: pd.DataFrame) -> dict:
    result = {
        "count": len(frame),
        "theta_q10_q50_q90_deg": np.quantile(frame.theta_deg, [0.1, 0.5, 0.9]).tolist(),
        "n_hits_q50_q90": np.quantile(frame.n_hits, [0.5, 0.9]).tolist(),
        "n_strings_q50_q90": np.quantile(frame.n_strings, [0.5, 0.9]).tolist(),
    }
    if "sigma_angle_deg" in frame and np.isfinite(frame.sigma_angle_deg).all():
        result["sigma_angle_q50_q90_deg"] = np.quantile(frame.sigma_angle_deg, [0.5, 0.9]).tolist()
    return result


def plot(
    exp: pd.DataFrame,
    mc: pd.DataFrame,
    weights: np.ndarray,
    output: Path,
    threshold: float,
    model_label: str,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    fig.suptitle(
        f"{model_label} predictions: experiment vs muatm MC, common p≥{threshold:.2f} and 2s8h cut"
    )
    samples = [
        (exp, None, "Experimental", "black", "-"),
        (mc, None, "muatm MC", "tab:orange", "-"),
        (mc, weights, "muatm MC, hit/string-matched", "tab:blue", "--"),
    ]
    single = mc[mc.single_track_known]
    if len(single) > 100:
        samples.append((single, None, "muatm MC, one truth track", "tab:green", ":"))
    specifications = (
        ("theta_deg", np.arange(0, 185, 5), "Predicted θ [degrees]", axes[0, 0]),
        ("phi_deg", np.arange(0, 365, 10), "Predicted φ [degrees]", axes[0, 1]),
        ("n_hits", np.arange(7.5, 41.5, 1), "Selected hits", axes[1, 0]),
        ("n_strings", np.arange(1.5, 13.5, 1), "Strings", axes[1, 1]),
    )
    for field, edges, xlabel, ax in specifications:
        for frame, sample_weights, label, color, linestyle in samples:
            if field in ("n_hits", "n_strings") and sample_weights is not None:
                continue
            values = frame[field].to_numpy()
            hist, _ = np.histogram(values, bins=edges, weights=sample_weights)
            hist = hist / max(hist.sum(), 1e-12)
            ax.stairs(hist, edges, label=label, color=color, linewidth=1.8, linestyle=linestyle)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Fraction per bin")
        ax.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-csv", type=Path, nargs="+", required=True)
    parser.add_argument("--mc-csv", type=Path, nargs="+", required=True)
    parser.add_argument("--p070-anchors", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--model-label", default="v3")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    exp = load_predictions(args.exp_csv)
    mc_raw = load_predictions(args.mc_csv, deduplicate=True)
    duplicate_rows_removed = mc_raw.attrs["duplicate_rows_removed"]
    mc, matching = attach_mc_track_validity(mc_raw, args.p070_anchors)
    weights, support = matching_weights(exp, mc)
    report = {
        "model_label": args.model_label,
        "selection": (
            f"same signal/noise checkpoint, p>={args.threshold:.2f}, >=8 hits on >=2 strings"
        ),
        "warning": (
            "angle model trained at p>=0.70; no experimental ground truth"
            if args.threshold != 0.7
            else "input hit threshold matches angle training; no experimental ground truth"
        ),
        "mc_duplicate_event_id_rows_removed": duplicate_rows_removed,
        "mc_duplicate_policy": "first occurrence per event ID; repeated IDs are not exact copies",
        "experimental": describe(exp),
        "muatm_mc": describe(mc),
        "anchor_matching": matching,
        "input_matching": support,
        "input_ks_before": {
            field: weighted_ks(
                exp[field].to_numpy(),
                mc[field].to_numpy(),
                np.full(len(mc), 1.0 / len(mc)),
            )
            for field in ("n_hits", "n_strings")
        },
        "input_ks_after": {
            field: weighted_ks(exp[field].to_numpy(), mc[field].to_numpy(), weights)
            for field in ("n_hits", "n_strings")
        },
        "unweighted": comparison(exp, mc, None),
        "input_matched": comparison(exp, mc, weights),
    }
    for split in ("exp_train", "exp_val", "exp_test"):
        subset = exp[exp.source_split == split]
        if len(subset):
            report[split] = describe(subset)
    if "exp_train" in report and "exp_test" in report:
        report["exp_train_vs_test"] = comparison(
            exp[exp.source_split == "exp_train"],
            exp[exp.source_split == "exp_test"],
            None,
        )
    for split in ("muatm_val", "muatm_test"):
        subset = mc[mc.source_split == split]
        if len(subset):
            report[split] = describe(subset)
    for name, subset in (
        ("muatm_single", mc[mc.single_track_known]),
        ("muatm_multi_or_unknown", mc[~mc.single_track_known]),
    ):
        if len(subset):
            report[name] = describe(subset)
            report[f"exp_vs_{name}"] = comparison(exp, subset, None)
    (args.output_dir / "domain_report.json").write_text(json.dumps(report, indent=2) + "\n")
    plot(
        exp,
        mc,
        weights,
        args.output_dir / "domain_predictions.png",
        args.threshold,
        args.model_label,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
