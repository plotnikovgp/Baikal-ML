#!/usr/bin/env python3
"""Compare the signal/noise score distribution in muatm MC and experiment."""

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
from analyze_domain_predictions import matching_weights
from scipy.stats import wasserstein_distance


def attach_scores(predictions: Path, data: Path, split: str) -> pd.DataFrame:
    result = pd.read_csv(predictions)
    source = result.source_event_index.to_numpy(dtype=np.int64)
    if (np.diff(source) <= 0).any():
        raise ValueError("source_event_index must be strictly increasing")
    with h5py.File(data, "r") as h5:
        group = h5[split]
        ids = np.char.decode(group["ev_ids/data"][source].astype("S"), "utf-8")
        if not np.array_equal(ids, result.event_id.to_numpy()):
            raise ValueError("Prediction event IDs do not match the HDF5 rows")
        starts = np.asarray(group["ev_starts/data"][:], dtype=np.int64)
        scores = np.asarray(group["signal_probability/data"][:], dtype=np.float32)
    mean = np.empty(len(source), dtype=np.float32)
    low_fraction = np.empty(len(source), dtype=np.float32)
    high_fraction = np.empty(len(source), dtype=np.float32)
    for i, index in enumerate(source):
        selected = scores[starts[index] : starts[index + 1]]
        mean[i] = selected.mean()
        low_fraction[i] = np.mean(selected < 0.8)
        high_fraction[i] = np.mean(selected >= 0.95)
    result["p_mean"] = mean
    result["p_below_080_fraction"] = low_fraction
    result["p_ge_095_fraction"] = high_fraction
    return result


def quantiles(values: np.ndarray, weights: np.ndarray | None = None) -> dict:
    if weights is None:
        return {
            "q10": float(np.quantile(values, 0.1)),
            "q50": float(np.quantile(values, 0.5)),
            "q90": float(np.quantile(values, 0.9)),
        }
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    sorted_values = values[order]
    return {
        f"q{int(fraction * 100)}": float(sorted_values[np.searchsorted(cumulative, fraction)])
        for fraction in (0.1, 0.5, 0.9)
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-pred", type=Path, required=True)
    parser.add_argument("--exp-data", type=Path, required=True)
    parser.add_argument("--mc-pred", type=Path, required=True)
    parser.add_argument("--mc-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    exp = attach_scores(args.exp_pred, args.exp_data, "test")
    mc = attach_scores(args.mc_pred, args.mc_data, "val")
    mc_duplicates = int(mc.event_id.duplicated().sum())
    mc = mc.drop_duplicates("event_id", keep="first")
    weights, support = matching_weights(exp, mc)
    fields = ("p_mean", "p_below_080_fraction", "p_ge_095_fraction")
    report = {
        "experimental_events": len(exp),
        "unique_muatm_mc_events": len(mc),
        "muatm_duplicates_removed": mc_duplicates,
        "input_matching": support,
        "fields": {},
    }
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7))
    for field, axis in zip(fields, axes):
        x, y = exp[field].to_numpy(), mc[field].to_numpy()
        report["fields"][field] = {
            "experiment": quantiles(x),
            "muatm_mc": quantiles(y),
            "muatm_mc_hit_string_matched": quantiles(y, weights),
            "w1_unweighted": float(wasserstein_distance(x, y)),
            "w1_hit_string_matched": float(wasserstein_distance(x, y, v_weights=weights)),
        }
        edges = np.linspace(0, 1, 41) if field != "p_mean" else np.linspace(0.7, 1, 31)
        axis.hist(x, bins=edges, weights=np.ones(len(x)) / len(x), histtype="step", label="exp")
        axis.hist(y, bins=edges, weights=weights, histtype="step", label="muatm MC matched")
        axis.set_xlabel(field)
        axis.grid(alpha=0.2)
    axes[0].legend()
    fig.tight_layout()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_dir / "signal_probability_domain.png", dpi=180)
    plt.close(fig)
    (args.output_dir / "signal_probability_domain.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
