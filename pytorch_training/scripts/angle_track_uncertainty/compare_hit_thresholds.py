#!/usr/bin/env python3
"""Compare p>=0.70 and p>=0.90 inputs on exactly the same MC/exp event IDs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def quantiles(values: np.ndarray) -> dict:
    return {
        "count": int(len(values)),
        "q50": float(np.quantile(values, 0.5)),
        "q68": float(np.quantile(values, 0.68)),
        "q95": float(np.quantile(values, 0.95)),
    }


def paired_mc(low_path: Path, high_path: Path) -> dict:
    low, high = np.load(low_path), np.load(high_path)
    common, i_low, i_high = np.intersect1d(low["event_id"], high["event_id"], return_indices=True)
    if len(common) < 100:
        raise ValueError("Too few matching MC event IDs")
    low_error, high_error = low["angle_deg"][i_low], high["angle_deg"][i_high]
    single_low, single_high = low["single_track"][i_low], high["single_track"][i_high]
    if not np.array_equal(single_low, single_high):
        raise ValueError("MC multiplicity labels disagree between thresholds")

    def summary(selected: np.ndarray) -> dict:
        earlier, later = low_error[selected], high_error[selected]
        return {
            "p070_angle_deg": quantiles(earlier),
            "p090_angle_deg": quantiles(later),
            "paired_delta_p090_minus_p070_deg": quantiles(later - earlier),
            "fraction_p090_better": float(np.mean(later < earlier)),
        }

    rng = np.random.default_rng(47)
    n = len(common)
    q68_deltas = np.empty(500)
    for index in range(len(q68_deltas)):
        sample = rng.integers(0, n, n)
        q68_deltas[index] = np.quantile(high_error[sample], 0.68) - np.quantile(
            low_error[sample], 0.68
        )
    return {
        "matching_event_ids": n,
        "p070_unique_count": len(low["event_id"]),
        "p090_unique_count": len(high["event_id"]),
        "all": summary(np.ones(n, dtype=bool)),
        "single_track": summary(single_low),
        "multi_track_or_invalid": summary(~single_low),
        "q68_delta_bootstrap_95pct_deg": np.quantile(q68_deltas, [0.025, 0.975]).tolist(),
    }


def read_predictions(paths: list[Path]) -> pd.DataFrame:
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    if frame.event_id.duplicated().any():
        raise ValueError("Experimental event IDs must be unique")
    return frame.set_index("event_id", verify_integrity=True)


def paired_exp(low_paths: list[Path], high_paths: list[Path]) -> dict:
    low, high = read_predictions(low_paths), read_predictions(high_paths)
    common = low.index.intersection(high.index)
    if len(common) < 100:
        raise ValueError("Too few matching experimental event IDs")
    earlier, later = low.loc[common], high.loc[common]
    columns = ["direction_x", "direction_y", "direction_z"]
    cosine = np.sum(earlier[columns].to_numpy() * later[columns].to_numpy(), axis=1)
    change_deg = np.rad2deg(np.arccos(np.clip(cosine, -1.0, 1.0)))
    return {
        "p070_event_count": len(low),
        "p090_event_count": len(high),
        "matching_event_ids": len(common),
        "p090_events_found_in_p070_fraction": len(common) / len(high),
        "direction_change_deg": quantiles(change_deg),
        "direction_change_gt5deg_fraction": float(np.mean(change_deg > 5)),
        "theta_change_p090_minus_p070_deg": quantiles(
            later.theta_deg.to_numpy() - earlier.theta_deg.to_numpy()
        ),
        "hits_removed_p070_minus_p090": quantiles(
            earlier.n_hits.to_numpy() - later.n_hits.to_numpy()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mc-p070", type=Path, required=True)
    parser.add_argument("--mc-p090", type=Path, required=True)
    parser.add_argument("--exp-p070", type=Path, nargs="+", required=True)
    parser.add_argument("--exp-p090", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "mc": paired_mc(args.mc_p070, args.mc_p090),
        "experiment": paired_exp(args.exp_p070, args.exp_p090),
        "caveat": (
            "MC multi-track prime_prty is a diagnostic label, not a unique truth track; "
            "repeated MC event IDs were deduplicated by first occurrence"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
