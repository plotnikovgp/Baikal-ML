#!/usr/bin/env python3
"""Event-paired bootstrap for direction q50/q68 differences on the reco test set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=400)
    args = parser.parse_args()
    with np.load(args.events) as events:
        official = np.asarray(events["official_cut"], dtype=bool)
        errors = {
            name: np.asarray(events[f"{name}_direction_error_deg"], dtype=np.float64)
            for name in ("standard_reco", "v4", "v4_photon", "short_photon")
        }
    rng = np.random.default_rng(20261013)
    comparisons = (
        ("v4_photon", "standard_reco"),
        ("short_photon", "v4_photon"),
        ("v4_photon", "v4"),
    )
    result = {
        "definition": "first minus second; resample identical events for both methods",
        "replicates": args.replicates,
        "stages": {},
    }
    for stage, mask in (
        ("no_official_cut", np.ones(len(official), dtype=bool)),
        ("official_reco_cut", official),
    ):
        stage_out = {"count": int(mask.sum()), "comparisons": {}}
        selected = {name: values[mask] for name, values in errors.items()}
        n = len(selected["v4"])
        for first, second in comparisons:
            pair = {}
            for q in (0.50, 0.68):
                observed = float(np.quantile(selected[first], q) - np.quantile(selected[second], q))
                samples = np.empty(args.replicates, dtype=np.float64)
                for rep in range(args.replicates):
                    indices = rng.integers(0, n, size=n)
                    samples[rep] = np.quantile(selected[first][indices], q) - np.quantile(
                        selected[second][indices], q
                    )
                pair[f"q{round(q * 100):02d}"] = {
                    "delta_deg": observed,
                    "ci95_deg": np.quantile(samples, (0.025, 0.975)).tolist(),
                }
            stage_out["comparisons"][f"{first}_minus_{second}"] = pair
        result["stages"][stage] = stage_out
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
