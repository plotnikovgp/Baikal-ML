#!/usr/bin/env python3
"""Compare MC direction errors of two checkpoints on matching unique event IDs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def summarize(before: np.ndarray, after: np.ndarray, rng: np.random.Generator) -> dict:
    if not len(before):
        return {"count": 0}
    changes = np.empty(500)
    for i in range(len(changes)):
        selection = rng.integers(0, len(before), len(before))
        changes[i] = np.quantile(after[selection], 0.68) - np.quantile(before[selection], 0.68)
    return {
        "count": int(len(before)),
        "q50_before_deg": float(np.quantile(before, 0.5)),
        "q50_after_deg": float(np.quantile(after, 0.5)),
        "q68_before_deg": float(np.quantile(before, 0.68)),
        "q68_after_deg": float(np.quantile(after, 0.68)),
        "q68_delta_after_minus_before_deg": float(
            np.quantile(after, 0.68) - np.quantile(before, 0.68)
        ),
        "q68_delta_bootstrap_95pct_deg": np.quantile(changes, [0.025, 0.975]).tolist(),
        "fraction_individual_errors_improved": float(np.mean(after < before)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    old, new = np.load(args.before), np.load(args.after)
    common, i_old, i_new = np.intersect1d(old["event_id"], new["event_id"], return_indices=True)
    if len(common) < 100:
        raise ValueError("Too few matching MC event IDs")
    old_error, new_error = old["angle_deg"][i_old], new["angle_deg"][i_new]
    single = old["single_track"][i_old]
    if not np.array_equal(single, new["single_track"][i_new]):
        raise ValueError("Track multiplicity labels differ on common IDs")
    rng = np.random.default_rng(107)
    report = {
        "matching_unique_event_ids": int(len(common)),
        "all": summarize(old_error, new_error, rng),
        "single_track": summarize(old_error[single], new_error[single], rng),
        "multi_track_or_invalid": summarize(old_error[~single], new_error[~single], rng),
        "caveat": "Multi-track prime_prty angle is not a unique-track ground truth.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
