#!/usr/bin/env python3
"""Paired unique-ID q50/q68 comparison of photon-head continuations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def summarize(before: np.ndarray, after: np.ndarray, reps: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    deltas = {q: np.empty(reps) for q in (0.50, 0.68)}
    for index in range(reps):
        selection = rng.integers(0, len(before), len(before))
        for q in deltas:
            deltas[q][index] = np.quantile(after[selection], q) - np.quantile(before[selection], q)
    return {
        "count": len(before),
        "before_q50_deg": float(np.quantile(before, 0.50)),
        "after_q50_deg": float(np.quantile(after, 0.50)),
        "before_q68_deg": float(np.quantile(before, 0.68)),
        "after_q68_deg": float(np.quantile(after, 0.68)),
        "q50_delta_ci95_deg": np.quantile(deltas[0.50], (0.025, 0.975)).tolist(),
        "q68_delta_ci95_deg": np.quantile(deltas[0.68], (0.025, 0.975)).tolist(),
        "before_ge20": int((before >= 20).sum()),
        "after_ge20": int((after >= 20).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=500)
    args = parser.parse_args()
    baseline = pd.read_csv(args.baseline).set_index("event_id", verify_integrity=True)
    candidate = pd.read_csv(args.candidate).set_index("event_id", verify_integrity=True)
    if not baseline.index.equals(candidate.index):
        raise ValueError("Candidate and baseline unique event IDs/order differ")
    report = {
        "baseline": str(args.baseline),
        "candidate": str(args.candidate),
        "replicates": args.replicates,
        "groups": {},
    }
    groups = {
        "all": np.ones(len(baseline), dtype=bool),
        "muatm": baseline.particle.to_numpy() == "muatm",
        "nuatm": baseline.particle.to_numpy() == "nuatm",
        "nue2": baseline.particle.to_numpy() == "nue2",
        "span_lt120": baseline.hit_span_m.to_numpy() < 120,
        "span_ge200": baseline.hit_span_m.to_numpy() >= 200,
    }
    for index, (name, selected) in enumerate(groups.items()):
        old = baseline.candidate_angle.to_numpy()[selected]
        new = candidate.candidate_angle.to_numpy()[selected]
        report["groups"][name] = summarize(old, new, args.replicates, 1901 + index)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
