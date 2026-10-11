#!/usr/bin/env python3
"""Paired, event-ID-deduplicated q68 comparison of two saved test predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def selected_ids(data: Path, anchors: Path | None, mask: np.ndarray) -> np.ndarray:
    with h5py.File(data, "r") as h5:
        ids = h5["test/ev_ids/data"][:]
    if len(ids) != len(mask):
        raise ValueError("Mask and data event counts differ")
    selected = mask.copy()
    if anchors is not None:
        with h5py.File(anchors, "r") as h5:
            valid = h5["test/valid"][:]
        if len(valid) != len(ids):
            raise ValueError("Anchor and data event counts differ")
        is_mu = np.fromiter((key.startswith(b"muatm") for key in ids), bool, len(ids))
        selected &= ~is_mu | valid
    return ids[selected]


def quantile(values: np.ndarray) -> float:
    return float(np.quantile(values, 0.68))


def paired_report(ids: np.ndarray, before: np.ndarray, after: np.ndarray, *, reps: int) -> dict:
    if not (len(ids) == len(before) == len(after)):
        raise ValueError("Saved predictions do not align with selected event IDs")
    _, unique_index = np.unique(ids, return_index=True)
    unique_index.sort()
    old, new = before[unique_index], after[unique_index]
    rng = np.random.default_rng(73)
    changes = np.empty(reps)
    for index in range(reps):
        sample = rng.integers(0, len(old), len(old))
        changes[index] = quantile(new[sample]) - quantile(old[sample])
    return {
        "rows": len(ids),
        "unique_event_ids": len(unique_index),
        "q68_before": quantile(old),
        "q68_after": quantile(new),
        "q68_delta_after_minus_before": quantile(new) - quantile(old),
        "paired_bootstrap_95pct_delta": np.quantile(changes, [0.025, 0.975]).tolist(),
        "fraction_individual_errors_improved": float(np.mean(new < old)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--masks", type=Path, required=True)
    parser.add_argument("--pred-data", type=Path, required=True)
    parser.add_argument("--pred-anchors", type=Path, required=True)
    parser.add_argument("--gt-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.masks) as masks:
        pred_ids = selected_ids(args.pred_data, args.pred_anchors, masks["pred_test"])
        gt_ids = selected_ids(args.gt_data, None, masks["gt_test"])
    old, new = np.load(args.before), np.load(args.after)
    report = {
        "pred_angle_deg": paired_report(pred_ids, old["pred_angle"], new["pred_angle"], reps=500),
        "pred_line_m": paired_report(pred_ids, old["pred_line"], new["pred_line"], reps=500),
        "gt_angle_deg": paired_report(gt_ids, old["gt_angle"], new["gt_angle"], reps=200),
        "gt_line_m": paired_report(gt_ids, old["gt_line"], new["gt_line"], reps=200),
        "note": (
            "Same leakage-free selected test IDs, one row per unique event ID; "
            "paired bootstrap resamples IDs, not hits"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
