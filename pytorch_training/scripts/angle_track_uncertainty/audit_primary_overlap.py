#!/usr/bin/env python3
"""Audit exact generated-primary overlap across Baikal angle datasets.

Equal float32 (theta, phi, energy) is evidence of a shared generated primary,
not by itself evidence of an identical detector event or hit realization.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def primary_keys(values: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(values[:, :3]).view("V12").reshape(-1)


def keys(path: Path, split: str) -> np.ndarray:
    with h5py.File(path, "r") as source:
        return primary_keys(source[f"{split}/prime_prty/data"][:])


def count(values: np.ndarray, reference: np.ndarray) -> dict[str, int | float]:
    seen = np.isin(values, reference)
    return {
        "events": int(len(values)),
        "same_primary": int(seen.sum()),
        "same_primary_fraction": float(seen.mean()),
        "new_primary": int((~seen).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predicted-2020", type=Path, required=True)
    parser.add_argument("--gt-2020", type=Path, required=True)
    parser.add_argument("--reco-2020", type=Path, required=True)
    parser.add_argument("--raw-2019", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pred_train = np.unique(keys(args.predicted_2020, "train"))
    gt_train = np.unique(keys(args.gt_2020, "train"))
    reco_train = np.unique(keys(args.reco_2020, "train"))
    internal_train = np.union1d(pred_train, gt_train)
    result = {
        "fingerprint": "exact float32 primary theta, phi, energy",
        "interpretation": "shared generated primary; not necessarily identical detector hits",
        "paths": {
            name: str(getattr(args, name))
            for name in ("predicted_2020", "gt_2020", "reco_2020", "raw_2019")
        },
        "unique_train_primaries": {
            "predicted_2020": int(len(pred_train)),
            "gt_2020": int(len(gt_train)),
            "reco_2020": int(len(reco_train)),
            "internal_union": int(len(internal_train)),
        },
        "reco_vs_internal_train": {
            split: count(keys(args.reco_2020, split), internal_train)
            for split in ("train", "val", "test")
        },
        "reco_vs_reco_train": {
            split: count(keys(args.reco_2020, split), reco_train) for split in ("val", "test")
        },
        "gt_vs_gt_train": {
            split: count(keys(args.gt_2020, split), gt_train) for split in ("val", "test")
        },
        "predicted_vs_internal_train": {
            split: count(keys(args.predicted_2020, split), internal_train)
            for split in ("val", "test")
        },
        "raw_2019_vs_internal_train": count(keys(args.raw_2019, "test"), internal_train),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
