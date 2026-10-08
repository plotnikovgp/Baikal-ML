#!/usr/bin/env python3
"""Build cross-dataset event masks for honest mixed-input validation/test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

SPLITS = ("train", "val", "test")


def ids(path: Path) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as h5:
        return {split: np.asarray(h5[f"{split}/ev_ids/data"]) for split in SPLITS}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predicted-hits", type=Path, required=True)
    parser.add_argument("--gt-nue2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    pred = ids(args.predicted_hits)
    gt = ids(args.gt_nue2)
    pred_holdout = np.concatenate((pred["val"], pred["test"]))
    masks = {
        # Nue2 GT train may not contain any event used to select or test the
        # all-particle predicted-hit model.
        "gt_train": ~np.isin(gt["train"], pred_holdout),
        # The supplied universal checkpoint was already trained on pred train.
        # These are the only independent portions of GT validation and test.
        "gt_val": ~np.isin(gt["val"], pred["train"]),
        "gt_test": ~np.isin(gt["test"], pred["train"]),
        # The specialist checkpoint was already trained on GT train. Its
        # all-particle predicted-hit validation/test must exclude those IDs.
        "pred_val": ~np.isin(pred["val"], gt["train"]),
        "pred_test": ~np.isin(pred["test"], gt["train"]),
    }
    counts = {
        key: {"total": int(len(mask)), "allowed": int(mask.sum())} for key, mask in masks.items()
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **masks)
    manifest = {
        "predicted_hits": str(args.predicted_hits),
        "gt_nue2": str(args.gt_nue2),
        "rule": (
            "GT train excludes predicted-hit val/test; GT val/test exclude "
            "predicted-hit train seen by the universal initialization; "
            "predicted-hit val/test exclude GT train seen by the specialist initialization"
        ),
        "counts": counts,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
