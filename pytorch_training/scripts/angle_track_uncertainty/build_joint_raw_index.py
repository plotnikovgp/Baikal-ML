#!/usr/bin/env python3
"""Index unfiltered MC/experiment events for a leakage-free joint-hit study.

Training events have >=8 true-signal hits on >=2 strings, inherited from the
audited compact signal file. Multi-track muatm events are excluded. Validation
and test use exactly the existing predicted-hit event-ID cuts, deduplicated by
ID. The raw HDF5 files remain untouched; this small index stores row numbers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def lookup_ids(source_ids: np.ndarray, wanted_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map IDs to first source row, with a present flag for missing IDs."""
    order = np.argsort(source_ids, kind="stable")
    sorted_ids = source_ids[order]
    positions = np.searchsorted(sorted_ids, wanted_ids)
    safe = np.minimum(positions, len(sorted_ids) - 1)
    present = (positions < len(sorted_ids)) & (sorted_ids[safe] == wanted_ids)
    result = np.full(len(wanted_ids), -1, dtype=np.int64)
    result[present] = order[safe[present]]
    return result, present


def unique_first(ids: np.ndarray, *aligned: np.ndarray) -> tuple[np.ndarray, ...]:
    """Keep first row for a duplicate event ID, preserving original order."""
    _unique, first = np.unique(ids, return_index=True)
    first.sort()
    return (ids[first], *(array[first] for array in aligned))


def map_truth_points(
    ids: np.ndarray,
    anchor_ids: np.ndarray,
    anchor_valid: np.ndarray,
    anchor_point: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    index, present = lookup_ids(anchor_ids, ids)
    safe = np.maximum(index, 0)
    valid = present & anchor_valid[safe] & np.isfinite(anchor_point[safe]).all(axis=1)
    points = np.zeros((len(ids), 3), dtype=np.float32)
    points[valid] = anchor_point[safe[valid]]
    return points, valid


def class_counts(ids: np.ndarray) -> dict[str, int]:
    prefix = ids.astype("S5")
    return {
        name: int(np.count_nonzero(prefix == code))
        for name, code in (
            ("muatm", b"muatm"),
            ("nuatm", b"nuatm"),
            ("nue2", b"nue2_"),
        )
    }


def build(args: argparse.Namespace) -> dict:
    arrays: dict[str, np.ndarray] = {}
    summary: dict[str, object] = {
        "files": {
            name: str(getattr(args, name).resolve())
            for name in (
                "raw_mc",
                "clean_data",
                "clean_anchors",
                "pred_data",
                "pred_anchors",
                "gt_data",
                "masks",
                "raw_exp",
                "pred_exp",
            )
        }
    }
    with (
        h5py.File(args.raw_mc, "r") as raw,
        h5py.File(args.clean_data, "r") as clean,
        h5py.File(args.clean_anchors, "r") as clean_anchor,
        h5py.File(args.pred_data, "r") as predicted,
        h5py.File(args.pred_anchors, "r") as pred_anchor,
        h5py.File(args.gt_data, "r") as gt,
        h5py.File(args.raw_exp, "r") as raw_exp,
        h5py.File(args.pred_exp, "r") as pred_exp,
        np.load(args.masks) as masks,
    ):
        excluded = np.unique(
            np.concatenate(
                (
                    np.asarray(predicted["val/ev_ids/data"])[masks["pred_val"]],
                    np.asarray(predicted["test/ev_ids/data"])[masks["pred_test"]],
                    np.asarray(gt["val/ev_ids/data"])[masks["gt_val"]],
                    np.asarray(gt["test/ev_ids/data"])[masks["gt_test"]],
                )
            )
        )
        print(f"excluded evaluation IDs: {len(excluded):,}", flush=True)

        clean_ids = np.asarray(clean["train/ev_ids/data"])
        clean_valid = np.asarray(clean_anchor["train/valid"], dtype=bool)
        if not np.array_equal(clean_ids, np.asarray(clean_anchor["train/event_ids"])):
            raise ValueError("Clean data and clean anchors do not align")
        prefix = clean_ids.astype("S5")
        use = (prefix == b"nuatm") | (prefix == b"nue2_") | ((prefix == b"muatm") & clean_valid)
        print(f"clean-cut, single-track train rows: {int(use.sum()):,}", flush=True)
        (train_ids,) = unique_first(clean_ids[use])
        overlap = np.isin(train_ids, excluded, assume_unique=False)
        overlap_count = int(overlap.sum())
        train_ids = train_ids[~overlap]
        raw_ids = np.asarray(raw["train/ev_ids/data"])
        train_index, present = lookup_ids(raw_ids, train_ids)
        missing_train = int((~present).sum())
        train_ids, train_index = train_ids[present], train_index[present]
        anchor_ids = np.asarray(pred_anchor["train/event_ids"])
        anchor_valid = np.asarray(pred_anchor["train/valid"], dtype=bool)
        anchor_point = np.asarray(pred_anchor["train/raw_particle_point_m"], dtype=np.float32)
        train_point, train_point_valid = map_truth_points(
            train_ids, anchor_ids, anchor_valid, anchor_point
        )
        order = np.argsort(train_index)
        arrays["train_ids"] = train_ids[order]
        arrays["train_raw_index"] = train_index[order]
        arrays["train_point_m"] = train_point[order]
        arrays["train_point_valid"] = train_point_valid[order]
        summary["train"] = {
            "rows": int(len(train_ids)),
            "unique_event_ids": int(len(np.unique(train_ids))),
            "class_counts": class_counts(train_ids),
            "point_valid": int(train_point_valid.sum()),
            "excluded_eval_overlap": overlap_count,
            "missing_raw_source": missing_train,
        }
        print(f"train index complete: {summary['train']}", flush=True)

        for split in ("val", "test"):
            pred_ids = np.asarray(predicted[f"{split}/ev_ids/data"])
            pred_anchor_ids = np.asarray(pred_anchor[f"{split}/event_ids"])
            if not np.array_equal(pred_ids, pred_anchor_ids):
                raise ValueError(f"Pred data and anchors do not align for {split}")
            select = masks[f"pred_{split}"] & np.asarray(pred_anchor[f"{split}/valid"], dtype=bool)
            ids, points = unique_first(
                pred_ids[select],
                np.asarray(pred_anchor[f"{split}/raw_particle_point_m"], dtype=np.float32)[select],
            )
            source_ids = np.asarray(raw[f"{split}/ev_ids/data"])
            index, present = lookup_ids(source_ids, ids)
            missing = int((~present).sum())
            ids, index, points = ids[present], index[present], points[present]
            if not np.isfinite(points).all():
                raise ValueError(f"Nonfinite selected {split} truth points")
            order = np.argsort(index)
            arrays[f"{split}_ids"] = ids[order]
            arrays[f"{split}_raw_index"] = index[order]
            arrays[f"{split}_point_m"] = points[order]
            arrays[f"{split}_point_valid"] = np.ones(len(ids), dtype=bool)
            summary[split] = {
                "rows": int(len(ids)),
                "class_counts": class_counts(ids),
                "missing_raw_source": missing,
                "train_id_overlap": int(np.isin(ids, train_ids).sum()),
            }
            print(f"{split} index complete: {summary[split]}", flush=True)

        (exp_ids,) = unique_first(np.asarray(pred_exp["test/ev_ids/data"]))
        exp_raw_ids = np.asarray(raw_exp["test/ev_ids/data"])
        exp_index, present = lookup_ids(exp_raw_ids, exp_ids)
        arrays["exp_ids"] = exp_ids[present]
        arrays["exp_raw_index"] = exp_index[present]
        summary["exp"] = {
            "rows": int(present.sum()),
            "missing_raw_source": int((~present).sum()),
        }
        print(f"experimental index complete: {summary['exp']}", flush=True)

        val_test_overlap = int(np.isin(arrays["val_ids"], arrays["test_ids"]).sum())
        summary["val_test_id_overlap"] = val_test_overlap
        if any(summary[s]["train_id_overlap"] for s in ("val", "test")) or val_test_overlap:
            raise ValueError("Cross-split event ID leakage")
        if missing_train or any(summary[s]["missing_raw_source"] for s in ("val", "test")):
            raise ValueError("Expected selected MC events are missing in the raw source")
        if summary["exp"]["missing_raw_source"]:
            raise ValueError("Expected experimental events are missing in the raw source")

        for split in ("train", "val", "test"):
            # One sequential read is much faster than >1M HDF5 point queries.
            starts = np.asarray(raw[f"{split}/ev_starts/data"], dtype=np.int64)
            indices = arrays[f"{split}_raw_index"]
            length = starts[indices + 1] - starts[indices]
            summary[split]["raw_hits_q50_q68_q95_q99"] = [
                float(np.quantile(length, q)) for q in (0.5, 0.68, 0.95, 0.99)
            ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-mc", type=Path, required=True)
    parser.add_argument("--clean-data", type=Path, required=True)
    parser.add_argument("--clean-anchors", type=Path, required=True)
    parser.add_argument("--pred-data", type=Path, required=True)
    parser.add_argument("--pred-anchors", type=Path, required=True)
    parser.add_argument("--gt-data", type=Path, required=True)
    parser.add_argument("--masks", type=Path, required=True)
    parser.add_argument("--raw-exp", type=Path, required=True)
    parser.add_argument("--pred-exp", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    build(args)


if __name__ == "__main__":
    main()
