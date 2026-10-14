#!/usr/bin/env python3
"""Rebase single-track truth lines onto a lower-threshold predicted-hit set.

Only events already present in the p>=0.70 *training* split are eligible.  The
source of the physical truth-line point is the audited p>=0.70 anchor file;
the new point target is its projection onto the centroid of p>=0.50 hits.
Evaluation IDs are excluded independently, even if a split was constructed
incorrectly upstream.  This file is a training-only anchor, not a new split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def direction_from_angles(prime: np.ndarray) -> np.ndarray:
    theta = np.deg2rad(prime[:, 0].astype(np.float64))
    phi = np.deg2rad(prime[:, 1].astype(np.float64))
    return np.column_stack(
        (np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta))
    )


def hit_centroids(h5: h5py.File, chunk_events: int) -> np.ndarray:
    group = h5["train"]
    starts = group["ev_starts/data"]
    count = len(starts) - 1
    mean = np.asarray(h5["norm_param/mean"], dtype=np.float64)
    std = np.asarray(h5["norm_param/std"], dtype=np.float64)
    output = np.empty((count, 3), dtype=np.float32)
    for begin in range(0, count, chunk_events):
        end = min(begin + chunk_events, count)
        chunk_starts = np.asarray(starts[begin : end + 1], dtype=np.int64)
        lengths = np.diff(chunk_starts)
        if np.any(lengths < 8):
            raise ValueError("An event has fewer than eight selected hits")
        hits = np.asarray(
            group["data/data"][int(chunk_starts[0]) : int(chunk_starts[-1]), 2:5],
            dtype=np.float64,
        )
        hits = hits * std[2:5] + mean[2:5]
        sums = np.add.reduceat(hits, chunk_starts[:-1] - chunk_starts[0], axis=0)
        output[begin:end] = (sums / lengths[:, None]).astype(np.float32)
    return output


def exact_membership(values: np.ndarray, sorted_unique: np.ndarray) -> np.ndarray:
    position = np.searchsorted(sorted_unique, values)
    return (position < len(sorted_unique)) & (
        sorted_unique[np.minimum(position, len(sorted_unique) - 1)] == values
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--reference-data", type=Path, required=True)
    parser.add_argument("--reference-anchors", type=Path, required=True)
    parser.add_argument("--gt-data", type=Path, required=True)
    parser.add_argument("--masks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-events", type=int, default=50_000)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with (
        h5py.File(args.data, "r") as data,
        h5py.File(args.reference_data, "r") as reference,
        h5py.File(args.reference_anchors, "r") as reference_anchors,
        h5py.File(args.gt_data, "r") as gt,
        np.load(args.masks) as masks,
    ):
        ids = np.asarray(data["train/ev_ids/data"])
        ref_ids = np.asarray(reference["train/ev_ids/data"])
        anchor_ids = np.asarray(reference_anchors["train/event_ids"])
        if not np.array_equal(ref_ids, anchor_ids):
            raise ValueError("Reference anchors are not row-aligned to reference data")
        valid_ref = np.asarray(reference_anchors["train/valid"], dtype=bool)
        ref_rows = np.flatnonzero(valid_ref)
        good_ids = ref_ids[ref_rows]
        order = np.argsort(good_ids, kind="stable")
        sorted_ids = good_ids[order]
        sorted_rows = ref_rows[order]
        unique_ids, first = np.unique(sorted_ids, return_index=True)
        unique_rows = sorted_rows[first]
        lookup = np.searchsorted(unique_ids, ids)
        matched = exact_membership(ids, unique_ids)
        reference_row = np.full(len(ids), -1, dtype=np.int64)
        reference_row[matched] = unique_rows[lookup[matched]]

        eval_ids = []
        for split in ("val", "test"):
            gt_keep = np.asarray(masks[f"gt_{split}"], dtype=bool)
            eval_ids.append(np.asarray(gt[f"{split}/ev_ids/data"])[gt_keep])
            pred_keep = np.asarray(masks[f"pred_{split}"], dtype=bool)
            eval_ids.append(np.asarray(reference[f"{split}/ev_ids/data"])[pred_keep])
        evaluation_ids = np.unique(np.concatenate(eval_ids))
        overlaps_evaluation = exact_membership(ids, evaluation_ids)

        points = np.full((len(ids), 3), np.nan, dtype=np.float32)
        reference_angles = np.full((len(ids), 2), np.nan, dtype=np.float32)
        if matched.any():
            rows = reference_row[matched]
            unique_rows_to_read, inverse = np.unique(rows, return_inverse=True)
            points[matched] = np.asarray(
                reference_anchors["train/raw_particle_point_m"][unique_rows_to_read],
                dtype=np.float32,
            )[inverse]
            reference_angles[matched] = np.asarray(
                reference["train/prime_prty/data"][unique_rows_to_read, :2],
                dtype=np.float32,
            )[inverse]
        prime = np.asarray(data["train/prime_prty/data"], dtype=np.float32)
        directions = direction_from_angles(prime)
        reference_directions = direction_from_angles(reference_angles)
        angular_difference = np.rad2deg(
            np.arccos(np.clip(np.sum(directions * reference_directions, axis=1), -1, 1))
        )
        direction_agrees = angular_difference < 0.01
        valid = matched & ~overlaps_evaluation & direction_agrees & np.isfinite(points).all(axis=1)
        centers = hit_centroids(data, args.chunk_events)
        delta = centers.astype(np.float64) - points.astype(np.float64)
        anchors = (
            points.astype(np.float64)
            + np.sum(delta * directions, axis=1, keepdims=True) * directions
        )
        offsets = anchors - centers
        anchors[~valid] = np.nan
        offsets[~valid] = np.nan
        points[~valid] = np.nan
        summary = {
            "rows": int(len(ids)),
            "matched_single_track_reference": int(matched.sum()),
            "overlaps_evaluation": int(overlaps_evaluation.sum()),
            "matched_overlaps_evaluation": int((matched & overlaps_evaluation).sum()),
            "matched_angle_disagrees": int((matched & ~direction_agrees).sum()),
            "eligible_train": int(valid.sum()),
            "unique_eligible_ids": int(len(np.unique(ids[valid]))),
            "valid_by_particle": {
                particle: int(np.char.startswith(ids[valid], particle.encode() + b"_").sum())
                for particle in ("muatm", "nuatm", "nue2")
            },
            "angle_difference_max_deg": float(np.nanmax(angular_difference[matched])),
            "anchor_transverse_max_m": float(
                np.nanmax(np.abs(np.sum(offsets[valid] * directions[valid], axis=1)))
            ),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(args.output, "w") as out:
            out.attrs["definition"] = (
                "single-track truth-line point closest to p>=0.50 hit centroid"
            )
            out.attrs["reference_data"] = str(args.reference_data)
            out.attrs["reference_anchors"] = str(args.reference_anchors)
            out.attrs["summary"] = json.dumps(summary)
            group = out.create_group("train")
            group.create_dataset("event_ids", data=ids, compression="lzf")
            group.create_dataset("valid", data=valid, compression="lzf")
            group.create_dataset("source_event_index", data=reference_row, compression="lzf")
            group.create_dataset("raw_particle_point_m", data=points, compression="lzf")
            group.create_dataset("hit_centroid_m", data=centers, compression="lzf")
            group.create_dataset(
                "anchor_point_m", data=anchors.astype(np.float32), compression="lzf"
            )
            group.create_dataset(
                "anchor_offset_m", data=offsets.astype(np.float32), compression="lzf"
            )
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
