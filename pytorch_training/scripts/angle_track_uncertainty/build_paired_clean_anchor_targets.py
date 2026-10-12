#!/usr/bin/env python3
"""Rebase validated pred-hit MC truth lines onto matched clean-hit centroids.

Only nuatm and uniquely tracked muatm are retained. The source of each raw
track point is the already-audited predicted-hit anchor file, not a fit to
the clean hits. This avoids an expensive second scan of the raw MC archive.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from build_all_track_anchor_targets import event_centers_and_directions


def map_valid_raw_points(
    compact_ids: np.ndarray,
    pred_anchors: h5py.Group,
) -> tuple[np.ndarray, np.ndarray]:
    pred_ids = np.asarray(pred_anchors["event_ids"])
    pred_valid = np.asarray(pred_anchors["valid"], dtype=bool)
    pred_points = np.asarray(pred_anchors["raw_particle_point_m"], dtype=np.float32)
    use = pred_valid & np.isfinite(pred_points).all(axis=1)
    valid_ids = pred_ids[use]
    valid_points = pred_points[use]
    unique_ids, first = np.unique(valid_ids, return_index=True)
    unique_points = valid_points[first]

    location = np.searchsorted(unique_ids, compact_ids)
    safe = np.minimum(location, len(unique_ids) - 1)
    present = (location < len(unique_ids)) & (unique_ids[safe] == compact_ids)
    requested_particle = np.char.startswith(
        compact_ids.astype("S6"), b"nuatm_"
    ) | np.char.startswith(compact_ids.astype("S6"), b"muatm_")
    selected = present & requested_particle
    raw = np.full((len(compact_ids), 3), np.nan, dtype=np.float32)
    raw[selected] = unique_points[safe[selected]]
    return raw, selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-data", type=Path, required=True)
    parser.add_argument("--pred-anchors", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-events", type=int, default=100_000)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    if args.chunk_events < 1:
        raise ValueError("chunk-events must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    summaries = {}
    with (
        h5py.File(args.clean_data, "r") as clean,
        h5py.File(args.pred_anchors, "r") as pred,
        h5py.File(args.output, "w") as target,
    ):
        target.attrs["clean_data"] = str(args.clean_data.resolve())
        target.attrs["pred_anchors"] = str(args.pred_anchors.resolve())
        target.attrs["definition"] = (
            "truth-line point closest to the clean-hit centroid, from matched "
            "valid predicted-hit raw MC track point; nuatm or single-track muatm only"
        )
        for split in ("train", "val", "test"):
            compact_ids = np.asarray(clean[f"{split}/ev_ids/data"])
            raw_points, valid = map_valid_raw_points(compact_ids, pred[split])
            centers, directions = event_centers_and_directions(
                clean, split, args.chunk_events, None
            )
            delta = centers.astype(np.float64) - raw_points.astype(np.float64)
            along = np.sum(delta * directions, axis=1, keepdims=True)
            offsets = raw_points.astype(np.float64) + along * directions - centers
            offsets[~valid] = np.nan

            group = target.create_group(split)
            group.create_dataset("event_ids", data=compact_ids, compression="lzf")
            group.create_dataset("valid", data=valid, compression="lzf")
            group.create_dataset("raw_particle_point_m", data=raw_points, compression="lzf")
            group.create_dataset("hit_centroid_m", data=centers, compression="lzf")
            group.create_dataset(
                "anchor_offset_m", data=offsets.astype(np.float32), compression="lzf"
            )
            nuatm = np.char.startswith(compact_ids.astype("S6"), b"nuatm_")
            muatm = np.char.startswith(compact_ids.astype("S6"), b"muatm_")
            summary = {
                "events": int(len(compact_ids)),
                "valid_total": int(valid.sum()),
                "nuatm_total": int(nuatm.sum()),
                "nuatm_valid": int((valid & nuatm).sum()),
                "muatm_total": int(muatm.sum()),
                "single_muatm_valid": int((valid & muatm).sum()),
            }
            summaries[split] = summary
            print(f"{split} {json.dumps(summary, sort_keys=True)}", flush=True)
        target.attrs["summary_json"] = json.dumps(summaries, sort_keys=True)
    args.output.with_suffix(".json").write_text(
        json.dumps(summaries, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
