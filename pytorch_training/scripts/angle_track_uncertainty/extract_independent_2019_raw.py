#!/usr/bin/env python3
"""Extract a deterministic, physically unnormalized 2019 nue2 holdout.

The original merged HDF5 is part-sharded and very large.  This script copies
contiguous events from evenly spaced source parts without using any truth hit
labels to select events.  The downstream frozen hit classifier applies the
same p>=0.70, 2-string/8-hit cut as for the 2020 evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

EVENT_KEYS = ("ev_ids", "prime_prty")
RAW_EVENT_KEYS = ("cluster_ids", "num_un_strings")
HIT_KEYS = ("data", "channels", "labels", "t_res")


def append(dst: h5py.Dataset, values: np.ndarray) -> None:
    old = len(dst)
    dst.resize(old + len(values), axis=0)
    dst[old:] = values


def make_dataset(group: h5py.Group, name: str, source: h5py.Dataset) -> h5py.Dataset:
    shape = (0,) + source.shape[1:]
    return group.require_group(name).create_dataset(
        "data",
        shape=shape,
        maxshape=(None,) + source.shape[1:],
        chunks=True,
        compression="lzf",
        dtype=source.dtype,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parts", type=int, default=20)
    parser.add_argument("--events-per-part", type=int, default=5000)
    args = parser.parse_args()
    if args.parts < 1 or args.events_per_part < 1:
        raise ValueError("parts and events-per-part must be positive")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.source, "r") as source, h5py.File(args.output, "x") as output:
        src = source["nue2_2019"]
        parts = sorted(src["ev_ids"])
        part_indices = np.unique(
            np.linspace(0, len(parts) - 1, min(args.parts, len(parts)), dtype=int)
        )
        selected_parts = [parts[i] for i in part_indices]
        norm = output.create_group("norm_param")
        norm.create_dataset("mean", data=np.zeros(5, dtype=np.float32))
        norm.create_dataset("std", data=np.ones(5, dtype=np.float32))
        dst = output.create_group("test")
        first_part = selected_parts[0]
        datasets = {}
        for key in EVENT_KEYS:
            datasets[key] = make_dataset(dst, key, src[key][first_part]["data"])
        for key in RAW_EVENT_KEYS + HIT_KEYS:
            datasets[key] = make_dataset(dst, key, src["raw"][key][first_part]["data"])
        starts_out = dst.require_group("ev_starts").create_dataset(
            "data", shape=(1,), maxshape=(None,), chunks=True, dtype=np.int64
        )
        starts_out[0] = 0
        events = hits = 0
        for part in selected_parts:
            n = min(args.events_per_part, len(src["ev_ids"][part]["data"]))
            starts = np.asarray(src["raw/ev_starts"][part]["data"][: n + 1], dtype=np.int64)
            hit_count = int(starts[-1])
            for key in EVENT_KEYS:
                append(datasets[key], src[key][part]["data"][:n])
            for key in RAW_EVENT_KEYS:
                append(datasets[key], src["raw"][key][part]["data"][:n])
            for key in HIT_KEYS:
                append(datasets[key], src["raw"][key][part]["data"][:hit_count])
            append(starts_out, starts[1:] + hits)
            events += n
            hits += hit_count
            print(f"{part}: events={events:,} hits={hits:,}", flush=True)
        assert len(starts_out) == events + 1
        assert int(starts_out[-1]) == hits
        assert all(len(datasets[key]) == events for key in EVENT_KEYS + RAW_EVENT_KEYS)
        assert all(len(datasets[key]) == hits for key in HIT_KEYS)
        manifest = {
            "source": str(args.source),
            "selected_parts": selected_parts,
            "events_per_part": args.events_per_part,
            "events": events,
            "hits": hits,
            "selection": "evenly spaced source parts; first N events per part; no truth-hit selection",
        }
        output.attrs["metadata_json"] = json.dumps(manifest, sort_keys=True)
    args.output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {args.output} ({args.output.stat().st_size / 1e9:.3f} GB)", flush=True)


if __name__ == "__main__":
    main()
