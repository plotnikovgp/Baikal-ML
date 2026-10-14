#!/usr/bin/env python3
"""Copy exact paired reco events from raw 2020 HDF5, before ``reco_mask``.

No hit label, reconstructed-hit mask, or reconstruction-quality cut is used
here.  The output is a compact, physically unnormalized ``test`` split for
the frozen signal/noise classifier.  Event IDs must match the requested list
one-to-one; missing or duplicate IDs are fatal.
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


def make_appendable(group: h5py.Group, name: str, src: h5py.Dataset) -> h5py.Dataset:
    return group.require_group(name).create_dataset(
        "data",
        shape=(0,) + src.shape[1:],
        maxshape=(None,) + src.shape[1:],
        dtype=src.dtype,
        chunks=True,
        compression="lzf",
    )


def append(dst: h5py.Dataset, values: np.ndarray) -> None:
    if len(values) == 0:
        return
    old = len(dst)
    dst.resize(old + len(values), axis=0)
    dst[old:] = values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--filtered", type=Path, required=True)
    parser.add_argument("--event-ids", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    requested = np.load(args.event_ids).astype("S45")
    if requested.ndim != 1 or len(np.unique(requested)) != len(requested):
        raise ValueError("Requested event IDs must be a unique 1D vector")
    sorted_requested = np.sort(requested)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with (
        h5py.File(args.source, "r") as source,
        h5py.File(args.filtered, "r") as filtered,
        h5py.File(args.output, "x") as output,
    ):
        src = source["nue2"]
        filt = filtered["nue2"]
        parts = sorted(src["ev_ids"])
        norm = output.create_group("norm_param")
        norm.create_dataset("mean", data=np.zeros(5, dtype=np.float32))
        norm.create_dataset("std", data=np.ones(5, dtype=np.float32))
        out = output.create_group("test")
        first = parts[0]
        datasets = {}
        for key in EVENT_KEYS:
            datasets[key] = make_appendable(out, key, src[key][first]["data"])
        for key in RAW_EVENT_KEYS + HIT_KEYS:
            datasets[key] = make_appendable(out, key, src["raw"][key][first]["data"])
        starts_out = out.require_group("ev_starts").create_dataset(
            "data", shape=(1,), maxshape=(None,), chunks=True, dtype=np.int64
        )
        starts_out[0] = 0
        found = np.zeros(len(sorted_requested), dtype=bool)
        total_events = total_hits = 0
        for part_number, part in enumerate(parts, 1):
            part_ids = np.asarray(src["ev_ids"][part]["data"])
            filtered_ids = np.asarray(filt["ev_ids"][part]["data"])
            position = np.searchsorted(sorted_requested, filtered_ids)
            matched_filtered = (position < len(sorted_requested)) & (
                sorted_requested[np.minimum(position, len(sorted_requested) - 1)] == filtered_ids
            )
            filtered_rows = np.flatnonzero(matched_filtered)
            if len(filtered_rows) == 0:
                continue
            if found[position[matched_filtered]].any():
                raise ValueError(f"Duplicate requested ID encountered in {part}")
            found[position[matched_filtered]] = True
            candidate_rows: dict[bytes, list[int]] = {}
            for i, event_id in enumerate(part_ids):
                candidate_rows.setdefault(bytes(event_id), []).append(i)
            starts = np.asarray(src["raw/ev_starts"][part]["data"], dtype=np.int64)
            raw_data = np.asarray(src["raw/data"][part]["data"])
            raw_channels = np.asarray(src["raw/channels"][part]["data"])
            reco_mask = np.asarray(src["reco_mask"][part]["data"], dtype=bool)
            filtered_starts = np.asarray(filt["ev_starts"][part]["data"], dtype=np.int64)
            filtered_data = np.asarray(filt["data"][part]["data"])
            filtered_channels = np.asarray(filt["channels"][part]["data"])
            chosen = []
            for filtered_row in filtered_rows:
                event_id = bytes(filtered_ids[filtered_row])
                fa, fb = filtered_starts[filtered_row : filtered_row + 2]
                matches = []
                for candidate in candidate_rows.get(event_id, []):
                    a, b = starts[candidate : candidate + 2]
                    reco_selected = reco_mask[a:b]
                    if np.array_equal(
                        raw_data[a:b][reco_selected], filtered_data[fa:fb]
                    ) and np.array_equal(
                        raw_channels[a:b][reco_selected], filtered_channels[fa:fb]
                    ):
                        matches.append(candidate)
                if len(matches) != 1:
                    raise ValueError(
                        f"Expected exactly one raw cluster for ID {event_id!r} in {part}; "
                        f"found {matches}"
                    )
                chosen.append(matches[0])
            chosen = np.asarray(chosen, dtype=np.int64)
            lengths = np.diff(starts)[chosen]
            if np.any(lengths <= 0):
                raise ValueError(f"Empty raw event in {part}")
            for key in EVENT_KEYS:
                append(datasets[key], np.asarray(src[key][part]["data"])[chosen])
            for key in RAW_EVENT_KEYS:
                append(datasets[key], np.asarray(src["raw"][key][part]["data"])[chosen])
            for key in HIT_KEYS:
                values = np.asarray(src["raw"][key][part]["data"])
                append(
                    datasets[key],
                    np.concatenate([values[starts[i] : starts[i + 1]] for i in chosen]),
                )
            append(starts_out, total_hits + np.cumsum(lengths))
            total_events += len(chosen)
            total_hits += int(lengths.sum())
            if part_number % 25 == 0:
                print(
                    f"parts={part_number}/{len(parts)} events={total_events} hits={total_hits}",
                    flush=True,
                )
        if not found.all():
            raise ValueError(f"Missing {int((~found).sum())} requested IDs")
        if total_events != len(requested) or int(starts_out[-1]) != total_hits:
            raise AssertionError("Output offsets and requested IDs disagree")
        if len(np.unique(out["ev_ids/data"][:])) != total_events:
            raise AssertionError("Duplicate IDs in output")
        metadata = {
            "source": str(args.source),
            "filtered_reference": str(args.filtered),
            "event_ids": str(args.event_ids),
            "events": total_events,
            "hits": total_hits,
            "selection": "exact paired IDs and cluster matched to filtered reco hits; all raw hits without reco_mask or truth-label hit filter",
        }
        output.attrs["metadata_json"] = json.dumps(metadata, sort_keys=True)
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
