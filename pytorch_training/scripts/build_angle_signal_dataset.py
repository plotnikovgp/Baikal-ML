#!/usr/bin/env python3
"""Create a compact angle-reconstruction HDF5 containing only truth signal hits.

The input is the normalized flat Baikal MC format used by pytorch_training.  Events
are retained when they have at least ``min_hits`` truth-signal hits on at least
``min_strings`` strings.  A particle-prefix filter can optionally be applied.

Stored hit features remain in the source normalization so the file is written in
one streaming pass.  Signal-only training statistics are also accumulated and
stored under ``signal_norm_param`` for optional on-the-fly renormalization.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np

EVENT_KEYS = ("ev_ids", "num_un_strings", "prime_prty")
HIT_KEYS = ("data", "labels", "channels", "t_res")
SPLITS = ("train", "val", "test")


def append(dataset: h5py.Dataset, values: np.ndarray) -> None:
    old_size = dataset.shape[0]
    dataset.resize(old_size + len(values), axis=0)
    dataset[old_size:] = values


def create_appendable(
    group: h5py.Group,
    name: str,
    source: h5py.Dataset,
    compression: str | None,
) -> h5py.Dataset:
    shape = (0,) + source.shape[1:]
    maxshape = (None,) + source.shape[1:]
    kwargs = {"chunks": True}
    if compression:
        kwargs["compression"] = compression
    return group.create_dataset(name, shape=shape, maxshape=maxshape, dtype=source.dtype, **kwargs)


def particle_mask(ev_ids: np.ndarray, particles: set[str]) -> np.ndarray:
    if not particles:
        return np.ones(len(ev_ids), dtype=bool)
    prefixes = ev_ids.astype("S5")
    mask = np.zeros(len(ev_ids), dtype=bool)
    prefix_map = {"muatm": b"muatm", "nuatm": b"nuatm", "nue2": b"nue2_"}
    for particle in particles:
        try:
            mask |= prefixes == prefix_map[particle]
        except KeyError as exc:
            raise ValueError(
                f"Unknown particle {particle!r}; choose from {sorted(prefix_map)}"
            ) from exc
    return mask


def iter_event_chunks(n_events: int, chunk_events: int) -> Iterable[tuple[int, int]]:
    for start in range(0, n_events, chunk_events):
        yield start, min(start + chunk_events, n_events)


def make_split(
    src: h5py.File,
    dst: h5py.File,
    split: str,
    particles: set[str],
    min_hits: int,
    min_strings: int,
    chunk_events: int,
    channels_per_string: int,
    compression: str | None,
    source_mean: np.ndarray,
    source_std: np.ndarray,
) -> tuple[dict[str, int], tuple[int, np.ndarray, np.ndarray]]:
    src_group = src[split]
    dst_group = dst.create_group(split)

    event_outputs: dict[str, h5py.Dataset] = {}
    for key in EVENT_KEYS:
        path = f"{key}/data"
        if path in src_group:
            event_outputs[key] = create_appendable(
                dst_group.require_group(key), "data", src_group[path], compression=None
            )

    hit_outputs: dict[str, h5py.Dataset] = {}
    for key in HIT_KEYS:
        path = f"{key}/data"
        if path in src_group:
            hit_outputs[key] = create_appendable(
                dst_group.require_group(key), "data", src_group[path], compression=compression
            )

    starts_out = dst_group.require_group("ev_starts").create_dataset(
        "data", shape=(1,), maxshape=(None,), dtype=np.int64, chunks=True
    )
    starts_out[0] = 0

    source_starts = src_group["ev_starts/data"]
    n_events = len(source_starts) - 1
    total_events = 0
    total_hits = 0
    seen_by_type = {"muatm": 0, "nuatm": 0, "nue2": 0}
    kept_by_type = {"muatm": 0, "nuatm": 0, "nue2": 0}
    stat_count = 0
    stat_sum = np.zeros_like(source_mean, dtype=np.float64)
    stat_sum2 = np.zeros_like(source_mean, dtype=np.float64)

    for chunk_idx, (event_start, event_end) in enumerate(
        iter_event_chunks(n_events, chunk_events), start=1
    ):
        starts = np.asarray(source_starts[event_start : event_end + 1], dtype=np.int64)
        hit_start, hit_end = int(starts[0]), int(starts[-1])
        rel_starts = starts - hit_start
        event_lengths = np.diff(rel_starts)

        ev_ids = np.asarray(src_group["ev_ids/data"][event_start:event_end])
        prefixes = ev_ids.astype("S5")
        for name, prefix in (("muatm", b"muatm"), ("nuatm", b"nuatm"), ("nue2", b"nue2_")):
            seen_by_type[name] += int(np.count_nonzero(prefixes == prefix))

        labels = np.asarray(src_group["labels/data"][hit_start:hit_end])
        channels = np.asarray(src_group["channels/data"][hit_start:hit_end])
        signal = labels != 0

        signal_counts = np.add.reduceat(signal.astype(np.int32), rel_starts[:-1])
        strings = channels // channels_per_string
        sentinel = np.iinfo(strings.dtype).max
        signal_min_string = np.minimum.reduceat(
            np.where(signal, strings, sentinel), rel_starts[:-1]
        )
        signal_max_string = np.maximum.reduceat(np.where(signal, strings, -1), rel_starts[:-1])
        if min_strings <= 1:
            enough_strings = signal_counts > 0
        else:
            enough_strings = signal_min_string < signal_max_string

        keep_events = (
            particle_mask(ev_ids, particles) & (signal_counts >= min_hits) & enough_strings
        )
        if not np.any(keep_events):
            continue

        for name, prefix in (("muatm", b"muatm"), ("nuatm", b"nuatm"), ("nue2", b"nue2_")):
            kept_by_type[name] += int(np.count_nonzero(keep_events & (prefixes == prefix)))

        keep_hits = signal & np.repeat(keep_events, event_lengths)
        selected_hits = int(np.count_nonzero(keep_hits))
        selected_event_lengths = signal_counts[keep_events].astype(np.int64)
        new_starts = total_hits + np.cumsum(selected_event_lengths, dtype=np.int64)
        append(starts_out, new_starts)

        for key, output in event_outputs.items():
            values = np.asarray(src_group[f"{key}/data"][event_start:event_end])
            if key == "num_un_strings":
                values = np.maximum(signal_max_string - signal_min_string, 0)[keep_events]
                # The exact number is not needed by training; preserve a conservative >=2 value.
                values = np.where(values > 0, 2, 1).astype(output.dtype, copy=False)
            else:
                values = values[keep_events]
            append(output, values)

        selected_data = None
        for key, output in hit_outputs.items():
            values = np.asarray(src_group[f"{key}/data"][hit_start:hit_end])
            values = values[keep_hits]
            append(output, values)
            if key == "data":
                selected_data = values.astype(np.float64, copy=False)

        if split == "train" and selected_data is not None:
            raw_data = selected_data * source_std + source_mean
            stat_count += len(raw_data)
            stat_sum += raw_data.sum(axis=0, dtype=np.float64)
            stat_sum2 += np.square(raw_data, dtype=np.float64).sum(axis=0, dtype=np.float64)

        total_events += int(np.count_nonzero(keep_events))
        total_hits += selected_hits

        if chunk_idx % 20 == 0 or event_end == n_events:
            print(
                f"{split}: scanned {event_end:,}/{n_events:,}; "
                f"kept {total_events:,} events, {total_hits:,} hits",
                flush=True,
            )

    summary = {
        "source_events": n_events,
        "kept_events": total_events,
        "kept_hits": total_hits,
        **{f"seen_{k}": v for k, v in seen_by_type.items()},
        **{f"kept_{k}": v for k, v in kept_by_type.items()},
    }
    return summary, (stat_count, stat_sum, stat_sum2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--particles", nargs="*", default=[])
    parser.add_argument("--min-hits", type=int, default=8)
    parser.add_argument("--min-strings", type=int, default=2)
    parser.add_argument("--channels-per-string", type=int, default=36)
    parser.add_argument("--chunk-events", type=int, default=20_000)
    parser.add_argument("--compression", choices=("none", "lzf", "gzip"), default="lzf")
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")

    compression = None if args.compression == "none" else args.compression
    particles = set(args.particles)
    summaries = {}

    with h5py.File(args.source, "r") as src, h5py.File(output, "w") as dst:
        source_mean = np.asarray(src["norm_param/mean"], dtype=np.float64)
        source_std = np.asarray(src["norm_param/std"], dtype=np.float64)
        src.copy("norm_param", dst)

        train_stats = None
        for split in SPLITS:
            summary, stats = make_split(
                src=src,
                dst=dst,
                split=split,
                particles=particles,
                min_hits=args.min_hits,
                min_strings=args.min_strings,
                chunk_events=args.chunk_events,
                channels_per_string=args.channels_per_string,
                compression=compression,
                source_mean=source_mean,
                source_std=source_std,
            )
            summaries[split] = summary
            if split == "train":
                train_stats = stats

        assert train_stats is not None
        count, sum_x, sum_x2 = train_stats
        signal_mean = sum_x / count
        signal_var = np.maximum(sum_x2 / count - signal_mean**2, 0.0)
        signal_std = np.sqrt(signal_var)
        signal_std[signal_std == 0] = 1.0
        signal_norm = dst.create_group("signal_norm_param")
        signal_norm.create_dataset("mean", data=signal_mean.astype(np.float32))
        signal_norm.create_dataset("std", data=signal_std.astype(np.float32))

        metadata = {
            "source": str(Path(args.source).resolve()),
            "particles": sorted(particles) if particles else ["muatm", "nuatm", "nue2"],
            "min_hits": args.min_hits,
            "min_strings": args.min_strings,
            "channels_per_string": args.channels_per_string,
            "stored_normalization": "norm_param",
            "recommended_normalization": "signal_norm_param",
            "summaries": summaries,
        }
        dst.attrs["metadata_json"] = json.dumps(metadata, sort_keys=True)

    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)
    print(f"Wrote {output} ({output.stat().st_size / 1e9:.3f} GB)", flush=True)


if __name__ == "__main__":
    main()
