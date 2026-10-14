#!/usr/bin/env python3
"""Raise the frozen-classifier threshold in an already selected-hit HDF5.

The source threshold must be no larger than the requested threshold. No
classifier is rerun; events failing the post-threshold 2-string/8-hit cut
are dropped. This preserves original hit features and normalization.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

HIT_FIELDS = ("channels", "data", "labels", "signal_probability", "t_res")
EVENT_FIELDS = ("cluster_ids", "ev_ids", "num_un_strings", "prime_prty")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not 0 < args.threshold <= 1:
        raise ValueError("threshold must be in (0, 1]")

    with h5py.File(args.source) as source:
        group = source[args.split]
        starts = np.asarray(group["ev_starts/data"], dtype=np.int64)
        probabilities = np.asarray(group["signal_probability/data"], dtype=np.float32)
        channels = np.asarray(group["channels/data"], dtype=np.int32)
        if len(starts) < 2 or len(probabilities) != starts[-1]:
            raise ValueError("Invalid source HDF5 starts/probabilities")
        if probabilities.min() < 0 or probabilities.max() > 1:
            raise ValueError("Source contains invalid classifier scores")

        event_indices = []
        hit_indices = []
        new_starts = [0]
        string_counts = []
        for event_index, (begin, end) in enumerate(zip(starts[:-1], starts[1:])):
            selected = np.flatnonzero(probabilities[begin:end] >= args.threshold) + begin
            if len(selected) < 8:
                continue
            n_strings = len(np.unique(channels[selected] // 36))
            if n_strings < 2:
                continue
            event_indices.append(event_index)
            hit_indices.append(selected)
            new_starts.append(new_starts[-1] + len(selected))
            string_counts.append(n_strings)

        event_indices = np.asarray(event_indices, dtype=np.int64)
        hit_indices = (
            np.concatenate(hit_indices).astype(np.int64)
            if hit_indices
            else np.empty(0, dtype=np.int64)
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        partial = args.output.with_suffix(args.output.suffix + ".partial")
        if partial.exists():
            raise FileExistsError(partial)
        with h5py.File(partial, "x") as output:
            source.copy("norm_param", output)
            target = output.create_group(args.split)
            for name in HIT_FIELDS:
                target.create_dataset(
                    f"{name}/data", data=np.asarray(group[f"{name}/data"])[hit_indices]
                )
            for name in EVENT_FIELDS:
                if name == "num_un_strings":
                    values = np.asarray(string_counts, dtype=np.int32)
                else:
                    values = np.asarray(group[f"{name}/data"])[event_indices]
                target.create_dataset(f"{name}/data", data=values)
            target.create_dataset("ev_starts/data", data=np.asarray(new_starts, dtype=np.int64))
        partial.replace(args.output)

    print(
        json.dumps(
            {
                "source": str(args.source),
                "output": str(args.output),
                "threshold": args.threshold,
                "source_events": len(starts) - 1,
                "retained_events": len(event_indices),
                "retained_hits": len(hit_indices),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
