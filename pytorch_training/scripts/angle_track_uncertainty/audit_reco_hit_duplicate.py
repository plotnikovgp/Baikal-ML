#!/usr/bin/env python3
"""Check exact selected-hit duplicates between high-quality reco test and GT train."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from analyze_reco_central_gap import root_ten_cut
from audit_primary_overlap import primary_keys


def fingerprint(group: h5py.Group, index: int) -> bytes:
    starts = group["ev_starts/data"]
    low, high = int(starts[index]), int(starts[index + 1])
    values = np.asarray(group["data/data"][low:high], dtype=np.float32)
    order = np.lexsort(tuple(values[:, column] for column in range(values.shape[1] - 1, -1, -1)))
    return hashlib.blake2b(values[order].tobytes(), digest_size=16).digest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired", type=Path, required=True)
    parser.add_argument("--root-fields", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--fair", type=Path, required=True)
    parser.add_argument("--gt-train", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-count", type=int)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")

    with np.load(args.paired) as paired:
        event_ids = paired["event_ids"].astype(str)
        fair_indices = paired["fair_index"].astype(np.int64)
    root = (
        pd.read_csv(args.root_fields)
        .set_index("event_id", verify_integrity=True)
        .reindex(event_ids)
    )
    available = pd.Index(event_ids).isin(
        pd.read_csv(args.gate, usecols=["event_id"]).event_id.astype(str).to_numpy()
    )
    high_quality = (
        root_ten_cut(root)
        & available
        & (root.fNHits.to_numpy() >= 20)
        & (root.fNTriplets.to_numpy() >= 12)
    )
    selected_fair = fair_indices[high_quality]
    if args.expected_count is not None and len(selected_fair) != args.expected_count:
        raise ValueError(
            f"Expected {args.expected_count} high-quality reco events, got {len(selected_fair)}"
        )

    with h5py.File(args.fair) as fair, h5py.File(args.gt_train) as train:
        if not np.array_equal(fair["norm_param/mean"][:], train["norm_param/mean"][:]):
            raise ValueError("Normalization means differ")
        if not np.array_equal(fair["norm_param/std"][:], train["norm_param/std"][:]):
            raise ValueError("Normalization scales differ")
        selected_primaries = np.unique(
            primary_keys(np.asarray(fair["test/prime_prty/data"])[selected_fair])
        )
        train_primaries = primary_keys(np.asarray(train["train/prime_prty/data"]))
        matching_train = np.flatnonzero(np.isin(train_primaries, selected_primaries))
        exact_id_overlap = np.isin(
            np.asarray(event_ids[high_quality], dtype="S64"),
            np.unique(np.asarray(train["train/ev_ids/data"], dtype="S64")),
        )
        print(
            f"high_quality_reco={len(selected_fair)} matching_primary_train={len(matching_train)}",
            flush=True,
        )
        train_fingerprints = set()
        for offset, index in enumerate(matching_train, 1):
            train_fingerprints.add(fingerprint(train["train"], int(index)))
            if offset % 2000 == 0:
                print(f"fingerprinted_train={offset}/{len(matching_train)}", flush=True)
        overlaps = 0
        for offset, index in enumerate(selected_fair, 1):
            overlaps += fingerprint(fair["test"], int(index)) in train_fingerprints
            if offset % 2000 == 0:
                print(f"fingerprinted_reco={offset}/{len(selected_fair)}", flush=True)
    report = {
        "selection": "ROOT-observable preliminary 10/13 cuts, fNHits>=20, fNTriplets>=12",
        "reco_high_quality_events": int(len(selected_fair)),
        "unique_reco_primaries": int(len(selected_primaries)),
        "matching_primary_gt_train_events": int(len(matching_train)),
        "exact_matching_event_ids": int(exact_id_overlap.sum()),
        "unique_train_hit_fingerprints": int(len(train_fingerprints)),
        "exact_matching_hit_sets": int(overlaps),
        "fingerprint": "BLAKE2b-128 of lexicographically sorted normalized float32 hit rows",
        "caveat": (
            "Exact hit equality only. Changed normalization, filtering or hit values can evade this test; "
            "shared primary theta/phi/energy still means this is not a new-primary validation set."
        ),
        "inputs": {
            name: str(getattr(args, name))
            for name in ("paired", "root_fields", "gate", "fair", "gt_train")
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
