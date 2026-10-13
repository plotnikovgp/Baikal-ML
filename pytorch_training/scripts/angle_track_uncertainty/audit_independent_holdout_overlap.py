#!/usr/bin/env python3
"""Verify a new raw 2019 holdout is disjoint from previous extractions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def signatures(path: Path) -> tuple[np.ndarray, np.ndarray, set[str]]:
    with h5py.File(path) as h5:
        event_ids = np.asarray(h5["test/ev_ids/data"]).astype("S128")
        prime = np.ascontiguousarray(h5["test/prime_prty/data"][:, :3], dtype=np.float32)
        parts = set(json.loads(h5.attrs["metadata_json"])["selected_parts"])
    primary = prime.view("V12").reshape(-1)
    return event_ids, primary, parts


def overlap(candidate: Path, references: list[Path]) -> dict:
    candidate_ids, candidate_primary, candidate_parts = signatures(candidate)
    result = {
        "candidate": str(candidate),
        "candidate_events": len(candidate_ids),
        "candidate_source_parts": len(candidate_parts),
        "references": {},
    }
    for path in references:
        ids, primary, parts = signatures(path)
        result["references"][str(path)] = {
            "source_part_overlap": len(candidate_parts & parts),
            "event_id_overlap": int(np.isin(candidate_ids, np.unique(ids)).sum()),
            "generated_primary_overlap": int(np.isin(candidate_primary, np.unique(primary)).sum()),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = overlap(args.candidate, args.reference)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
