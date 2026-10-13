#!/usr/bin/env python3
"""Resolve compact 2019 MC event IDs to cluster-local truth-line points."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from build_all_track_anchor_targets import raw_particle_points


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--particle", choices=("nue2", "nuatm", "muatm"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with h5py.File(args.raw) as compact:
        ids = np.asarray(compact["test/ev_ids/data"])
        prime = np.asarray(compact["test/prime_prty/data"])
    _, inverse, counts = np.unique(ids, return_inverse=True, return_counts=True)
    unique_id = counts[inverse] == 1
    with h5py.File(args.source) as source:
        points, parsed, single, indices = raw_particle_points(source, f"{args.particle}_2019", ids)
    # One generated event may be simulated in multiple clusters with the same
    # ID but a different cluster-local coordinate origin. The generic source
    # lookup cannot disambiguate those rare rows, so exclude every duplicate.
    valid = parsed & single & unique_id & np.isfinite(points).all(axis=1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        event_ids=ids,
        prime=prime,
        point_m=points,
        valid=valid,
        source_index=indices,
    )
    summary = {
        "source": str(args.source),
        "raw": str(args.raw),
        "output": str(args.output),
        "particle": args.particle,
        "events": len(ids),
        "parsed": int(parsed.sum()),
        "single": int(single.sum()),
        "valid": int(valid.sum()),
        "duplicate_id_rows_excluded": int((~unique_id).sum()),
    }
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
