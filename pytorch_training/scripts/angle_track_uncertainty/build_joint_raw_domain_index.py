#!/usr/bin/env python3
"""Map the fixed p>=0.70 exp/muatm-MC comparison cohort to raw-hit rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from build_joint_raw_index import lookup_ids, unique_first


def selected_metadata(group: h5py.Group, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    starts = np.asarray(group["ev_starts/data"], dtype=np.int64)
    hits = starts[indices + 1] - starts[indices]
    strings = np.asarray(group["num_un_strings/data"])[indices]
    return hits.astype(np.int32), strings.astype(np.int32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-mc", type=Path, required=True)
    parser.add_argument("--pred-mc", type=Path, required=True)
    parser.add_argument("--pred-anchors", type=Path, required=True)
    parser.add_argument("--raw-exp", type=Path, required=True)
    parser.add_argument("--pred-exp", type=Path, required=True)
    parser.add_argument("--joint-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    arrays: dict[str, np.ndarray] = {}
    summary = {}
    with (
        h5py.File(args.raw_mc, "r") as raw_mc,
        h5py.File(args.pred_mc, "r") as pred_mc,
        h5py.File(args.pred_anchors, "r") as anchors,
        h5py.File(args.raw_exp, "r") as raw_exp,
        h5py.File(args.pred_exp, "r") as pred_exp,
        np.load(args.joint_index) as joint,
    ):
        mc_ids_full = np.asarray(pred_mc["val/ev_ids/data"][:25_972])
        if not np.all(mc_ids_full.astype("S5") == b"muatm"):
            raise ValueError("The first 25,972 MC validation rows are not all muatm")
        mc_ids, pred_positions = unique_first(mc_ids_full, np.arange(len(mc_ids_full)))
        mc_raw_ids = np.asarray(raw_mc["val/ev_ids/data"])
        mc_index, present = lookup_ids(mc_raw_ids, mc_ids)
        if not present.all():
            raise ValueError(f"Missing {int((~present).sum())} MC raw events")
        hits, strings = selected_metadata(pred_mc["val"], pred_positions)
        anchor_valid = np.asarray(anchors["val/valid"], dtype=bool)[pred_positions]
        anchor_point = np.asarray(anchors["val/raw_particle_point_m"], dtype=np.float32)[
            pred_positions
        ]
        anchor_point[~anchor_valid] = 0.0
        order = np.argsort(mc_index)
        arrays.update(
            domain_mc_ids=mc_ids[order],
            domain_mc_raw_index=mc_index[order],
            domain_mc_point_m=anchor_point[order],
            domain_mc_point_valid=anchor_valid[order],
            domain_mc_hits=hits[order],
            domain_mc_strings=strings[order],
        )
        train_overlap = int(np.isin(mc_ids, joint["train_ids"]).sum())
        summary["domain_mc"] = {
            "rows": int(len(mc_ids)),
            "single_track": int(anchor_valid.sum()),
            "train_id_overlap": train_overlap,
        }

        exp_ids, exp_positions = unique_first(
            np.asarray(pred_exp["test/ev_ids/data"]),
            np.arange(len(pred_exp["test/ev_ids/data"])),
        )
        exp_raw_ids = np.asarray(raw_exp["test/ev_ids/data"])
        exp_index, present = lookup_ids(exp_raw_ids, exp_ids)
        if not present.all():
            raise ValueError(f"Missing {int((~present).sum())} experimental raw events")
        hits, strings = selected_metadata(pred_exp["test"], exp_positions)
        order = np.argsort(exp_index)
        arrays.update(
            exp_ids=exp_ids[order],
            exp_raw_index=exp_index[order],
            exp_hits=hits[order],
            exp_strings=strings[order],
        )
        summary["exp"] = {"rows": int(len(exp_ids))}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
