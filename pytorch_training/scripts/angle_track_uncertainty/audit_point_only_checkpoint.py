#!/usr/bin/env python3
"""Verify a point-only continuation cannot alter fixed-input directions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

POINT_KEYS = {"hit_head.5.weight", "hit_head.5.bias"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--initial-validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    # Both are trusted, local training checkpoints with argparse Path fields.
    initial = torch.load(args.initial, map_location="cpu", weights_only=False)
    candidate = torch.load(args.candidate, map_location="cpu", weights_only=False)
    before, after = initial["model"], candidate["model"]
    if before.keys() != after.keys():
        raise ValueError("Checkpoint state keys differ")
    changed = [key for key in before if not torch.equal(before[key], after[key])]
    if not changed or set(changed) != POINT_KEYS:
        raise ValueError(f"Unexpected changed parameters: {changed}")
    if not all(torch.equal(before[key][:2], after[key][:2]) for key in POINT_KEYS):
        raise ValueError("A direction-vote row changed")
    initial_val = json.loads(args.initial_validation.read_text())
    selected_val = candidate["metrics"]
    result = {
        "initial_checkpoint": str(args.initial),
        "candidate_checkpoint": str(args.candidate),
        "initial_sha256": sha256(args.initial),
        "candidate_sha256": sha256(args.candidate),
        "selected_step": int(selected_val["step"]),
        "changed_state_keys": changed,
        "all_direction_and_shared_parameters_exact": True,
        "initial_pred_val": initial_val["pred_val"],
        "candidate_pred_val": selected_val["pred_val"],
        "initial_gt_val_clean": initial_val["gt_val_clean"],
        "candidate_gt_val_clean": selected_val["gt_val_clean"],
        "selection_score_initial": initial_val["score"],
        "selection_score_candidate": selected_val["score"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "selected_step",
                    "changed_state_keys",
                    "candidate_sha256",
                    "selection_score_initial",
                    "selection_score_candidate",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
