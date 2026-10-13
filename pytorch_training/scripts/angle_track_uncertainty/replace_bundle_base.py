#!/usr/bin/env python3
"""Build a training-only bundle with another direction/point base model."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from bundle_universal_model import build_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    bundle = torch.load(args.bundle, map_location="cpu", weights_only=True)
    checkpoint = torch.load(args.base_checkpoint, map_location="cpu", weights_only=False)
    original = build_model(bundle["model_config"])
    original.load_state_dict(bundle["state_dict"], strict=True)
    original_keys = original.base.state_dict().keys()
    replacement = checkpoint["model"]
    if original_keys != replacement.keys():
        raise ValueError("Replacement base architecture differs from bundle")
    for key, value in replacement.items():
        if bundle["state_dict"][f"base.{key}"].shape != value.shape:
            raise ValueError(f"Shape differs for {key}")
        bundle["state_dict"][f"base.{key}"] = value.clone()
    bundle["training_only_notice"] = (
        "Direction/point base replaced; uncertainty heads are stale and must not be "
        "used or distributed without retraining and calibration."
    )
    updated = build_model(bundle["model_config"])
    updated.load_state_dict(bundle["state_dict"], strict=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, args.output)
    print(f"Wrote training-only bundle: {args.output}")


if __name__ == "__main__":
    main()
