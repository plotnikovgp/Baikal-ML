#!/usr/bin/env python3
"""Export the two tiny normalization files needed before released inference."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch


def read_norm(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path) as h5:
        mean = np.asarray(h5["norm_param/mean"], dtype=np.float32)
        std = np.asarray(h5["norm_param/std"], dtype=np.float32)
    if (
        mean.shape != (5,)
        or std.shape != (5,)
        or not np.isfinite(mean).all()
        or not np.all(std > 0)
    ):
        raise ValueError(f"Expected five finite mean/positive std values in {path}")
    return mean, std


def write_norm(path: Path, values: tuple[np.ndarray, np.ndarray]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        group = h5.create_group("norm_param")
        group.create_dataset("mean", data=values[0])
        group.create_dataset("std", data=values[1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classifier-norm-source", type=Path, required=True)
    parser.add_argument("--angle-norm-source", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--classifier-output", type=Path, required=True)
    parser.add_argument("--angle-output", type=Path, required=True)
    args = parser.parse_args()
    classifier = read_norm(args.classifier_norm_source)
    angle = read_norm(args.angle_norm_source)
    bundle = torch.load(args.bundle, map_location="cpu", weights_only=True)
    prep = bundle["preprocessing"]
    if not np.array_equal(angle[0], np.asarray(prep["input_norm_mean"], dtype=np.float32)):
        raise ValueError("Angle mean does not match bundled preprocessing")
    if not np.array_equal(angle[1], np.asarray(prep["input_norm_std"], dtype=np.float32)):
        raise ValueError("Angle std does not match bundled preprocessing")
    if args.classifier_output == args.angle_output:
        raise ValueError("Normalization outputs must be distinct")
    write_norm(args.classifier_output, classifier)
    write_norm(args.angle_output, angle)
    print(
        f"Exported classifier and angle normalization to {args.classifier_output} and {args.angle_output}"
    )


if __name__ == "__main__":
    main()
