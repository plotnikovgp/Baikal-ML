"""Small end-to-end check of low-threshold truth-line rebasing and split safety."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np


def add_ids(file: h5py.File, split: str, values: list[bytes]) -> None:
    group = file.require_group(split)
    group.require_group("ev_ids").create_dataset("data", data=np.asarray(values, dtype="S32"))


def test_rebased_anchor_excludes_eval_ids(tmp_path: Path) -> None:
    data_path = tmp_path / "p050.h5"
    reference_path = tmp_path / "p070.h5"
    anchors_path = tmp_path / "p070_anchors.h5"
    gt_path = tmp_path / "gt.h5"
    masks_path = tmp_path / "masks.npz"
    output_path = tmp_path / "p050_anchors.h5"
    match_id = b"nue2_0_1"
    eval_id = b"muatm_0_2"
    unmatched_id = b"nuatm_0_3"

    with h5py.File(data_path, "w") as data:
        add_ids(data, "train", [match_id, eval_id, unmatched_id])
        norm = data.create_group("norm_param")
        norm.create_dataset("mean", data=np.zeros(5, dtype=np.float32))
        norm.create_dataset("std", data=np.ones(5, dtype=np.float32))
        train = data["train"]
        train.require_group("ev_starts").create_dataset("data", data=[0, 8, 16, 24])
        hits = np.zeros((24, 5), dtype=np.float32)
        hits[:8, 2:5] = [10.0, 5.0, 2.0]
        hits[8:16, 2:5] = [1.0, 1.0, 1.0]
        hits[16:, 2:5] = [2.0, 2.0, 2.0]
        train.require_group("data").create_dataset("data", data=hits)
        train.require_group("prime_prty").create_dataset(
            "data", data=np.asarray([[90.0, 0.0], [90.0, 0.0], [90.0, 0.0]])
        )
    with h5py.File(reference_path, "w") as reference:
        add_ids(reference, "train", [match_id, eval_id])
        add_ids(reference, "val", [])
        add_ids(reference, "test", [])
        reference["train"].require_group("prime_prty").create_dataset(
            "data", data=np.asarray([[90.0, 0.0], [90.0, 0.0]])
        )
    with h5py.File(anchors_path, "w") as anchors:
        train = anchors.create_group("train")
        train.create_dataset("event_ids", data=np.asarray([match_id, eval_id], dtype="S32"))
        train.create_dataset("valid", data=[True, True])
        train.create_dataset("raw_particle_point_m", data=np.zeros((2, 3), dtype=np.float32))
    with h5py.File(gt_path, "w") as gt:
        add_ids(gt, "val", [eval_id])
        add_ids(gt, "test", [])
    np.savez(
        masks_path,
        gt_val=np.asarray([True]),
        gt_test=np.asarray([], dtype=bool),
        pred_val=np.asarray([], dtype=bool),
        pred_test=np.asarray([], dtype=bool),
    )

    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("build_p050_anchor_from_p070.py")),
            "--data",
            str(data_path),
            "--reference-data",
            str(reference_path),
            "--reference-anchors",
            str(anchors_path),
            "--gt-data",
            str(gt_path),
            "--masks",
            str(masks_path),
            "--output",
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    with h5py.File(output_path) as anchors:
        np.testing.assert_array_equal(anchors["train/valid"][:], [True, False, False])
        np.testing.assert_allclose(
            anchors["train/anchor_offset_m"][0], [0.0, -5.0, -2.0], atol=1e-5
        )
        assert np.isnan(anchors["train/anchor_offset_m"][1:]).all()
