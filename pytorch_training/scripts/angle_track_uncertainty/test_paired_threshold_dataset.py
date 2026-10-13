"""Regression checks for paired p>=0.70/p>=0.90 training views."""

from __future__ import annotations

import h5py
import numpy as np
import torch
from train_mixed_universal import AdaptedTrackDataset


def test_paired_views_keep_original_line_and_cut(tmp_path):
    data_path = tmp_path / "events.h5"
    anchor_path = tmp_path / "anchors.h5"
    hits = np.zeros((20, 5), dtype=np.float32)
    hits[:, 0] = 5.0
    hits[:, 2] = np.tile([0.0] * 5 + [10.0] * 5, 2)
    hits[:, 4] = np.tile(np.arange(10), 2)
    probabilities = np.full(20, 0.95, dtype=np.float32)
    probabilities[0] = 0.72  # Valid nine-hit high-threshold view.
    probabilities[10:13] = 0.72  # Seven hits: must fall back to the original.
    ids = np.asarray([b"muatm_0_0", b"muatm_0_1"])
    with h5py.File(data_path, "w") as source:
        norm = source.create_group("norm_param")
        norm.create_dataset("mean", data=np.zeros(5, dtype=np.float32))
        norm.create_dataset("std", data=np.ones(5, dtype=np.float32))
        group = source.create_group("train")
        for name, values in (
            ("ev_starts", np.asarray([0, 10, 20])),
            ("data", hits),
            ("prime_prty", np.asarray([[0.0, 0.0], [0.0, 0.0]])),
            ("ev_ids", ids),
            ("signal_probability", probabilities),
        ):
            group.create_group(name).create_dataset("data", data=values)
    with h5py.File(anchor_path, "w") as source:
        group = source.create_group("train")
        group.create_dataset("event_ids", data=ids)
        group.create_dataset("valid", data=[True, True])
        group.create_dataset("anchor_offset_m", data=np.asarray([[1, 2, 0], [1, 2, 0]]))
        group.create_dataset("hit_centroid_m", data=np.asarray([[5, 0, 4.5], [5, 0, 4.5]]))

    data = AdaptedTrackDataset(
        str(data_path),
        "train",
        2,
        128,
        False,
        None,
        False,
        anchor_path=str(anchor_path),
        keep=None,
        expected_mean=np.zeros(5),
        expected_std=np.ones(5),
        single_muatm_only=True,
        pred_thinning_prob=1.0,
        pred_threshold_min=0.90,
        pred_threshold_max=0.90,
        return_unthinned=True,
    )
    x, direction, mask, _particle, point, valid, original_x, original_mask, original_point = data[0]
    assert bool(valid.all())
    assert len(mask) == 2 and mask.sum(1).tolist() == [9, 10]
    assert original_mask.sum(1).tolist() == [10, 10]
    assert torch.equal(original_x[1], x[1])
    original_center = original_x[0, original_mask[0], 2:5].mean(0)
    new_center = x[0, mask[0], 2:5].mean(0)
    difference = (original_center + original_point[0]) - (new_center + point[0])
    transverse = difference - direction[0] * (difference * direction[0]).sum()
    assert torch.allclose(transverse, torch.zeros(3), atol=1e-6)
