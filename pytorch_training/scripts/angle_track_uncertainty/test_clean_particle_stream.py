#!/usr/bin/env python3
"""Clean-particle angle batches keep only the requested class and units."""

from pathlib import Path

import h5py
import numpy as np
import torch
from train_mixed_universal import CleanParticleAngleDataset


def test_clean_particle_stream_filters_and_renormalizes(tmp_path: Path) -> None:
    path = tmp_path / "small_all_gt.h5"
    stored_mean = np.array([1.0, 10.0, 2.0, 3.0, 4.0], dtype=np.float32)
    stored_std = np.array([2.0, 20.0, 5.0, 6.0, 7.0], dtype=np.float32)
    expected_mean = np.zeros(5, dtype=np.float32)
    expected_std = np.full(5, 2.0, dtype=np.float32)
    stored = np.arange(30, dtype=np.float32).reshape(6, 5) / 10.0
    with h5py.File(path, "w") as h5:
        norm = h5.create_group("norm_param")
        norm.create_dataset("mean", data=stored_mean)
        norm.create_dataset("std", data=stored_std)
        train = h5.create_group("train")
        train.create_dataset("ev_starts/data", data=np.array([0, 2, 3, 6]))
        train.create_dataset("data/data", data=stored)
        train.create_dataset("prime_prty/data", data=np.zeros((3, 2)))
        train.create_dataset("ev_ids/data", data=np.array([b"nuatm_0", b"nue2_0", b"nuatm_1"]))

    dataset = CleanParticleAngleDataset(
        str(path),
        "train",
        3,
        4,
        False,
        None,
        False,
        expected_mean=expected_mean,
        expected_std=expected_std,
        particle_code=1,
    )
    x, direction, mask, particle = dataset[0]
    assert x.shape == (2, 3, 5)
    assert mask.tolist() == [[True, True, False], [True, True, True]]
    assert particle.tolist() == [1, 1]
    expected = (stored * stored_std + stored_mean - expected_mean) / expected_std
    torch.testing.assert_close(x[0, :2], torch.from_numpy(expected[:2]))
    torch.testing.assert_close(x[1], torch.from_numpy(expected[3:6]))
    torch.testing.assert_close(x[0, 2], torch.zeros(5))
    torch.testing.assert_close(direction.norm(dim=1), torch.ones(2))
