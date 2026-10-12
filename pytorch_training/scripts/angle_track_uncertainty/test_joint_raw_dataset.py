"""Raw-hit batches retain original labels and convert physical units once."""

from pathlib import Path

import h5py
import numpy as np
import torch
from joint_raw_dataset import IndexedRawBatchDataset


def test_indexed_raw_batch_normalization_and_labels(tmp_path: Path) -> None:
    source = tmp_path / "raw.h5"
    model_norm = tmp_path / "norm.h5"
    index = tmp_path / "index.npz"
    with h5py.File(source, "w") as h5:
        g = h5.create_group("norm_param")
        g.create_dataset("mean", data=np.array([10, 20, 0, 0, 0], dtype=np.float32))
        g.create_dataset("std", data=np.array([2, 2, 1, 1, 1], dtype=np.float32))
        train = h5.create_group("train")
        for name, values in {
            "ev_starts": [0, 2, 7],
            "labels": [0, -1, 2, 0, 0, -999999, 0],
        }.items():
            train.create_group(name).create_dataset("data", data=np.asarray(values))
        train.create_group("data").create_dataset(
            "data", data=np.arange(35, dtype=np.float32).reshape(7, 5)
        )
        train.create_group("prime_prty").create_dataset(
            "data", data=np.array([[0, 0], [90, 0]], dtype=np.float32)
        )
    with h5py.File(model_norm, "w") as h5:
        g = h5.create_group("norm_param")
        g.create_dataset("mean", data=np.zeros(5, dtype=np.float32))
        g.create_dataset("std", data=np.ones(5, dtype=np.float32))
    np.savez(
        index,
        train_raw_index=np.array([0, 1]),
        train_ids=np.array([b"nue2_1", b"nuatm_2"], dtype="S17"),
        train_point_m=np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
        train_point_valid=np.array([True, False]),
    )
    data = IndexedRawBatchDataset(source, index, "train", model_norm, 2, 3)
    x, mask, signal, direction, point, valid, particle = data[0]
    assert x.shape == (2, 3, 5)
    torch.testing.assert_close(x[0, 0, 0], torch.tensor(10.0))
    torch.testing.assert_close(x[0, 0, 1], torch.tensor(22.0))
    torch.testing.assert_close(signal[0, :2], torch.tensor([0.0, 1.0]))
    # Uniformly-spaced truncation selects the first, middle, and last hit.
    torch.testing.assert_close(signal[1], torch.tensor([1.0, 0.0, 0.0]))
    torch.testing.assert_close(direction[0], torch.tensor([0.0, 0.0, 1.0]))
    torch.testing.assert_close(point[0], torch.tensor([1.0, 2.0, 3.0]))
    assert mask.tolist() == [[True, True, False], [True, True, True]]
    assert valid.tolist() == [True, False]
    assert particle.tolist() == [2, 1]
