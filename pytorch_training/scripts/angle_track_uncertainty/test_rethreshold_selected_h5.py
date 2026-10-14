"""Regression checks for selected-hit rethresholding and event alignment."""

from __future__ import annotations

import sys

import h5py
import numpy as np
import pytest
from rethreshold_selected_h5 import main


def test_rethreshold_keeps_only_valid_events_and_aligned_fields(tmp_path, monkeypatch):
    source = tmp_path / "source.h5"
    output = tmp_path / "output.h5"
    channels = np.concatenate(
        (
            np.array([0, 1, 2, 3, 36, 37, 38, 39, 40, 41]),
            np.arange(10),
            np.array([0, 1, 2, 3, 36, 37, 38, 39, 40, 41]),
        )
    )
    scores = np.concatenate(
        (np.array([0.8] * 8 + [0.6] * 2), np.full(10, 0.8), np.array([0.8] * 7 + [0.6] * 3))
    )
    with h5py.File(source, "w") as h5:
        h5.create_dataset("norm_param/mean", data=np.zeros(5, dtype=np.float32))
        h5.create_dataset("norm_param/std", data=np.ones(5, dtype=np.float32))
        arrays = {
            "channels": channels,
            "data": np.arange(150, dtype=np.float32).reshape(30, 5),
            "labels": np.arange(30),
            "signal_probability": scores,
            "t_res": np.arange(30, dtype=np.float32),
            "cluster_ids": np.array([11, 22, 33]),
            "ev_ids": np.array([b"one", b"two", b"three"]),
            "num_un_strings": np.array([2, 1, 2]),
            "prime_prty": np.arange(18, dtype=np.float32).reshape(3, 6),
            "ev_starts": np.array([0, 10, 20, 30]),
        }
        for key, value in arrays.items():
            h5.create_dataset(f"test/{key}/data", data=value)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rethreshold_selected_h5.py",
            "--source",
            str(source),
            "--output",
            str(output),
            "--threshold",
            "0.70",
        ],
    )
    main()
    with h5py.File(output) as h5:
        assert h5["test/ev_ids/data"][:].tolist() == [b"one"]
        assert h5["test/ev_starts/data"][:].tolist() == [0, 8]
        assert h5["test/cluster_ids/data"][:].tolist() == [11]
        assert h5["test/num_un_strings/data"][:].tolist() == [2]
        np.testing.assert_array_equal(h5["test/channels/data"][:], channels[:8])
        np.testing.assert_array_equal(h5["test/data/data"][:], arrays["data"][:8])
    with pytest.raises(FileExistsError):
        main()
