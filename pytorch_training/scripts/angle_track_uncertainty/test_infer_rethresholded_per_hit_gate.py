"""Unit checks for the optional per-hit photon compatibility experiment."""

from __future__ import annotations

import h5py
import numpy as np
import pytest
from infer_rethresholded_for_audit import compatible_added_hit_masks, selected_batch


def test_photon_gate_keeps_compatible_hits_when_one_extra_is_bad(tmp_path):
    speed = 0.299792458
    z = np.tile(np.arange(4, dtype=np.float64), 2)
    time = z / speed + np.asarray([200, 200, 210, 220, 0, 0, 10, 100])
    features = np.column_stack((np.ones(8), time, np.zeros(8), np.zeros(8), z))
    probability = np.tile([0.9, 0.8, 0.6, 0.65], 2)
    path = tmp_path / "hits.h5"
    with h5py.File(path, "w") as h5:
        group = h5.create_group("test")
        group.create_dataset("ev_starts/data", data=[0, 4, 8])
        group.create_dataset("data/data", data=features)
        group.create_dataset("signal_probability/data", data=probability)
    with h5py.File(path) as h5:
        masks = compatible_added_hit_masks(
            h5["test"],
            begin=0,
            end=2,
            source_indices=np.asarray([0, 1]),
            directions=np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
            anchors=np.zeros((2, 3)),
            high_threshold=0.7,
            low_threshold=0.5,
            maximum_residual_ns=50.0,
            stored_mean=np.zeros(5),
            stored_std=np.ones(5),
        )
    assert masks[0].tolist() == [False, False, True, True]
    assert masks[1].tolist() == [False, False, True, False]


def test_selected_batch_uses_only_explicit_extra_hit_mask(tmp_path):
    path = tmp_path / "selected.h5"
    features = np.zeros((10, 5), dtype=np.float32)
    features[:, 2] = np.arange(10)
    probabilities = np.r_[np.full(8, 0.9), np.full(2, 0.6)]
    with h5py.File(path, "w") as h5:
        group = h5.create_group("test")
        group.create_dataset("ev_starts/data", data=[0, 10])
        group.create_dataset("ev_ids/data", data=np.asarray([b"nue2_example"]))
        group.create_dataset("data/data", data=features)
        group.create_dataset("signal_probability/data", data=probabilities)
        group.create_dataset("channels/data", data=np.asarray([0, 1, 2, 3, 4, 5, 6, 36, 37, 38]))
    prep = {
        "input_norm_mean": np.zeros(5),
        "input_norm_std": np.ones(5),
        "max_seq_len": 128,
        "center_time": False,
    }
    with h5py.File(path) as h5:
        kept, x, mask = selected_batch(
            h5["test"],
            0,
            1,
            0.7,
            prep,
            np.zeros(5),
            np.ones(5),
            36,
            0,
            1,
            extra_hit_masks={0: np.asarray([False] * 8 + [True, False])},
        )
        assert kept[0][2:4] == (9, 2)
        assert len(kept[0][5]) == 9
        assert x.shape == (1, 9, 5)
        assert int(mask.sum()) == 9
        assert kept[0][4][0] == pytest.approx(4.0)
        with pytest.raises(ValueError, match="only omitted hits"):
            selected_batch(
                h5["test"],
                0,
                1,
                0.7,
                prep,
                np.zeros(5),
                np.ones(5),
                36,
                0,
                1,
                extra_hit_masks={0: np.asarray([True] + [False] * 9)},
            )
