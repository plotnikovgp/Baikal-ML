"""Synthetic checks for the optional added-hit photon-time gate."""

from __future__ import annotations

import h5py
import numpy as np
from infer_rethresholded_for_audit import added_hits_pass_photon_gate


def test_all_added_hits_must_match_first_pass_clock(tmp_path):
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
        passed = added_hits_pass_photon_gate(
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
    assert passed.tolist() == [True, False]
