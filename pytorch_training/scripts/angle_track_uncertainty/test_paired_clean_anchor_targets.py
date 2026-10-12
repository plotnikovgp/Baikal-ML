#!/usr/bin/env python3
"""Matched clean-hit anchors use only unambiguous MC truth points."""

from pathlib import Path

import h5py
import numpy as np
from build_paired_clean_anchor_targets import map_valid_raw_points


def test_map_valid_raw_points_prefers_valid_and_excludes_nue2(tmp_path: Path) -> None:
    path = tmp_path / "pred_anchors.h5"
    with h5py.File(path, "w") as h5:
        split = h5.create_group("train")
        split.create_dataset(
            "event_ids",
            data=np.array([b"nuatm_1", b"muatm_2", b"muatm_2", b"nue2_3"]),
        )
        split.create_dataset("valid", data=np.array([True, False, True, True]))
        split.create_dataset(
            "raw_particle_point_m",
            data=np.array(
                [[1, 2, 3], [9, 9, 9], [4, 5, 6], [7, 8, 9]],
                dtype=np.float32,
            ),
        )
    compact_ids = np.array([b"muatm_2", b"nuatm_1", b"nue2_3", b"nuatm_missing"])
    with h5py.File(path, "r") as h5:
        points, selected = map_valid_raw_points(compact_ids, h5["train"])
    assert selected.tolist() == [True, True, False, False]
    np.testing.assert_array_equal(points[0], [4, 5, 6])
    np.testing.assert_array_equal(points[1], [1, 2, 3])
    assert np.isnan(points[2:]).all()
