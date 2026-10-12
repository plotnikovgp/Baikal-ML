"""Small checks for exact-ID mapping and valid raw track-point selection."""

import numpy as np
from build_joint_raw_index import lookup_ids, map_truth_points, unique_first


def test_lookup_uses_first_duplicate_and_marks_missing() -> None:
    source = np.array([b"b", b"a", b"b", b"c"], dtype="S4")
    wanted = np.array([b"b", b"c", b"d"], dtype="S4")
    index, present = lookup_ids(source, wanted)
    np.testing.assert_array_equal(index, [0, 3, -1])
    np.testing.assert_array_equal(present, [True, True, False])


def test_unique_first_keeps_aligned_row() -> None:
    ids = np.array([b"b", b"a", b"b"], dtype="S4")
    ids, value = unique_first(ids, np.array([4, 5, 6]))
    np.testing.assert_array_equal(ids, [b"b", b"a"])
    np.testing.assert_array_equal(value, [4, 5])


def test_truth_point_requires_valid_finite_match() -> None:
    ids = np.array([b"b", b"a", b"c", b"d"], dtype="S4")
    anchor_ids = np.array([b"a", b"b", b"c"], dtype="S4")
    valid = np.array([True, False, True])
    point = np.array([[1, 2, 3], [4, 5, 6], [np.nan, 1, 1]], dtype=np.float32)
    mapped, selected = map_truth_points(ids, anchor_ids, valid, point)
    np.testing.assert_array_equal(selected, [False, True, False, False])
    np.testing.assert_array_equal(mapped[1], [1, 2, 3])
    assert np.all(mapped[~selected] == 0)
