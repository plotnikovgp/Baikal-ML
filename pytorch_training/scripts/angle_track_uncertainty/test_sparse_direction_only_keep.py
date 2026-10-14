"""Safety checks for the p>=0.50 direction-only training stream."""

import numpy as np
import pytest
import torch
from train_mixed_universal import (
    exclude_evaluation_ids,
    multi_muatm_direction_only_keep,
    primary_fingerprints,
    sparse_direction_only_keep,
)
from train_track_anchor_v3 import point_loss


def test_only_invalid_neutrino_points_are_used() -> None:
    event_ids = np.asarray([b"nue2_0", b"nuatm_1", b"muatm_2", b"nue2_3", b"nuatm_4"])
    anchor_valid = np.asarray([False, False, False, True, True])
    np.testing.assert_array_equal(
        sparse_direction_only_keep(event_ids, anchor_valid),
        [True, True, False, False, False],
    )


def test_mismatched_lengths_rejected() -> None:
    with pytest.raises(ValueError, match="different lengths"):
        sparse_direction_only_keep(np.asarray([b"nue2_0"]), np.asarray([]))


def test_multi_muatm_keeps_only_events_without_a_unique_point() -> None:
    ids = np.asarray([b"muatm_1_0", b"muatm_1_1", b"nuatm_2", b"nue2_3"])
    valid = np.asarray([False, True, False, False])
    np.testing.assert_array_equal(
        multi_muatm_direction_only_keep(ids, valid), [True, False, False, False]
    )
    with pytest.raises(ValueError, match="different lengths"):
        multi_muatm_direction_only_keep(ids, valid[:2])


def test_held_out_ids_removed_from_sparse_stream() -> None:
    ids = np.asarray([b"nue2_0", b"nuatm_1", b"nue2_0", b"nuatm_2"])
    keep = np.asarray([True, True, True, False])
    result, excluded = exclude_evaluation_ids(ids, keep, np.asarray([b"nue2_0"]))
    np.testing.assert_array_equal(result, [False, True, False, False])
    assert excluded == 2
    np.testing.assert_array_equal(keep, [True, True, True, False])


def test_evaluation_filter_mismatched_lengths_rejected() -> None:
    with pytest.raises(ValueError, match="different lengths"):
        exclude_evaluation_ids(np.asarray([b"nue2_0"]), np.asarray([]), np.asarray([]))


def test_primary_fingerprint_exact_float32_triples() -> None:
    values = np.asarray(
        [[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 5.0], [1.0, 2.0, 3.01, 4.0]], dtype=np.float32
    )
    keys = primary_fingerprints(values)
    assert keys[0] == keys[1]
    assert keys[0] != keys[2]
    with pytest.raises(ValueError, match="float32"):
        primary_fingerprints(values.astype(np.float64))


def test_invalid_point_target_has_no_loss_or_gradient() -> None:
    prediction = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
    target = torch.full((1, 3), float("nan"))
    direction = torch.tensor([[0.0, 0.0, 1.0]])
    valid = torch.tensor([False])
    loss = point_loss(prediction, target, direction, valid, 50.0, "line_huber")
    assert loss.item() == 0.0
    loss.backward()
    torch.testing.assert_close(prediction.grad, torch.zeros_like(prediction))
