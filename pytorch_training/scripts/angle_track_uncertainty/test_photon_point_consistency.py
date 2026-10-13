"""Physical-line consistency must use absolute, centroid-corrected points."""

import torch
from train_photon_residual import point_consistency_loss


def test_consistency_is_invariant_to_hit_centroid_and_along_track_shift():
    features = torch.zeros((1, 2, 5))
    features[0, 1, 2] = 10.0
    thinned = torch.tensor([[True, False]])
    original = torch.tensor([[True, True]])
    direction = torch.tensor([[0.0, 0.0, 1.0]])
    valid = torch.tensor([True])
    mean = torch.zeros(5)
    std = torch.ones(5)
    zero = torch.zeros((1, 3))

    # Both offsets locate the same absolute x=5 m track line.
    same_line = point_consistency_loss(
        torch.tensor([[5.0, 0.0, 0.0]]),
        features,
        thinned,
        zero,
        features,
        original,
        direction,
        valid,
        mean,
        std,
    )
    assert same_line.item() < 1e-9

    # Sliding along the true z direction still describes the same line.
    along_track = point_consistency_loss(
        torch.tensor([[5.0, 0.0, 20.0]]),
        features,
        thinned,
        zero,
        features,
        original,
        direction,
        valid,
        mean,
        std,
    )
    assert along_track.item() < 1e-9

    displaced = torch.tensor([[10.0, 0.0, 0.0]], requires_grad=True)
    transverse = point_consistency_loss(
        displaced,
        features,
        thinned,
        zero,
        features,
        original,
        direction,
        valid,
        mean,
        std,
    )
    transverse.backward()
    assert transverse.item() > 0
    assert displaced.grad[0, 0] > 0
