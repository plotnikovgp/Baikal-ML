#!/usr/bin/env python3
"""Small physical-invariance checks for predicted-hit threshold augmentation."""

import torch
from train_mixed_universal import thin_predicted_hits


def test_threshold_preserves_true_line_and_selection_cut():
    x = torch.zeros(3, 10, 5)
    x[:, 5:, 2] = 10.0
    x[2, 5:8, 2] = 0.0
    x[:, :, 4] = torch.arange(10)
    mask = torch.ones(3, 10, dtype=torch.bool)
    probabilities = torch.full((3, 10), 0.95)
    probabilities[0, 0] = 0.72
    probabilities[1, :3] = 0.72
    probabilities[2, 8:] = 0.72
    direction = torch.tensor([[0.0, 0.0, 1.0]]).expand(3, -1).clone()
    point = torch.tensor([[1.0, 2.0, 0.0]]).expand(3, -1).clone()
    centroid = x[:, :, 2:5].mean(dim=1)
    new_x, new_mask, new_point = thin_predicted_hits(
        x,
        mask,
        probabilities,
        direction,
        point,
        torch.tensor([True, True, True]),
        centroid,
        torch.zeros(5),
        torch.ones(5),
        1.0,
        0.8,
        0.8,
    )
    assert int(new_mask[0].sum()) == 9
    assert int(new_mask[1].sum()) == 10  # Seven hits would violate the 2s8h cut.
    assert int(new_mask[2].sum()) == 10  # Eight hits on one string also fails.
    assert torch.all(new_x[0, 0] == 0)
    assert torch.equal(new_point[1], point[1])
    new_centroid = new_x[0, new_mask[0], 2:5].mean(dim=0)
    original_line = centroid[0] + point[0]
    new_line = new_centroid + new_point[0]
    assert torch.allclose(original_line[:2], new_line[:2], atol=1e-6)


if __name__ == "__main__":
    test_threshold_preserves_true_line_and_selection_cut()
