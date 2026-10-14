from __future__ import annotations

import torch
from sparse_hit_augmentation import crop_random_hits


def make_batch() -> tuple[torch.Tensor, ...]:
    x = torch.zeros((3, 16, 5), dtype=torch.float32)
    mask = torch.zeros((3, 16), dtype=torch.bool)
    mask[0, :16] = True
    mask[1, :10] = True
    mask[2, :8] = True
    x[..., 2] = 0.0
    x[:, 8:, 2] = 40.0
    x[1, 8:10, 2] = 40.0
    x[2, 4:8, 2] = 40.0
    x[..., 3] = torch.arange(16, dtype=torch.float32)[None]
    x[..., 4] = torch.arange(16, dtype=torch.float32)[None] * 7.0
    direction = torch.tensor([[0.0, 0.0, 1.0]]).repeat(3, 1)
    point = torch.tensor([[1.0, -2.0, 0.0]]).repeat(3, 1)
    valid = torch.tensor([True, True, False])
    mean = torch.zeros(5)
    std = torch.ones(5)
    return x, mask, direction, point, valid, mean, std


def test_crop_preserves_two_strings_and_global_truth_line() -> None:
    torch.manual_seed(9)
    x, mask, direction, point, valid, mean, std = make_batch()
    old_center = (x[..., 2:5] * mask[..., None]).sum(dim=1) / mask.sum(dim=1)[:, None]
    cropped_x, cropped_mask, cropped_point = crop_random_hits(
        x,
        mask,
        direction,
        point,
        valid,
        mean,
        std,
        probability=1.0,
        min_keep=8,
        max_keep=12,
    )
    assert 8 <= cropped_mask[0].sum() <= 12
    assert 8 <= cropped_mask[1].sum() <= 9
    assert torch.equal(cropped_mask[2], mask[2])
    assert torch.all((cropped_x == 0) | cropped_mask[..., None])
    for index in (0, 1):
        xy = cropped_x[index, cropped_mask[index], 2:4]
        assert xy[:, 0].max() - xy[:, 0].min() > 1.0
    new_center = (cropped_x[..., 2:5] * cropped_mask[..., None]).sum(dim=1) / cropped_mask.sum(
        dim=1
    )[:, None]
    old_global = old_center + point
    new_global = new_center + cropped_point
    transverse = new_global - old_global
    transverse -= (transverse * direction).sum(dim=1, keepdim=True) * direction
    assert torch.allclose(transverse[valid], torch.zeros_like(transverse[valid]), atol=1e-5)
    assert torch.equal(cropped_point[~valid], point[~valid])


def test_zero_probability_is_exact_identity() -> None:
    x, mask, direction, point, valid, mean, std = make_batch()
    cropped = crop_random_hits(
        x,
        mask,
        direction,
        point,
        valid,
        mean,
        std,
        probability=0.0,
    )
    assert all(torch.equal(before, after) for before, after in zip((x, mask, point), cropped))
