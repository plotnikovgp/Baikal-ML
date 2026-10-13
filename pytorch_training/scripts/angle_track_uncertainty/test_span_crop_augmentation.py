"""Physical-line and 2-string/8-hit invariants for vertical-span cropping."""

import torch
from span_crop_augmentation import crop_vertical_span


def test_span_crop_shortens_z_spread_without_moving_truth_line():
    torch.manual_seed(7)
    x = torch.zeros((1, 20, 5))
    x[0, 10:, 2] = 10.0  # second string
    x[0, :, 4] = torch.arange(20) * 10.0
    mask = torch.ones((1, 20), dtype=torch.bool)
    direction = torch.tensor([[0.0, 0.0, 1.0]])
    point = torch.tensor([[2.0, 3.0, 0.0]])
    old_center = x[0, :, 2:5].mean(dim=0)
    out, new_mask, new_point = crop_vertical_span(
        x,
        mask,
        direction,
        point,
        torch.tensor([True]),
        torch.zeros(5),
        torch.ones(5),
        probability=1.0,
        min_keep_fraction=0.7,
        max_keep_fraction=0.7,
    )
    assert int(new_mask.sum()) == 14
    assert torch.unique(out[0, new_mask[0], 2]).numel() == 2
    assert out[0, new_mask[0], 4].std() < x[0, :, 4].std()
    assert torch.all(out[~new_mask] == 0)
    new_center = out[0, new_mask[0], 2:5].mean(dim=0)
    torch.testing.assert_close(
        (old_center + point[0])[:2], (new_center + new_point[0])[:2], atol=1e-6, rtol=0
    )


def test_span_crop_falls_back_if_hit_or_string_cut_would_fail():
    x = torch.zeros((2, 8, 5))
    x[0, 4:, 2] = 10.0  # two strings but already at the 8-hit minimum
    x[1, :, 2] = 10.0  # one string only
    x[:, :, 4] = torch.arange(8)
    mask = torch.ones((2, 8), dtype=torch.bool)
    point = torch.tensor([[1.0, 2.0, 0.0], [1.0, 2.0, 0.0]])
    out, new_mask, new_point = crop_vertical_span(
        x,
        mask,
        torch.tensor([[0.0, 0.0, 1.0]]).expand(2, -1),
        point,
        torch.tensor([True, True]),
        torch.zeros(5),
        torch.ones(5),
        probability=1.0,
        min_keep_fraction=0.5,
        max_keep_fraction=0.5,
    )
    torch.testing.assert_close(out, x)
    torch.testing.assert_close(new_mask, mask)
    torch.testing.assert_close(new_point, point)


def test_zero_probability_is_exact_noop():
    x = torch.randn(1, 9, 5)
    mask = torch.ones((1, 9), dtype=torch.bool)
    direction = torch.tensor([[0.0, 0.0, 1.0]])
    point = torch.zeros((1, 3))
    out, new_mask, new_point = crop_vertical_span(
        x,
        mask,
        direction,
        point,
        torch.tensor([True]),
        torch.zeros(5),
        torch.ones(5),
        probability=0.0,
        min_keep_fraction=0.75,
        max_keep_fraction=0.9,
    )
    assert out is x and new_mask is mask and new_point is point
