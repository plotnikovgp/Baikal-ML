"""Identity initialization and physical symmetries of the pair-vote head."""

import math

import torch
from pair_vote_residual import PairVoteResidual
from torch import nn


class DummyBase(nn.Module):
    def forward(self, x, mask):
        direction = torch.tensor([0.6, 0.0, 0.8], device=x.device).expand(len(x), -1)
        point = torch.ones((len(x), 3), device=x.device)
        return direction, point


def hits():
    x = torch.zeros((2, 10, 5))
    x[:, :, 0] = torch.arange(10) + 1
    x[:, :, 1] = torch.arange(10) * 18.0
    x[:, :, 2] = torch.arange(10) * 4.0
    x[:, :, 3] = torch.sin(torch.arange(10).float()) * 7.0
    x[:, :, 4] = torch.arange(10) * 12.0
    mask = torch.ones((2, 10), dtype=torch.bool)
    mask[1, 8:] = False
    return x, mask


def test_zero_residual_is_exact_base_and_has_gradient():
    model = PairVoteResidual(DummyBase(), torch.zeros(5), torch.ones(5), sample_hits=8)
    x, mask = hits()
    direction, point = model(x, mask)
    expected_direction, expected_point = model.base(x, mask)
    torch.testing.assert_close(direction, expected_direction, atol=0.0, rtol=0.0)
    torch.testing.assert_close(point, expected_point, atol=0.0, rtol=0.0)
    direction[:, 1].sum().backward()
    assert model.edge_head[-1].weight.grad is not None
    assert model.edge_head[-1].weight.grad.abs().sum() > 0


def test_pair_vote_rotates_and_ignores_global_time_offset():
    torch.manual_seed(17)
    model = PairVoteResidual(DummyBase(), torch.zeros(5), torch.ones(5), sample_hits=8)
    with torch.no_grad():
        model.edge_head[-1].weight.normal_(std=0.04)
        model.edge_head[-1].bias.copy_(torch.tensor([0.1, 0.2, -0.1, 0.3]))
    x, mask = hits()
    direction = torch.tensor([[0.6, 0.0, 0.8], [0.3, 0.4, 0.8660254]])
    vote = model.pair_vote(x, mask, direction)
    shifted = x.clone()
    shifted[..., 1] += 250.0
    torch.testing.assert_close(
        model.pair_vote(shifted, mask, direction), vote, atol=1e-6, rtol=1e-5
    )

    angle = math.pi / 3
    cosine, sine = math.cos(angle), math.sin(angle)
    rotated = x.clone()
    rotated[..., 2] = cosine * x[..., 2] - sine * x[..., 3]
    rotated[..., 3] = sine * x[..., 2] + cosine * x[..., 3]
    direction_rotated = direction.clone()
    direction_rotated[:, 0] = cosine * direction[:, 0] - sine * direction[:, 1]
    direction_rotated[:, 1] = sine * direction[:, 0] + cosine * direction[:, 1]
    expected = vote.clone()
    expected[:, 0] = cosine * vote[:, 0] - sine * vote[:, 1]
    expected[:, 1] = sine * vote[:, 0] + cosine * vote[:, 1]
    torch.testing.assert_close(
        model.pair_vote(rotated, mask, direction_rotated),
        expected,
        atol=1e-6,
        rtol=1e-5,
    )
