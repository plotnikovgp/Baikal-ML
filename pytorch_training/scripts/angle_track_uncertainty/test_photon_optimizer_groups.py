"""The photon-head continuation must actually update the intended weights."""

from __future__ import annotations

import torch
from torch import nn
from train_photon_residual import optimizer_groups


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.hit_head = nn.Linear(7, 4)
        self.base = nn.Module()
        self.base.backbone = nn.Module()
        self.base.backbone.head = nn.Linear(3, 3)
        self.base.point_head = nn.Linear(3, 3)
        self.base.requires_grad_(False)


def test_photon_head_optimizer_has_live_parameters_and_updates():
    model = TinyModel()
    groups, trainable = optimizer_groups(
        model,
        hit_learning_rate=0.01,
        base_learning_rate=0.001,
        train_base_heads=False,
    )
    assert len(groups[0]["params"]) == len(list(model.hit_head.parameters()))
    assert len(trainable) == len(groups[0]["params"])
    optimizer = torch.optim.AdamW(groups)
    before = model.hit_head.weight.detach().clone()
    model.hit_head(torch.randn(5, 7)).sum().backward()
    optimizer.step()
    assert not torch.equal(model.hit_head.weight, before)


def test_joint_groups_only_unfreeze_output_heads():
    model = TinyModel()
    groups, trainable = optimizer_groups(
        model,
        hit_learning_rate=0.01,
        base_learning_rate=0.001,
        train_base_heads=True,
    )
    assert len(groups) == 2
    assert all(parameter.requires_grad for parameter in trainable)
    assert len({id(parameter) for parameter in trainable}) == len(trainable)
    assert all(parameter.requires_grad for parameter in model.base.backbone.head.parameters())
    assert all(parameter.requires_grad for parameter in model.base.point_head.parameters())


def test_point_only_optimizer_preserves_direction_rows_exactly():
    model = TinyModel()
    model.hit_head = nn.Sequential(nn.Linear(7, 8), nn.Tanh(), nn.Linear(8, 4))
    groups, trainable = optimizer_groups(
        model,
        hit_learning_rate=0.01,
        base_learning_rate=0.001,
        train_base_heads=False,
        point_only_head=True,
    )
    assert len(trainable) == 2
    assert groups[0]["weight_decay"] == 0.0
    assert not any(parameter.requires_grad for parameter in model.hit_head[0].parameters())
    last = model.hit_head[-1]
    before_weight = last.weight.detach().clone()
    before_bias = last.bias.detach().clone()
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
    model.hit_head(torch.randn(5, 7)).sum().backward()
    optimizer.step()
    assert torch.equal(last.weight[:2], before_weight[:2])
    assert torch.equal(last.bias[:2], before_bias[:2])
    assert not torch.equal(last.weight[2:], before_weight[2:])
