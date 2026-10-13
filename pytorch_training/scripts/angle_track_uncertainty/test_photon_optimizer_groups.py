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
