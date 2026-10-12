#!/usr/bin/env python3
"""Backward-compatible SetTransformer input with a per-hit signal score."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from train_angle_signal import SetTransformerDirection, engineered_features, masked_mean


class SoftScoreSetTransformerDirection(SetTransformerDirection):
    """Append one score to the ten existing engineered hit features.

    The extra projection column is zero-initialized when upgrading a trained
    five-feature backbone, so its initial predictions are unchanged.
    """

    def __init__(
        self,
        hidden: int,
        layers: int,
        heads: int,
        ff_mult: int,
        dropout: float,
        score_mode: str = "absolute",
    ) -> None:
        super().__init__(hidden, layers, heads, ff_mult, dropout, "engineered")
        if score_mode not in ("absolute", "centered"):
            raise ValueError(f"Invalid score mode {score_mode}")
        self.score_mode = score_mode
        self.embed[0] = nn.Linear(11, hidden)

    @classmethod
    def from_pretrained(cls, original: SetTransformerDirection, score_mode: str = "absolute"):
        if original.feature_mode != "engineered":
            raise ValueError("SoftScoreSetTransformerDirection requires engineered features")
        hidden = original.embed[0].out_features
        layer = original.encoder.layers[0]
        result = cls(
            hidden,
            len(original.encoder.layers),
            layer.self_attn.num_heads,
            layer.linear1.out_features // hidden,
            layer.dropout.p,
            score_mode,
        )
        old_state = original.state_dict()
        new_state = result.state_dict()
        for name, old_value in old_state.items():
            if name == "embed.0.weight":
                new_state[name][:, :10] = old_value
                new_state[name][:, 10] = 0
            elif new_state[name].shape == old_value.shape:
                new_state[name] = old_value
            else:
                raise ValueError(f"Unmatched backbone weight {name}")
        result.load_state_dict(new_state, strict=True)
        return result

    def forward(
        self, x: Tensor, mask: Tensor, return_features: bool = False
    ) -> Tensor | tuple[Tensor, Tensor]:
        if x.shape[-1] != 6:
            raise ValueError(f"Expected 6 hit features, got {x.shape[-1]}")
        physical = engineered_features(x[..., :5], mask)
        if self.score_mode == "centered":
            score = x[..., 5:6] - masked_mean(x[..., 5:6], mask)[:, None]
        else:
            score = x[..., 5:6] - 0.85
        score = (score / 0.15) * mask.unsqueeze(-1)
        h = self.embed(torch.cat((physical, score), dim=-1))
        cls = self.cls.expand(len(h), -1, -1)
        h = torch.cat((cls, h), dim=1)
        full_mask = torch.cat(
            (torch.ones((len(mask), 1), dtype=torch.bool, device=mask.device), mask),
            dim=1,
        )
        h = self.encoder(h, src_key_padding_mask=~full_mask)
        pooled = torch.cat((h[:, 0], masked_mean(h[:, 1:], mask)), dim=1)
        direction = torch.nn.functional.normalize(self.head(pooled), dim=1)
        return (direction, pooled) if return_features else direction


def upgrade_track_model(model: nn.Module, score_mode: str = "absolute") -> nn.Module:
    model.backbone = SoftScoreSetTransformerDirection.from_pretrained(model.backbone, score_mode)
    return model
