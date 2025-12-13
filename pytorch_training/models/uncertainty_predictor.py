from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from .encoder import Encoder


class UncertaintyPredictor(nn.Module):
    """Predicts uncertainty estimates using frozen encoder hiddens."""

    def __init__(
        self,
        model: Encoder,
        hidden_size: int = 64,
        **kwargs,
    ) -> None:
        super().__init__()

        self.device = "cuda"
        self.model = model.eval()
        self.hidden_size = hidden_size

        for p in self.model.parameters():
            p.requires_grad = False

        self.model.return_hiddens_by_layers = True

        self.feature_extractor = nn.Sequential(
            nn.Linear(model.hidden_size, model.hidden_size),
            nn.ReLU(),
            nn.Linear(model.hidden_size, hidden_size // 2),
        )

        self.predictor = nn.Sequential(
            nn.Linear(hidden_size // 2 + 3, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 3),
        )

        num_layers = len(model.enc.layers) if hasattr(model.enc, "layers") else model.num_layers
        self.layer_weights = nn.Parameter(torch.zeros(num_layers), requires_grad=True)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        with torch.no_grad():
            output, hiddens_by_layer = self.model(x, mask)

        output = output / output.norm(dim=1, keepdim=True)

        hiddens = torch.stack(hiddens_by_layer[1:], dim=0)
        pooled = hiddens.sum(2)

        weights = torch.softmax(self.layer_weights, dim=0)
        feats_summary = torch.sum(weights[:, None, None] * pooled, dim=0)

        feats = self.feature_extractor(feats_summary)
        combined = torch.cat([output, feats], dim=-1)
        log_sigma = self.predictor(combined)

        return torch.cat([output, log_sigma], dim=-1)
