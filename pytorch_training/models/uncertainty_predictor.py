from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from .encoder import Encoder


class UncertaintyPredictor(nn.Module):
    """Predicts uncertainty estimates using frozen encoder hiddens.

    Args:
        model: Pre-trained Encoder whose weights will be frozen.
        hidden_size: Width of the sigma predictor MLP hidden layers.
        num_predictor_layers: Number of hidden layers in the predictor MLP.
    """

    def __init__(
        self,
        model: Encoder,
        hidden_size: int = 512,
        num_predictor_layers: int = 1,
        **kwargs,
    ) -> None:
        super().__init__()

        self.device = "cuda"
        self.model = model.eval()

        for p in self.model.parameters():
            p.requires_grad = False

        self.model.return_hidden = True

        layers: list[nn.Module] = []
        in_size = model.hidden_size
        for _ in range(num_predictor_layers):
            layers.extend([nn.Linear(in_size, hidden_size), nn.ReLU()])
            in_size = hidden_size
        layers.append(nn.Linear(in_size, 3))

        self.predictor = nn.Sequential(*layers)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        with torch.no_grad():
            output, hidden = self.model(x, mask)

        output = output / output.norm(dim=1, keepdim=True)

        pooled = hidden.mean(1)
        log_sigma = self.predictor(pooled)

        return torch.cat([output, log_sigma], dim=-1)
