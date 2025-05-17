import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder import Encoder


class UncertaintyPredictor(nn.Module):
    def __init__(self, model: Encoder, hidden_size=64):
        super().__init__()
        self.device = "cuda"
        self.model = model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.return_hiddens_by_layers = True
        self.hidden_size = hidden_size

        self.feature_extractor_from_hidden = nn.Sequential(
            nn.Linear(model.hidden_size, model.hidden_size),
            nn.ReLU(),
            nn.Linear(model.hidden_size, self.hidden_size // 2),
        )
        self.predictor = nn.Sequential(
            nn.Linear(self.hidden_size // 2 + 3, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 3),
        )

        if hasattr(model.enc, "layers"):
            num_layers = len(model.enc.layers)
        else:
            num_layers = model.num_layers

        self.layer_weights = nn.Parameter(torch.zeros(num_layers), requires_grad=True)

    def forward(self, x, mask):
        with torch.no_grad():
            x, hiddens_by_layer = self.model(x, mask)

        x = x / x.norm(dim=1, keepdim=True)

        hiddens = torch.stack(
            hiddens_by_layer[1:], dim=0
        )  # (num_layers, B, seq, hidden)
        pooled = hiddens.sum(2)  # e.g., use [CLS] token, shape (num_layers, B, hidden)

        weights = torch.nn.functional.softmax(
            self.layer_weights, dim=0
        )  # (num_layers,)
        feats_summary = torch.sum(weights[:, None, None] * pooled, dim=0)  # (B, hidden)

        feats = self.feature_extractor_from_hidden(feats_summary)
        x_and_hidden = torch.cat((x, feats), dim=-1)
        log_sigma = self.predictor(x_and_hidden)
        return torch.cat([x, log_sigma], dim=-1)
