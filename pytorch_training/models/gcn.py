from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    """Graph Convolutional Network with 4 layers."""

    def __init__(
        self,
        in_features: int,
        hidden_size: int,
        out_size: int,
        dropout_p: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.conv1 = GCNConv(in_features, hidden_size)
        self.conv2 = GCNConv(hidden_size, hidden_size)
        self.conv3 = GCNConv(hidden_size, hidden_size)
        self.conv4 = GCNConv(hidden_size, out_size)

        self.dropout1 = nn.Dropout(p=dropout_p)
        self.dropout2 = nn.Dropout(p=dropout_p)
        self.dropout3 = nn.Dropout(p=dropout_p)

    def forward(self, x: Tensor, edge_index: Tensor, batch: Tensor | None = None) -> Tensor:
        x = F.elu(self.conv1(x, edge_index))
        x = self.dropout1(x)

        x = F.elu(self.conv2(x, edge_index))
        x = self.dropout2(x)

        x = F.elu(self.conv3(x, edge_index))
        x = self.dropout3(x)

        x = self.conv4(x, edge_index)
        return x
