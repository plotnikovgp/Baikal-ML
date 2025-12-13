from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GATv2Conv


class GAT(nn.Module):
    """Graph Attention Network using GATv2."""

    def __init__(
        self,
        in_features: int,
        hidden_size: int,
        out_size: int = 2,
        dropout_p: float = 0.0,
        heads: int = 1,
        **kwargs,
    ) -> None:
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.in_features = in_features
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.heads = heads

        self.gat1 = GATv2Conv(in_features, hidden_size, heads=heads)
        self.gat2 = GATv2Conv(hidden_size * heads, hidden_size, heads=1)
        self.dropout = nn.Dropout(p=dropout_p)
        self.linear = nn.Linear(hidden_size, out_size)

    def forward(self, x: Tensor, edge_index: Tensor, batch: Tensor | None = None) -> Tensor:
        x = F.elu(self.gat1(x, edge_index))
        x = self.dropout(x)
        x = F.elu(self.gat2(x, edge_index))
        x = self.linear(x)
        return x
