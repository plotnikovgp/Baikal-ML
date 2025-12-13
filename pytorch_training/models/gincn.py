from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric import nn as gnn


def _create_mlp(dim_in: int, dim_out: int) -> nn.Sequential:
    """Create a simple 2-layer MLP."""
    return nn.Sequential(
        gnn.Linear(dim_in, dim_out),
        nn.ReLU(),
        gnn.Linear(dim_out, dim_out),
    )


class GINCN(nn.Module):
    """Graph Isomorphism Network with Edge features."""

    def __init__(
        self,
        in_features: int,
        hidden_size: int,
        out_size: int,
        dropout_p: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__()

        self.conv1 = gnn.GINEConv(nn=_create_mlp(in_features, hidden_size))
        self.conv2 = gnn.GINEConv(nn=_create_mlp(hidden_size, hidden_size))
        self.conv3 = gnn.GINEConv(nn=_create_mlp(hidden_size, out_size))

        self.dropout1 = nn.Dropout(p=dropout_p)
        self.dropout2 = nn.Dropout(p=dropout_p)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = F.elu(self.conv1(x, edge_index))
        x = self.dropout1(x)

        x = F.elu(self.conv2(x, edge_index))
        x = self.dropout2(x)

        x = self.conv3(x, edge_index)
        return x


# Backward compatibility
get_mlp = _create_mlp
