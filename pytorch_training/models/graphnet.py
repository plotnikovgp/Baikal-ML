from __future__ import annotations

import ast

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric import nn as gnn
from torch_geometric.nn import knn_graph


class DynEdgeConv(gnn.EdgeConv):
    """Dynamic EdgeConv that recomputes KNN graph after each layer."""

    def __init__(
        self,
        nn: nn.Module,
        aggr: str = "max",
        nb_neighbors: int = 4,
        features_subset: list[int] | slice | None = None,
        **kwargs,
    ) -> None:
        super().__init__(nn=nn, aggr=aggr, **kwargs)

        self.nb_neighbors = nb_neighbors
        self.features_subset = features_subset if features_subset is not None else slice(None)

    def forward(
        self, x: Tensor, edge_index: Tensor, batch: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        x = super().forward(x, edge_index)

        edge_index = knn_graph(
            x=x[:, self.features_subset],
            k=self.nb_neighbors,
            batch=batch,
        ).to(x.device)

        return x, edge_index


class GraphnetDynedge(nn.Module):
    """Dynamic Edge Convolution Graph Neural Network."""

    DEFAULT_LAYER_SIZES = [
        (128, 256),
        (336, 256),
        (336, 256),
        (336, 256),
    ]

    def __init__(
        self,
        in_features: int,
        knn_neighbours: int = 4,
        dynedge_layer_sizes: list[str] | None = None,
        out_size: int = 2,
        second_head_out_size: int | None = None,
        aggregate_output: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.aggregate_output = aggregate_output

        layer_sizes = (
            [ast.literal_eval(size) for size in dynedge_layer_sizes]
            if dynedge_layer_sizes
            else self.DEFAULT_LAYER_SIZES
        )

        self._conv_layers = nn.ModuleList()
        self._activation = nn.ReLU()

        nb_latent_features = in_features
        for sizes in layer_sizes:
            layers = self._build_conv_layers(nb_latent_features, sizes)
            conv_layer = DynEdgeConv(
                nn.Sequential(*layers), aggr="add", nb_neighbors=knn_neighbours
            )
            self._conv_layers.append(conv_layer)
            nb_latent_features = sizes[-1]

        total_features = sum(sizes[-1] for sizes in layer_sizes) + in_features
        self._post_processing = self._build_post_processing(total_features, [336, 256])
        self.head = nn.Linear(256, out_size)

    def _build_conv_layers(self, nb_input: int, sizes: tuple[int, ...]) -> list[nn.Module]:
        layers: list[nn.Module] = []
        layer_sizes = [nb_input] + list(sizes)

        for ix, (nb_in, nb_out) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            if ix == 0:
                nb_in *= 2
            layers.append(nn.Linear(nb_in, nb_out))
            layers.append(self._activation)

        return layers

    def _build_post_processing(self, nb_input: int, layer_sizes: list[int]) -> nn.Sequential:
        layers: list[nn.Module] = []
        sizes = [nb_input] + layer_sizes

        for nb_in, nb_out in zip(sizes[:-1], sizes[1:]):
            layers.append(nn.Linear(nb_in, nb_out))
            layers.append(self._activation)

        return nn.Sequential(*layers)

    def forward(self, x: Tensor, edge_index: Tensor, batch: Tensor) -> Tensor:
        skip_connections = [x]

        for conv_layer in self._conv_layers:
            x, edge_index = conv_layer(x, edge_index, batch)
            skip_connections.append(x)

        x = torch.cat(skip_connections, dim=1)
        y = self._post_processing(x)
        z = self.head(y)

        if self.aggregate_output:
            return gnn.global_mean_pool(z, batch)
        return z


class GraphnetEncoder(nn.Module):
    """Standalone encoder for use within GraphnetAndEncoderStack."""

    def __init__(
        self,
        in_features: int,
        hidden_size: int,
        num_layers: int,
        dim_feedforward_size: int,
        n_heads: int,
        out_size: int,
        dropout_p: float = 0.0,
        aggregator: nn.Module | None = None,
        second_head_out_size: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.aggregator = aggregator

        self.first_layer = nn.Linear(in_features, hidden_size)
        enc_layer = nn.TransformerEncoderLayer(
            hidden_size, n_heads, dim_feedforward_size, dropout_p, batch_first=True
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers)
        self.head = nn.Linear(hidden_size, out_size)

        self.second_head = (
            nn.Linear(hidden_size, second_head_out_size)
            if second_head_out_size is not None
            else None
        )

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        x = self.first_layer(x)
        x = self.enc(x, src_key_padding_mask=~mask)
        y = self.head(x)

        if self.aggregator is not None:
            return self.aggregator(x)

        if self.second_head is not None:
            z = self.second_head(x)
            return torch.cat([y, z], dim=-1)

        return y


class GraphnetAndEncoderStack(nn.Module):
    """Stacked GraphNet and Transformer Encoder."""

    def __init__(self, **kwargs) -> None:
        super().__init__()

        self.encoder = GraphnetEncoder(**kwargs["encoder_params"])
        self.graphnet = GraphnetDynedge(**kwargs["graphnet_params"])

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        x = self.graphnet(x, mask)
        return self.encoder(x, mask)


# Backward compatibility alias
Encoder = GraphnetEncoder
