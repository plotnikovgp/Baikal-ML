#!/usr/bin/env python3
"""Identity-initialized local photon-geometry graph around the v4 backbone."""

from __future__ import annotations

from torch import nn
from train_angle_signal import PhysicsGraphTransformerDirection, SetTransformerDirection


def upgrade_track_model_graph(
    model: nn.Module,
    *,
    graph_layers: int,
    graph_k: int,
    photon_speed_m_per_ns: float,
    minkowski_edge: bool,
) -> nn.Module:
    original = model.backbone
    if not isinstance(original, SetTransformerDirection) or original.feature_mode != "engineered":
        raise ValueError("Graph upgrade requires the engineered-feature SetTransformer")
    if graph_layers < 1 or graph_k < 1 or photon_speed_m_per_ns <= 0:
        raise ValueError("Graph layers, neighbors, and photon speed must be positive")
    hidden = original.embed[0].out_features
    layer = original.encoder.layers[0]
    graph = PhysicsGraphTransformerDirection(
        hidden,
        len(original.encoder.layers),
        layer.self_attn.num_heads,
        layer.linear1.out_features // hidden,
        layer.dropout.p,
        original.feature_mode,
        graph_layers,
        graph_k,
        photon_speed_m_per_ns,
        minkowski_edge,
    )
    old_state = original.state_dict()
    new_state = graph.state_dict()
    for name, value in old_state.items():
        if name not in new_state or new_state[name].shape != value.shape:
            raise ValueError(f"Cannot transfer backbone weight {name}")
        new_state[name] = value
    graph.load_state_dict(new_state, strict=True)
    model.backbone = graph
    return model
