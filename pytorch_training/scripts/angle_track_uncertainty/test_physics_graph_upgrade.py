#!/usr/bin/env python3
"""Check physical graph units and exact v4 initialization."""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from physics_graph_upgrade import upgrade_track_model_graph
from train_mixed_universal import AdaptedTrackDataset, model_from_checkpoints


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-data", type=Path, required=True)
    parser.add_argument("--pred-anchors", type=Path, required=True)
    parser.add_argument("--norm-data", type=Path, required=True)
    parser.add_argument("--angle-checkpoint", type=Path, required=True)
    parser.add_argument("--track-checkpoint", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--graph-layers", type=int, default=2)
    parser.add_argument("--graph-k", type=int, default=12)
    args = parser.parse_args()
    torch.set_num_threads(4)
    with h5py.File(args.norm_data, "r") as h5:
        mean = np.asarray(h5["norm_param/mean"], dtype=np.float32)
        std = np.asarray(h5["norm_param/std"], dtype=np.float32)
    dataset = AdaptedTrackDataset(
        str(args.pred_data),
        "val",
        16,
        128,
        False,
        16,
        False,
        anchor_path=str(args.pred_anchors),
        keep=None,
        expected_mean=mean,
        expected_std=std,
    )
    x, _truth, mask, *_ = dataset[0]
    original = model_from_checkpoints(args.angle_checkpoint, args.track_checkpoint, 50.0, "cpu")
    checkpoint = torch.load(args.base_checkpoint, map_location="cpu", weights_only=False)
    original.load_state_dict(checkpoint["model"], strict=True)
    graph = upgrade_track_model_graph(
        copy.deepcopy(original),
        graph_layers=args.graph_layers,
        graph_k=args.graph_k,
        photon_speed_m_per_ns=0.225,
        minkowski_edge=True,
    )
    graph.backbone.set_feature_norm(mean, std)
    original.eval()
    graph.eval()
    with torch.inference_mode():
        base_dir, base_point = original(x, mask)
        graph_dir, graph_point = graph(x, mask)
        neighbors, edge, valid = graph.backbone.make_graph(x, mask)
    torch.testing.assert_close(base_dir, graph_dir, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(base_point, graph_point, rtol=1e-5, atol=1e-5)
    assert edge.shape[-1] == 11 and neighbors.shape == valid.shape
    assert torch.isfinite(edge).all()
    baseline_times, graph_times = [], []
    with torch.inference_mode():
        for _ in range(4):
            start = time.perf_counter()
            original(x, mask)
            baseline_times.append(time.perf_counter() - start)
            start = time.perf_counter()
            graph(x, mask)
            graph_times.append(time.perf_counter() - start)
    print(
        {
            "events": len(x),
            "graph_layers": args.graph_layers,
            "graph_k": args.graph_k,
            "max_direction_diff": float((base_dir - graph_dir).abs().max()),
            "max_point_diff_m": float((base_point - graph_point).abs().max()),
            "cpu_ms_per_event_base": 1000 * np.median(baseline_times) / len(x),
            "cpu_ms_per_event_graph": 1000 * np.median(graph_times) / len(x),
        }
    )


if __name__ == "__main__":
    main()
