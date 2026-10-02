#!/usr/bin/env python3
"""Train and evaluate signal-hit angle reconstruction models.

This harness intentionally keeps each HDF5 item as a pre-batched contiguous slice,
matching the existing Baikal reader while adding proper masked/CLS pooling,
mixed-precision training, finite runs, q68-first checkpointing, per-particle
metrics, and resumable optimizer state.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import h5py
import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, Subset

PARTICLE_NAMES = {0: "muatm", 1: "nuatm", 2: "nue2"}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def direction_from_angles(prime: Tensor) -> Tensor:
    theta = torch.deg2rad(prime[:, 0].float())
    phi = torch.deg2rad(prime[:, 1].float())
    return torch.stack(
        (torch.sin(theta) * torch.cos(phi), torch.sin(theta) * torch.sin(phi), torch.cos(theta)),
        dim=1,
    )


def particle_codes(ev_ids: np.ndarray) -> Tensor:
    prefixes = ev_ids.astype("S5")
    codes = np.full(len(ev_ids), -1, dtype=np.int64)
    codes[prefixes == b"muatm"] = 0
    codes[prefixes == b"nuatm"] = 1
    codes[prefixes == b"nue2_"] = 2
    return torch.from_numpy(codes)


class BatchedAngleDataset(Dataset):
    def __init__(
        self,
        path: str,
        split: str,
        events_per_batch: int,
        max_seq_len: int = 256,
        use_signal_norm: bool = False,
        max_events: int | None = None,
        center_time: bool = False,
    ) -> None:
        self.path = path
        self.split = split
        self.events_per_batch = events_per_batch
        self.max_seq_len = max_seq_len
        self.center_time = center_time
        self._file = None
        with h5py.File(path, "r") as h5:
            self.n_events = len(h5[f"{split}/ev_starts/data"]) - 1
            if max_events is not None:
                self.n_events = min(self.n_events, max_events)
            self.stored_mean = np.asarray(h5["norm_param/mean"], dtype=np.float32)
            self.stored_std = np.asarray(h5["norm_param/std"], dtype=np.float32)
            if use_signal_norm and "signal_norm_param" in h5:
                self.target_mean = np.asarray(h5["signal_norm_param/mean"], dtype=np.float32)
                self.target_std = np.asarray(h5["signal_norm_param/std"], dtype=np.float32)
            else:
                self.target_mean = self.stored_mean.copy()
                self.target_std = self.stored_std.copy()
        self.renormalize = not (
            np.allclose(self.stored_mean, self.target_mean)
            and np.allclose(self.stored_std, self.target_std)
        )

    @property
    def h5(self) -> h5py.File:
        if self._file is None:
            self._file = h5py.File(self.path, "r", swmr=True)
        return self._file

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_file"] = None
        return state

    def __len__(self) -> int:
        return math.ceil(self.n_events / self.events_per_batch)

    def __getitem__(self, batch_index: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        event_start = batch_index * self.events_per_batch
        event_end = min(event_start + self.events_per_batch, self.n_events)
        if event_start >= event_end:
            raise IndexError(batch_index)

        starts = np.asarray(
            self.h5[f"{self.split}/ev_starts/data"][event_start : event_end + 1],
            dtype=np.int64,
        )
        hit_start, hit_end = int(starts[0]), int(starts[-1])
        features = np.asarray(
            self.h5[f"{self.split}/data/data"][hit_start:hit_end], dtype=np.float32
        )
        if self.renormalize:
            features = (
                features * self.stored_std + self.stored_mean - self.target_mean
            ) / self.target_std

        rel_starts = starts - hit_start
        lengths = np.minimum(np.diff(rel_starts), self.max_seq_len).astype(np.int64)
        batch_size = event_end - event_start
        seq_len = int(lengths.max())
        x = np.zeros((batch_size, seq_len, features.shape[1]), dtype=np.float32)
        mask = np.zeros((batch_size, seq_len), dtype=bool)
        for i, length in enumerate(lengths):
            source_start = int(rel_starts[i])
            event_features = features[source_start : source_start + length].copy()
            if self.center_time:
                event_features[:, 1] -= event_features[:, 1].mean()
            x[i, :length] = event_features
            mask[i, :length] = True

        prime = torch.from_numpy(
            np.asarray(
                self.h5[f"{self.split}/prime_prty/data"][event_start:event_end],
                dtype=np.float32,
            )
        )
        ids = np.asarray(self.h5[f"{self.split}/ev_ids/data"][event_start:event_end])
        return (
            torch.from_numpy(x),
            direction_from_angles(prime),
            torch.from_numpy(mask),
            particle_codes(ids),
        )


def masked_mean(x: Tensor, mask: Tensor) -> Tensor:
    weights = mask.unsqueeze(-1).to(x.dtype)
    return (x * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def masked_max(x: Tensor, mask: Tensor) -> Tensor:
    return x.masked_fill(~mask.unsqueeze(-1), -torch.inf).amax(dim=1)


def engineered_features(x: Tensor, mask: Tensor) -> Tensor:
    mean = masked_mean(x, mask)
    centered = x[..., 1:5] - mean[:, None, 1:5]
    radius = torch.linalg.vector_norm(centered[..., 1:4], dim=-1, keepdim=True)
    return torch.cat((x, centered, radius), dim=-1)


def augment_batch(
    x: Tensor,
    target: Tensor,
    mask: Tensor,
    mean: Tensor,
    std: Tensor,
    args: argparse.Namespace,
) -> tuple[Tensor, Tensor, Tensor]:
    """Apply event-consistent physical augmentations in unnormalized units."""
    if not any(
        (
            args.rotate_z_prob,
            args.time_jitter_ns,
            args.event_time_jitter_ns,
            args.coord_jitter_m,
            args.charge_log_jitter,
            args.hit_dropout,
        )
    ):
        return x, target, mask

    raw = x * std + mean
    weights = mask.unsqueeze(-1).to(raw.dtype)
    counts = weights.sum(dim=1).clamp_min(1.0)

    if args.event_time_jitter_ns > 0:
        offset = torch.randn((len(raw), 1), device=raw.device, dtype=raw.dtype)
        raw[..., 1] += offset * args.event_time_jitter_ns
    if args.time_jitter_ns > 0:
        raw[..., 1] += torch.randn_like(raw[..., 1]) * args.time_jitter_ns
    if args.coord_jitter_m > 0:
        raw[..., 2:5] += torch.randn_like(raw[..., 2:5]) * args.coord_jitter_m
    if args.charge_log_jitter > 0:
        scale = torch.exp(
            torch.randn((len(raw), 1), device=raw.device, dtype=raw.dtype) * args.charge_log_jitter
        )
        raw[..., 0] *= scale

    if args.rotate_z_prob > 0:
        apply = torch.rand(len(raw), device=raw.device) < args.rotate_z_prob
        angle = torch.rand(len(raw), device=raw.device, dtype=raw.dtype) * (2 * math.pi)
        angle = torch.where(apply, angle, torch.zeros_like(angle))
        cosine, sine = torch.cos(angle), torch.sin(angle)
        xy_center = (raw[..., 2:4] * weights).sum(dim=1) / counts
        xy = raw[..., 2:4] - xy_center[:, None]
        x_rot = cosine[:, None] * xy[..., 0] - sine[:, None] * xy[..., 1]
        y_rot = sine[:, None] * xy[..., 0] + cosine[:, None] * xy[..., 1]
        raw[..., 2] = x_rot + xy_center[:, None, 0]
        raw[..., 3] = y_rot + xy_center[:, None, 1]
        target_x = cosine * target[:, 0] - sine * target[:, 1]
        target_y = sine * target[:, 0] + cosine * target[:, 1]
        target = torch.stack((target_x, target_y, target[:, 2]), dim=1)

    if args.hit_dropout > 0:
        dropped_mask = mask & (torch.rand(mask.shape, device=mask.device) >= args.hit_dropout)
        valid = dropped_mask.sum(dim=1) >= args.min_hits_after_dropout
        mask = torch.where(valid[:, None], dropped_mask, mask)

    raw = raw * weights
    return (raw - mean) / std, target, mask


class ResidualMLP(nn.Module):
    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, 2 * width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * width, width),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.net(x)


class DeepSetDirection(nn.Module):
    def __init__(self, hidden: int, layers: int, dropout: float, feature_mode: str) -> None:
        super().__init__()
        self.feature_mode = feature_mode
        in_features = 10 if feature_mode == "engineered" else 5
        self.embed = nn.Sequential(nn.Linear(in_features, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.blocks = nn.ModuleList([ResidualMLP(hidden, dropout) for _ in range(layers)])
        self.attention = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))
        self.head = nn.Sequential(
            nn.LayerNorm(3 * hidden),
            nn.Linear(3 * hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 3),
        )

    def forward(self, x: Tensor, mask: Tensor, return_features: bool = False):
        if self.feature_mode == "engineered":
            x = engineered_features(x, mask)
        h = self.embed(x)
        for block in self.blocks:
            h = block(h)
        logits = self.attention(h).squeeze(-1).masked_fill(~mask, -torch.inf)
        attention_pool = torch.sum(h * torch.softmax(logits, dim=1).unsqueeze(-1), dim=1)
        pooled = torch.cat((attention_pool, masked_mean(h, mask), masked_max(h, mask)), dim=1)
        direction = torch.nn.functional.normalize(self.head(pooled), dim=1)
        return (direction, pooled) if return_features else direction


class SetTransformerDirection(nn.Module):
    def __init__(
        self,
        hidden: int,
        layers: int,
        heads: int,
        ff_mult: int,
        dropout: float,
        feature_mode: str,
    ) -> None:
        super().__init__()
        self.feature_mode = feature_mode
        in_features = 10 if feature_mode == "engineered" else 5
        self.embed = nn.Sequential(nn.Linear(in_features, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.cls = nn.Parameter(torch.empty(1, 1, hidden))
        nn.init.normal_(self.cls, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=hidden * ff_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers, norm=nn.LayerNorm(hidden))
        self.head = nn.Sequential(
            nn.LayerNorm(2 * hidden),
            nn.Linear(2 * hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 3),
        )

    def forward(self, x: Tensor, mask: Tensor, return_features: bool = False):
        if self.feature_mode == "engineered":
            x = engineered_features(x, mask)
        h = self.embed(x)
        cls = self.cls.expand(len(h), -1, -1)
        h = torch.cat((cls, h), dim=1)
        full_mask = torch.cat(
            (torch.ones((len(mask), 1), dtype=torch.bool, device=mask.device), mask), dim=1
        )
        h = self.encoder(h, src_key_padding_mask=~full_mask)
        pooled = torch.cat((h[:, 0], masked_mean(h[:, 1:], mask)), dim=1)
        direction = torch.nn.functional.normalize(self.head(pooled), dim=1)
        return (direction, pooled) if return_features else direction


class PhysicsGraphBlock(nn.Module):
    """Directed message passing over a fixed, physically constructed hit graph."""

    def __init__(self, hidden: int, edge_features: int, dropout: float) -> None:
        super().__init__()
        self.node_norm = nn.LayerNorm(hidden)
        self.neighbor = nn.Linear(hidden, hidden, bias=False)
        self.edge = nn.Sequential(
            nn.Linear(edge_features, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.gate = nn.Sequential(
            nn.Linear(edge_features, hidden),
            nn.Sigmoid(),
        )
        self.update = nn.Sequential(
            nn.LayerNorm(2 * hidden),
            nn.Linear(2 * hidden, 2 * hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden, hidden),
        )

    def forward(
        self,
        h: Tensor,
        neighbor_index: Tensor,
        edge_features: Tensor,
        neighbor_mask: Tensor,
    ) -> Tensor:
        batch, nodes, neighbors = neighbor_index.shape
        normalized = self.node_norm(h)
        expanded = normalized[:, None].expand(-1, nodes, -1, -1)
        neighbor_h = torch.gather(
            expanded,
            2,
            neighbor_index[..., None].expand(-1, -1, -1, normalized.shape[-1]),
        )
        message = torch.nn.functional.gelu(self.neighbor(neighbor_h) + self.edge(edge_features))
        message = message * self.gate(edge_features) * neighbor_mask[..., None].to(message.dtype)
        denominator = neighbor_mask.sum(dim=2, keepdim=True).clamp_min(1).to(message.dtype)
        aggregate = message.sum(dim=2) / denominator
        return h + self.update(torch.cat((normalized, aggregate), dim=-1))


class PhysicsGraphDirection(nn.Module):
    """Hit graph with topology and edge features motivated by photon propagation."""

    def __init__(
        self,
        hidden: int,
        layers: int,
        dropout: float,
        feature_mode: str,
        graph_k: int,
        photon_speed_m_per_ns: float,
        minkowski_edge: bool,
    ) -> None:
        super().__init__()
        self.feature_mode = feature_mode
        self.graph_k = graph_k
        self.photon_speed_m_per_ns = photon_speed_m_per_ns
        self.minkowski_edge = minkowski_edge
        edge_feature_count = 11 if minkowski_edge else 10
        in_features = 10 if feature_mode == "engineered" else 5
        self.embed = nn.Sequential(nn.Linear(in_features, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.blocks = nn.ModuleList(
            [PhysicsGraphBlock(hidden, edge_feature_count, dropout) for _ in range(layers)]
        )
        self.attention = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))
        self.head = nn.Sequential(
            nn.LayerNorm(3 * hidden),
            nn.Linear(3 * hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 3),
        )
        # Persistent buffers make a graph checkpoint self-contained at evaluation time.
        self.register_buffer("feature_mean", torch.zeros(1, 1, 5))
        self.register_buffer("feature_std", torch.ones(1, 1, 5))

    def set_feature_norm(self, mean: np.ndarray | Tensor, std: np.ndarray | Tensor) -> None:
        self.feature_mean.copy_(
            torch.as_tensor(mean, device=self.feature_mean.device).view(1, 1, 5)
        )
        self.feature_std.copy_(torch.as_tensor(std, device=self.feature_std.device).view(1, 1, 5))

    def make_graph(self, x: Tensor, mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        # Graph construction is deterministic and has no trainable path. Keep it in
        # float32: timing residual ranking is less stable in fp16 for distant hits.
        with torch.no_grad(), torch.autocast(device_type="cuda", enabled=False):
            raw = x.float() * self.feature_std + self.feature_mean
            position = raw[..., 2:5]
            pair_position = position[:, None, :, :] - position[:, :, None, :]
            pair_distance = torch.linalg.vector_norm(pair_position, dim=-1)
            pair_dt = raw[:, None, :, 1] - raw[:, :, None, 1]
            propagation_residual_m = torch.abs(pair_dt) * self.photon_speed_m_per_ns - pair_distance
            score = pair_distance + 0.25 * torch.abs(propagation_residual_m)

            valid_pair = mask[:, :, None] & mask[:, None, :]
            identity = torch.eye(mask.shape[1], dtype=torch.bool, device=mask.device)[None]
            score = score.masked_fill(~valid_pair | identity, torch.inf)
            neighbors = min(self.graph_k, mask.shape[1])
            neighbor_index = torch.topk(score, neighbors, dim=2, largest=False).indices
            neighbor_mask = torch.gather(
                valid_pair,
                2,
                neighbor_index,
            ) & torch.isfinite(torch.gather(score, 2, neighbor_index))

            expanded_raw = raw[:, None].expand(-1, mask.shape[1], -1, -1)
            neighbor_raw = torch.gather(
                expanded_raw,
                2,
                neighbor_index[..., None].expand(-1, -1, -1, raw.shape[-1]),
            )
            delta = neighbor_raw - raw[:, :, None, :]
            delta_xyz = delta[..., 2:5]
            distance = torch.linalg.vector_norm(delta_xyz, dim=-1)
            transverse = torch.linalg.vector_norm(delta_xyz[..., :2], dim=-1)
            dt = delta[..., 1]
            travel_time = distance / self.photon_speed_m_per_ns
            causal_residual = torch.abs(dt) - travel_time
            apparent_velocity = dt / (torch.abs(dt) + travel_time + 5.0)
            same_string_affinity = torch.exp(-torch.square(transverse / 2.0))
            charge_contrast = torch.tanh((neighbor_raw[..., 0] - raw[:, :, None, 0]) / 20.0)
            edge_parts = [
                (dt / 100.0)[..., None],
                delta_xyz / 100.0,
                (distance / 100.0)[..., None],
                (transverse / 100.0)[..., None],
                (causal_residual / 100.0)[..., None],
                apparent_velocity[..., None],
                same_string_affinity[..., None],
                charge_contrast[..., None],
            ]
            if self.minkowski_edge:
                # s^2=(v_water*dt)^2-|dr|^2 is in m^2. The signed square root
                # preserves time-like/space-like side and is scaled by 100 m.
                minkowski_square_m2 = torch.square(self.photon_speed_m_per_ns * dt) - torch.square(
                    distance
                )
                signed_minkowski_distance = torch.sign(minkowski_square_m2) * torch.sqrt(
                    torch.abs(minkowski_square_m2) + 1e-6
                )
                edge_parts.append((signed_minkowski_distance / 100.0)[..., None])
            edge_features = torch.cat(edge_parts, dim=-1)
        return neighbor_index, edge_features, neighbor_mask

    def forward(self, x: Tensor, mask: Tensor, return_features: bool = False):
        neighbor_index, edge_features, neighbor_mask = self.make_graph(x, mask)
        node_features = engineered_features(x, mask) if self.feature_mode == "engineered" else x
        h = self.embed(node_features)
        for block in self.blocks:
            h = block(h, neighbor_index, edge_features, neighbor_mask)
        logits = self.attention(h).squeeze(-1).masked_fill(~mask, -torch.inf)
        attention_pool = torch.sum(h * torch.softmax(logits, dim=1).unsqueeze(-1), dim=1)
        pooled = torch.cat((attention_pool, masked_mean(h, mask), masked_max(h, mask)), dim=1)
        direction = torch.nn.functional.normalize(self.head(pooled), dim=1)
        return (direction, pooled) if return_features else direction


class PhysicsGraphTransformerDirection(nn.Module):
    """Winning global Transformer augmented by identity-initialized local physics messages."""

    def __init__(
        self,
        hidden: int,
        layers: int,
        heads: int,
        ff_mult: int,
        dropout: float,
        feature_mode: str,
        graph_layers: int,
        graph_k: int,
        photon_speed_m_per_ns: float,
        minkowski_edge: bool,
    ) -> None:
        super().__init__()
        self.feature_mode = feature_mode
        self.graph_k = graph_k
        self.photon_speed_m_per_ns = photon_speed_m_per_ns
        self.minkowski_edge = minkowski_edge
        in_features = 10 if feature_mode == "engineered" else 5
        edge_feature_count = 11 if minkowski_edge else 10
        # Names intentionally match SetTransformerDirection so its checkpoint can
        # initialize every global-model weight with strict=False.
        self.embed = nn.Sequential(nn.Linear(in_features, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.graph_blocks = nn.ModuleList(
            [PhysicsGraphBlock(hidden, edge_feature_count, dropout) for _ in range(graph_layers)]
        )
        for block in self.graph_blocks:
            nn.init.zeros_(block.update[-1].weight)
            nn.init.zeros_(block.update[-1].bias)
        self.cls = nn.Parameter(torch.empty(1, 1, hidden))
        nn.init.normal_(self.cls, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=hidden * ff_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers, norm=nn.LayerNorm(hidden))
        self.head = nn.Sequential(
            nn.LayerNorm(2 * hidden),
            nn.Linear(2 * hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 3),
        )
        self.register_buffer("feature_mean", torch.zeros(1, 1, 5))
        self.register_buffer("feature_std", torch.ones(1, 1, 5))

    def set_feature_norm(self, mean: np.ndarray | Tensor, std: np.ndarray | Tensor) -> None:
        self.feature_mean.copy_(
            torch.as_tensor(mean, device=self.feature_mean.device).view(1, 1, 5)
        )
        self.feature_std.copy_(torch.as_tensor(std, device=self.feature_std.device).view(1, 1, 5))

    def make_graph(self, x: Tensor, mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        return PhysicsGraphDirection.make_graph(self, x, mask)

    def forward(self, x: Tensor, mask: Tensor, return_features: bool = False):
        neighbor_index, edge_features, neighbor_mask = self.make_graph(x, mask)
        node_features = engineered_features(x, mask) if self.feature_mode == "engineered" else x
        h = self.embed(node_features)
        for block in self.graph_blocks:
            h = block(h, neighbor_index, edge_features, neighbor_mask)
        cls = self.cls.expand(len(h), -1, -1)
        h = torch.cat((cls, h), dim=1)
        full_mask = torch.cat(
            (torch.ones((len(mask), 1), dtype=torch.bool, device=mask.device), mask), dim=1
        )
        h = self.encoder(h, src_key_padding_mask=~full_mask)
        pooled = torch.cat((h[:, 0], masked_mean(h[:, 1:], mask)), dim=1)
        direction = torch.nn.functional.normalize(self.head(pooled), dim=1)
        return (direction, pooled) if return_features else direction


def build_model(args: argparse.Namespace) -> nn.Module:
    if args.model == "deepset":
        return DeepSetDirection(args.hidden, args.layers, args.dropout, args.feature_mode)
    if args.model == "physics_graph":
        return PhysicsGraphDirection(
            args.hidden,
            args.layers,
            args.dropout,
            args.feature_mode,
            args.graph_k,
            args.photon_speed,
            getattr(args, "minkowski_edge", False),
        )
    if args.model == "physics_graph_transformer":
        return PhysicsGraphTransformerDirection(
            args.hidden,
            args.layers,
            args.heads,
            args.ff_mult,
            args.dropout,
            args.feature_mode,
            args.graph_layers,
            args.graph_k,
            args.photon_speed,
            getattr(args, "minkowski_edge", False),
        )
    return SetTransformerDirection(
        args.hidden,
        args.layers,
        args.heads,
        args.ff_mult,
        args.dropout,
        args.feature_mode,
    )


class Lion(torch.optim.Optimizer):
    """Minimal decoupled-weight-decay Lion optimizer."""

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad
                state = self.state[parameter]
                if not state:
                    state["exp_avg"] = torch.zeros_like(parameter)
                exponential_average = state["exp_avg"]
                if group["weight_decay"]:
                    parameter.mul_(1.0 - group["lr"] * group["weight_decay"])
                update = exponential_average.mul(beta1).add(gradient, alpha=1.0 - beta1)
                parameter.add_(torch.sign(update), alpha=-group["lr"])
                exponential_average.mul_(beta2).add_(gradient, alpha=1.0 - beta2)
        return loss


def build_optimizer(args: argparse.Namespace, model: nn.Module, device: torch.device):
    common = dict(lr=args.lr, weight_decay=args.weight_decay)
    if args.optimizer == "radam":
        return torch.optim.RAdam(model.parameters(), **common)
    if args.optimizer == "nadam":
        return torch.optim.NAdam(model.parameters(), **common)
    if args.optimizer == "lion":
        return Lion(model.parameters(), **common)
    return torch.optim.AdamW(model.parameters(), **common, fused=device.type == "cuda")


def angular_errors_degrees(prediction: Tensor, target: Tensor) -> Tensor:
    dot = torch.sum(prediction * target, dim=1).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(dot))


def loss_function(
    prediction: Tensor,
    target: Tensor,
    name: str,
    power: float,
    weights: Tensor | None = None,
    cap_degrees: float = 10.0,
) -> Tensor:
    dot = torch.sum(prediction * target, dim=1).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    robust_weights = None
    if name == "mae":
        elements = torch.abs(prediction - target).mean(dim=1)
    elif name == "mse":
        elements = torch.square(prediction - target).mean(dim=1)
    elif name == "cosine":
        elements = 1.0 - dot
    elif name == "chord":
        chord = torch.sqrt(torch.sum((prediction - target) ** 2, dim=1) + 1e-8)
        elements = chord.pow(power)
    elif name == "trimmed_angular":
        angle = torch.acos(dot)
        cap = math.radians(cap_degrees)
        transition = max(math.radians(2.0), 0.2 * cap)
        # Do not let the model game its own sample weights; only the angular
        # objective carries gradients. This is a smooth trimmed-risk estimator.
        robust_weights = torch.sigmoid((cap - angle.detach()) / transition)
        elements = angle
    elif name == "angular_huber":
        angle = torch.acos(dot)
        delta = math.radians(cap_degrees)
        elements = torch.where(
            angle < delta,
            0.5 * angle.square() / delta,
            angle - 0.5 * delta,
        )
    else:
        elements = torch.acos(dot).pow(power)
    if robust_weights is not None:
        weights = robust_weights if weights is None else weights * robust_weights
    if weights is None:
        return elements.mean()
    return torch.sum(elements * weights) / weights.sum().clamp_min(1e-8)


def infinite_batches(loader: DataLoader) -> Iterator:
    while True:
        yield from loader


def metric_summary(errors: np.ndarray, particle: np.ndarray) -> dict[str, float]:
    result = {
        "count": int(len(errors)),
        "mean": float(np.mean(errors)),
        "q50": float(np.quantile(errors, 0.50)),
        "q68": float(np.quantile(errors, 0.68)),
        "q90": float(np.quantile(errors, 0.90)),
    }
    for code, name in PARTICLE_NAMES.items():
        subset = errors[particle == code]
        if len(subset):
            result[f"{name}_count"] = int(len(subset))
            result[f"{name}_q50"] = float(np.quantile(subset, 0.50))
            result[f"{name}_q68"] = float(np.quantile(subset, 0.68))
            result[f"{name}_q90"] = float(np.quantile(subset, 0.90))
    return result


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    error_parts = []
    particle_parts = []
    for x, target, mask, particle in loader:
        x = x.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            prediction = model(x, mask)
        error_parts.append(angular_errors_degrees(prediction.float(), target).cpu().numpy())
        particle_parts.append(particle.numpy())
    return metric_summary(np.concatenate(error_parts), np.concatenate(particle_parts))


def make_loader(
    dataset: Dataset,
    workers: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=None,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        prefetch_factor=2 if workers > 0 else None,
        generator=generator,
    )


def append_csv(path: Path, row: dict) -> None:
    exists = path.exists()
    with path.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    step: int,
    best_q68: float,
    best_q50: float,
    args: argparse.Namespace,
    metrics: dict,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "step": step,
            "best_q68": best_q68,
            "best_q50": best_q50,
            "args": vars(args),
            "metrics": metrics,
        },
        path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--model",
        choices=("transformer", "deepset", "physics_graph", "physics_graph_transformer"),
        default="transformer",
    )
    parser.add_argument("--feature-mode", choices=("basic", "engineered"), default="engineered")
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--ff-mult", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--graph-k", type=int, default=12)
    parser.add_argument("--graph-layers", type=int, default=2)
    parser.add_argument("--photon-speed", type=float, default=0.225)
    parser.add_argument("--minkowski-edge", action="store_true")
    parser.add_argument(
        "--loss",
        choices=(
            "angular",
            "angular_huber",
            "trimmed_angular",
            "chord",
            "cosine",
            "mae",
            "mse",
        ),
        default="chord",
    )
    parser.add_argument("--loss-power", type=float, default=1.0)
    parser.add_argument("--loss-cap-deg", type=float, default=10.0)
    parser.add_argument(
        "--particle-loss-weights",
        type=float,
        nargs=3,
        default=(1.0, 1.0, 1.0),
        metavar=("MUATM", "NUATM", "NUE2"),
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--max-train-events", type=int, default=None)
    parser.add_argument("--max-val-events", type=int, default=50_000)
    parser.add_argument("--max-test-events", type=int, default=None)
    parser.add_argument("--use-signal-norm", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=20_000)
    parser.add_argument("--val-every", type=int, default=1_000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--optimizer", choices=("adamw", "radam", "nadam", "lion"), default="adamw")
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=1_000)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rotate-z-prob", type=float, default=0.0)
    parser.add_argument("--center-time", action="store_true")
    parser.add_argument("--time-jitter-ns", type=float, default=0.0)
    parser.add_argument("--event-time-jitter-ns", type=float, default=0.0)
    parser.add_argument("--coord-jitter-m", type=float, default=0.0)
    parser.add_argument("--charge-log-jitter", type=float, default=0.0)
    parser.add_argument("--hit-dropout", type=float, default=0.0)
    parser.add_argument("--min-hits-after-dropout", type=int, default=8)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--eval-split", choices=("val", "test"), default="test")
    parser.add_argument("--checkpoint", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True) + "\n")

    seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args).to(device)
    if args.init_checkpoint:
        initial = torch.load(args.init_checkpoint, map_location=device)
        state_dict = initial["model"] if "model" in initial else initial
        strict = args.model != "physics_graph_transformer"
        load_result = model.load_state_dict(state_dict, strict=strict)
        if not strict:
            print(
                f"non-strict initialization missing={load_result.missing_keys} "
                f"unexpected={load_result.unexpected_keys}",
                flush=True,
            )
        print(f"initialized model weights from {args.init_checkpoint}", flush=True)
    print(f"device={device}; parameters={sum(p.numel() for p in model.parameters()):,}", flush=True)

    if args.eval_only:
        checkpoint_path = args.checkpoint or args.resume
        if not checkpoint_path:
            raise ValueError("--eval-only requires --checkpoint")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model"] if "model" in checkpoint else checkpoint)
        max_events = args.max_test_events if args.eval_split == "test" else args.max_val_events
        dataset = BatchedAngleDataset(
            args.data,
            args.eval_split,
            args.batch_size,
            args.max_seq_len,
            args.use_signal_norm,
            max_events,
            args.center_time,
        )
        metrics = evaluate(model, make_loader(dataset, args.workers, False, args.seed), device)
        print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)
        (output_dir / f"{args.eval_split}_metrics.json").write_text(
            json.dumps(metrics, indent=2, sort_keys=True) + "\n"
        )
        return

    train_dataset = BatchedAngleDataset(
        args.data,
        "train",
        args.batch_size,
        args.max_seq_len,
        args.use_signal_norm,
        args.max_train_events,
        args.center_time,
    )
    val_dataset = BatchedAngleDataset(
        args.data,
        "val",
        args.batch_size,
        args.max_seq_len,
        args.use_signal_norm,
        args.max_val_events,
        args.center_time,
    )
    if hasattr(model, "set_feature_norm"):
        model.set_feature_norm(train_dataset.target_mean, train_dataset.target_std)
    train_loader = make_loader(train_dataset, args.workers, True, args.seed)
    val_loader = make_loader(val_dataset, args.workers, False, args.seed)
    train_batches = infinite_batches(train_loader)
    feature_mean = torch.as_tensor(train_dataset.target_mean, device=device).view(1, 1, -1)
    feature_std = torch.as_tensor(train_dataset.target_std, device=device).view(1, 1, -1)
    particle_loss_weights = torch.as_tensor(args.particle_loss_weights, device=device)

    optimizer = build_optimizer(args, model, device)

    def lr_lambda(step: int) -> float:
        if step < args.warmup_steps:
            return max((step + 1) / max(args.warmup_steps, 1), args.min_lr / args.lr)
        progress = (step - args.warmup_steps) / max(args.max_steps - args.warmup_steps, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return args.min_lr / args.lr + (1.0 - args.min_lr / args.lr) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    start_step = 0
    best_q68 = float("inf")
    best_q50 = float("inf")

    resume_path = args.resume
    if resume_path:
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_step = int(checkpoint["step"])
        best_q68 = float(checkpoint.get("best_q68", best_q68))
        best_q50 = float(checkpoint.get("best_q50", best_q50))
        print(f"resumed {resume_path} at step {start_step}", flush=True)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    running_loss = 0.0
    running_count = 0
    last_log_time = time.time()
    validations_without_improvement = 0
    metrics_csv = output_dir / "metrics.csv"

    for step in range(start_step + 1, args.max_steps + 1):
        x, target, mask, particle = next(train_batches)
        x = x.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        particle = particle.to(device, non_blocking=True)
        x, target, mask = augment_batch(x, target, mask, feature_mean, feature_std, args)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            prediction = model(x, mask)
            example_weights = particle_loss_weights[particle.clamp_min(0)]
            loss = loss_function(
                prediction,
                target,
                args.loss,
                args.loss_power,
                example_weights,
                args.loss_cap_deg,
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        running_loss += float(loss.detach())
        running_count += 1
        if step % args.log_every == 0:
            now = time.time()
            elapsed = now - last_log_time
            print(
                f"step={step} loss={running_loss / running_count:.6f} "
                f"lr={optimizer.param_groups[0]['lr']:.3e} steps/s={running_count / elapsed:.2f}",
                flush=True,
            )
            running_loss = 0.0
            running_count = 0
            last_log_time = now

        if step % args.val_every == 0 or step == args.max_steps:
            metrics = evaluate(model, val_loader, device)
            metrics["step"] = step
            metrics["lr"] = optimizer.param_groups[0]["lr"]
            print("validation " + json.dumps(metrics, sort_keys=True), flush=True)
            append_csv(metrics_csv, metrics)

            improved_q68 = metrics["q68"] < best_q68
            improved_q50 = metrics["q50"] < best_q50
            if improved_q68:
                best_q68 = metrics["q68"]
                validations_without_improvement = 0
                save_checkpoint(
                    output_dir / "best_q68.pt",
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    step,
                    best_q68,
                    min(best_q50, metrics["q50"]),
                    args,
                    metrics,
                )
            else:
                validations_without_improvement += 1
            if improved_q50:
                best_q50 = metrics["q50"]
                save_checkpoint(
                    output_dir / "best_q50.pt",
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    step,
                    best_q68,
                    best_q50,
                    args,
                    metrics,
                )
            save_checkpoint(
                output_dir / "last.pt",
                model,
                optimizer,
                scheduler,
                scaler,
                step,
                best_q68,
                best_q50,
                args,
                metrics,
            )
            model.train()
            if validations_without_improvement >= args.patience:
                print(
                    f"early stopping after {validations_without_improvement} validations",
                    flush=True,
                )
                break


if __name__ == "__main__":
    main()
