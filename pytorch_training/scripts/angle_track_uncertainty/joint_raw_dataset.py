#!/usr/bin/env python3
"""Pre-batched raw-hit events selected by the audited joint-training index."""

from __future__ import annotations

import math
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from train_angle_signal import direction_from_angles, particle_codes


class IndexedRawBatchDataset(Dataset):
    def __init__(
        self,
        source: Path,
        index: Path,
        split: str,
        model_norm_source: Path,
        batch_events: int = 128,
        max_seq_len: int = 256,
    ) -> None:
        if split not in ("train", "val", "test", "domain_mc", "exp"):
            raise ValueError(f"Unsupported split: {split}")
        if min(batch_events, max_seq_len) < 1:
            raise ValueError("Batch size and sequence length must be positive")
        self.source = str(source)
        self.split = split
        self.h5_split = "test" if split == "exp" else "val" if split == "domain_mc" else split
        self.is_experiment = split == "exp"
        self.batch_events = batch_events
        self.max_seq_len = max_seq_len
        self._file: h5py.File | None = None
        with np.load(index) as selection:
            self.indices = np.asarray(selection[f"{split}_raw_index"], dtype=np.int64)
            self.ids = np.asarray(selection[f"{split}_ids"])
            if self.is_experiment:
                self.point_m = np.zeros((len(self.indices), 3), dtype=np.float32)
                self.point_valid = np.zeros(len(self.indices), dtype=bool)
            else:
                self.point_m = np.asarray(selection[f"{split}_point_m"], dtype=np.float32)
                self.point_valid = np.asarray(selection[f"{split}_point_valid"], dtype=bool)
        if not np.all(self.indices[1:] >= self.indices[:-1]):
            raise ValueError("Selected raw rows must be sorted")
        if len(np.unique(self.ids)) != len(self.ids):
            raise ValueError("Event IDs in joint index must be unique")
        with h5py.File(source, "r") as h5:
            self.source_mean = np.asarray(h5["norm_param/mean"], dtype=np.float32)
            self.source_std = np.asarray(h5["norm_param/std"], dtype=np.float32)
            n_events = len(h5[f"{self.h5_split}/ev_starts/data"]) - 1
        if len(self.indices) and (self.indices[0] < 0 or self.indices[-1] >= n_events):
            raise ValueError("Joint index contains an out-of-range raw row")
        with h5py.File(model_norm_source, "r") as h5:
            self.model_mean = np.asarray(h5["norm_param/mean"], dtype=np.float32)
            self.model_std = np.asarray(h5["norm_param/std"], dtype=np.float32)
        self.particle = (
            particle_codes(self.ids)
            if not self.is_experiment
            else torch.full((len(self.ids),), -1, dtype=torch.int64)
        )

    @property
    def h5(self) -> h5py.File:
        if self._file is None:
            self._file = h5py.File(self.source, "r", swmr=True)
        return self._file

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_file"] = None
        return state

    def __len__(self) -> int:
        return math.ceil(len(self.indices) / self.batch_events)

    def __getitem__(self, batch_index: int):
        begin = batch_index * self.batch_events
        end = min(begin + self.batch_events, len(self.indices))
        if begin >= end:
            raise IndexError(batch_index)
        indices = self.indices[begin:end]
        first, last = int(indices[0]), int(indices[-1])
        group = self.h5[self.h5_split]
        starts = np.asarray(group["ev_starts/data"][first : last + 2], dtype=np.int64)
        hit_begin, hit_end = int(starts[0]), int(starts[-1])
        features = np.asarray(group["data/data"][hit_begin:hit_end], dtype=np.float32)
        raw = features * self.source_std + self.source_mean
        normalized = (raw - self.model_mean) / self.model_std
        labels = None if self.is_experiment else np.asarray(group["labels/data"][hit_begin:hit_end])
        prime = (
            None if self.is_experiment else np.asarray(group["prime_prty/data"][first : last + 1])
        )
        lengths = np.minimum(
            starts[indices - first + 1] - starts[indices - first], self.max_seq_len
        )
        seq_len = int(lengths.max())
        x = np.zeros((len(indices), seq_len, 5), dtype=np.float32)
        mask = np.zeros((len(indices), seq_len), dtype=bool)
        signal = np.zeros((len(indices), seq_len), dtype=np.float32)
        for row, source_index in enumerate(indices):
            a = int(starts[source_index - first] - hit_begin)
            b = int(starts[source_index - first + 1] - hit_begin)
            length = int(lengths[row])
            take = np.arange(a, b)
            if b - a > self.max_seq_len:
                take = take[np.linspace(0, b - a - 1, self.max_seq_len).astype(np.int64)]
            x[row, :length] = normalized[take]
            mask[row, :length] = True
            if labels is not None:
                signal[row, :length] = labels[take] != 0
        if prime is None:
            direction = torch.zeros((len(indices), 3), dtype=torch.float32)
        else:
            direction = direction_from_angles(torch.from_numpy(prime[indices - first, :2].copy()))
        return (
            torch.from_numpy(x),
            torch.from_numpy(mask),
            torch.from_numpy(signal),
            direction,
            torch.from_numpy(self.point_m[begin:end].copy()),
            torch.from_numpy(self.point_valid[begin:end].copy()),
            self.particle[begin:end],
        )
