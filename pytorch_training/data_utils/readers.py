import logging
import math
from typing import Any

import h5py as h5
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .preprocessors import BaseGraphPreprocessor, BasePreprocessor

MAX_SEQ_LEN = 256


class BaikalDataset(Dataset):
    def __init__(
        self,
        data_file: str,
        split_type: str,
        batch_size: int,
        preprocessor: BasePreprocessor | BaseGraphPreprocessor,
        set_tres_stats: bool = False,
        is_graph: bool = False,
        events_amount: int | None = None,
        renorm_params: tuple | None = None,
        *args,
        **kwargs,
    ) -> None:
        self.path_to_data_file = data_file
        self.hfile = h5.File(data_file, "r")
        self.split_type = split_type
        self.batch_size = batch_size
        self.is_graph = is_graph
        self.preprocessor = preprocessor
        self.renorm_params = renorm_params

        print(f"Loading {data_file}, keys: {list(self.hfile.keys())}")

        try:
            self.events_amount = self.hfile[f"{split_type}/ev_starts/data"].shape[0] - 1
        except KeyError:
            self.events_amount = self.hfile[f"{split_type}/data"].shape[0] - 1

        if events_amount is not None and split_type == "train":
            self.events_amount = min(self.events_amount, events_amount)

        if set_tres_stats:
            self._set_tres_stats()

    def _set_tres_stats(self):
        logging.info(f"Computing tres stats for {self.split_type}...")
        tres_data = np.array(self.hfile["train/t_res/data"])[: min(self.events_amount, 100000)]
        self.preprocessor.set_stats(tres_data.mean(), tres_data.std())
        logging.info("Done")

    def __len__(self) -> int:
        if self.is_graph:
            return self.events_amount
        if self.events_amount == 0:
            return 0
        return (self.events_amount + self.batch_size - 1) // self.batch_size

    def _get_batch_indices(self, idx: int) -> tuple[int, int]:
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds for dataset of length {len(self)}")
        start = idx * self.batch_size
        end = min((idx + 1) * self.batch_size, self.events_amount)
        return start, end

    def _load_event_data(self, batch_start: int, batch_end: int) -> dict[str, Any]:
        event_starts = self.hfile[f"{self.split_type}/ev_starts/data"][batch_start : batch_end + 1]
        global_start, global_end = event_starts[0], event_starts[-1]

        data = torch.tensor(self.hfile[f"{self.split_type}/data/data"][global_start:global_end])
        if self.renorm_params is not None:
            src_mean, src_std, dst_mean, dst_std = self.renorm_params
            data = (data.float() * src_std + src_mean - dst_mean) / dst_std

        return {
            "event_starts": event_starts,
            "global_start": global_start,
            "global_end": global_end,
            "data": data,
        }

    def _collate(
        self,
        event_starts: np.ndarray,
        raw_data_x: Tensor,
        raw_data_y: Tensor | None = None,
        max_length: int = MAX_SEQ_LEN,
    ) -> tuple[Tensor, Tensor | None, Tensor]:
        batch_x, batch_y, seq_lengths = [], [], []
        global_start = event_starts[0]

        for i in range(len(event_starts) - 1):
            start = event_starts[i] - global_start
            end = event_starts[i + 1] - global_start
            length = min(end - start, max_length)

            batch_x.append(raw_data_x[start : start + length])
            seq_lengths.append(length)

            if raw_data_y is not None:
                y = raw_data_y[start : start + length]
                if not isinstance(y, Tensor):
                    y = torch.tensor(y)
                batch_y.append(y)

        data_x = torch.nn.utils.rnn.pad_sequence(batch_x, batch_first=True, padding_value=0)
        data_y = (
            torch.nn.utils.rnn.pad_sequence(batch_y, batch_first=True, padding_value=0)
            if batch_y
            else None
        )

        mask = torch.zeros((len(seq_lengths), data_x.shape[1]), dtype=torch.bool)
        for i, length in enumerate(seq_lengths):
            mask[i, :length] = True

        return data_x, data_y, mask

    def __getitem__(self, idx: int):
        batch_start, batch_end = self._get_batch_indices(idx)
        info = self._load_event_data(batch_start, batch_end)

        labels = torch.tensor(
            self.hfile[f"{self.split_type}/labels/data"][info["global_start"] : info["global_end"]]
        )
        tres = torch.tensor(
            self.hfile[f"{self.split_type}/t_res/data"][info["global_start"] : info["global_end"]]
        )

        raw_data = torch.cat([info["data"], tres.unsqueeze(-1)], dim=-1)
        data_x, data_y, mask = self._collate(info["event_starts"], raw_data, labels)
        data_x, t_res = data_x[:, :, :-1], data_x[:, :, -1]

        return self.preprocessor(data_x, data_y, t_res, mask)


class BaikalDatasetSingle(BaikalDataset):
    def __getitem__(self, idx: int):
        start, end = self.hfile[f"{self.split_type}/ev_starts/data"][idx : idx + 2]
        data = torch.tensor(self.hfile[f"{self.split_type}/data/data"][start:end])
        labels = torch.tensor(self.hfile[f"{self.split_type}/labels/data"][start:end]).long()
        return self.preprocessor(data, labels)


class BaikalDatasetTres(BaikalDataset):
    def __getitem__(self, idx: int):
        batch_start, batch_end = self._get_batch_indices(idx)
        info = self._load_event_data(batch_start, batch_end)

        tres = torch.tensor(
            self.hfile[f"{self.split_type}/t_res/data"][info["global_start"] : info["global_end"]]
        )
        raw_data = torch.cat([info["data"], tres.unsqueeze(-1)], dim=-1)

        data, _, mask = self._collate(info["event_starts"], raw_data)
        data_x, tres_flat = data[:, :, :-1], data[:, :, -1].reshape(-1)

        return self.preprocessor(data_x, tres_flat, mask)


class BaikalDatasetTresSingle(BaikalDataset):
    def __getitem__(self, idx: int):
        start, end = self.hfile[f"{self.split_type}/ev_starts/data"][idx : idx + 2]
        data_x = torch.tensor(
            self.hfile[f"{self.split_type}/data/data"][start:end], dtype=torch.float32
        )
        data_y = torch.tensor(self.tres_data[start:end], dtype=torch.float32)
        return self.preprocessor(data_x, data_y)


class BaikalDatasetTrackCascade(BaikalDataset):
    def __getitem__(self, idx: int):
        batch_start, batch_end = self._get_batch_indices(idx)
        info = self._load_event_data(batch_start, batch_end)

        tres = torch.tensor(
            self.hfile[f"{self.split_type}/t_res/data"][info["global_start"] : info["global_end"]]
        )
        labels = self.hfile[f"{self.split_type}/labels/data"][
            info["global_start"] : info["global_end"]
        ]

        raw_data = torch.cat([info["data"], tres.unsqueeze(-1)], dim=-1)
        data_x, data_y, mask = self._collate(info["event_starts"], raw_data, labels)
        tres = data_x[:, :, -1]
        data_x = data_x[:, :, :-1]

        return self.preprocessor(data_x, data_y, tres, mask)


class BaikalDatasetTrackCascadeSingle(BaikalDataset):
    def __getitem__(self, idx: int):
        start, end = self.hfile[f"{self.split_type}/ev_starts/data"][idx : idx + 2]
        data = torch.tensor(self.hfile[f"{self.split_type}/data/data"][start:end])
        labels = torch.tensor(self.hfile[f"{self.split_type}/labels/data"][start:end])
        return self.preprocessor(data, labels)


class BaikalDatasetAngles(BaikalDataset):
    def __getitem__(self, idx: int):
        batch_start, batch_end = self._get_batch_indices(idx)
        info = self._load_event_data(batch_start, batch_end)

        prime_prty = torch.tensor(
            self.hfile[f"{self.split_type}/prime_prty/data"][batch_start:batch_end]
        )
        hits_labels = torch.tensor(
            self.hfile[f"{self.split_type}/labels/data"][info["global_start"] : info["global_end"]]
        )
        tres = torch.tensor(
            self.hfile[f"{self.split_type}/t_res/data"][info["global_start"] : info["global_end"]]
        )

        raw_data = torch.cat([info["data"], hits_labels.unsqueeze(-1), tres.unsqueeze(-1)], dim=-1)
        data_x, _, mask = self._collate(info["event_starts"], raw_data)
        data_x, labels, tres = data_x[:, :, :-2], data_x[:, :, -2], data_x[:, :, -1]

        return self.preprocessor(data_x, prime_prty, mask, labels, tres)


class BaikalDatasetAnglesOld(BaikalDataset):
    def __getitem__(self, idx: int):
        batch_start = idx * self.batch_size
        batch_end = min((idx + 1) * self.batch_size, len(self.hfile[f"{self.split_type}/data"]) - 1)

        angles = torch.tensor(self.hfile[f"{self.split_type}/ev_chars"][batch_start:batch_end])

        batch_x, seq_lengths = [], []
        for i in range(batch_start, batch_end):
            mask_len = int(np.sum(self.hfile[f"{self.split_type}/mask"][i], axis=-1))
            seq_lengths.append(mask_len)
            data = self.hfile[f"{self.split_type}/data"][i][:mask_len]
            batch_x.append(torch.tensor(data, dtype=torch.float32))

        data_x = torch.nn.utils.rnn.pad_sequence(batch_x, batch_first=True, padding_value=0)
        mask = torch.zeros((len(seq_lengths), data_x.shape[1]), dtype=torch.bool)
        for i, length in enumerate(seq_lengths):
            mask[i, :length] = True

        return self.preprocessor(data_x, angles, mask)


class BaikalDatasetAnglesSingle(BaikalDataset):
    def __getitem__(self, idx: int):
        start, end = self.hfile[f"{self.split_type}/ev_starts/data"][idx : idx + 2]
        data = np.array(self.hfile[f"{self.split_type}/data/data"][start:end])
        theta, phi = self.hfile[f"{self.split_type}/prime_prty/data"][idx][:2]

        theta = float(theta) * math.pi / 180
        phi = float(phi) * math.pi / 180
        vec = torch.tensor(
            [
                math.sin(theta) * math.cos(phi),
                math.sin(theta) * math.sin(phi),
                math.cos(theta),
            ],
            dtype=torch.float32,
        )
        vec = vec / vec.norm()

        data_x = torch.tensor(data, dtype=torch.float32)
        return self.preprocessor(data_x, vec)


class BaikalDatasetAnglesOldSingle(BaikalDataset):
    def __getitem__(self, idx: int):
        angles = torch.tensor(self.hfile[f"{self.split_type}/ev_chars"][idx])
        mask_len = int(np.sum(self.hfile[f"{self.split_type}/mask"][idx], axis=-1))
        data_x = torch.tensor(self.hfile[f"{self.split_type}/data"][idx][:mask_len])
        return self.preprocessor(data_x, angles)


class BaikalDatasetAnglesAndTrackCascade(BaikalDataset):
    def __getitem__(self, idx: int):
        batch_start, batch_end = self._get_batch_indices(idx)
        info = self._load_event_data(batch_start, batch_end)

        prime_prty = torch.tensor(
            self.hfile[f"{self.split_type}/prime_prty/data"][batch_start:batch_end]
        )
        hits_labels = torch.tensor(
            self.hfile[f"{self.split_type}/labels/data"][info["global_start"] : info["global_end"]]
        )
        tres = torch.tensor(
            self.hfile[f"{self.split_type}/t_res/data"][info["global_start"] : info["global_end"]]
        )

        raw_data = torch.cat([info["data"], hits_labels.unsqueeze(-1), tres.unsqueeze(-1)], dim=-1)
        data_x, _, mask = self._collate(info["event_starts"], raw_data)
        data_x, labels, tres = data_x[:, :, :-2], data_x[:, :, -2], data_x[:, :, -1]

        return self.preprocessor(data_x, prime_prty, mask, labels, tres)


class BaikalDatasetNoLabels(BaikalDataset):
    def __init__(self, *args, **kwargs):
        kwargs.pop("set_tres_stats", None)
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx: int):
        batch_start, batch_end = self._get_batch_indices(idx)
        info = self._load_event_data(batch_start, batch_end)

        data_x, _, mask = self._collate(info["event_starts"], info["data"])
        return self.preprocessor(data_x, mask)


class BaikalDatasetDirection(BaikalDataset):
    def __getitem__(self, idx: int):
        batch_start, batch_end = self._get_batch_indices(idx)
        info = self._load_event_data(batch_start, batch_end)

        angles = torch.tensor(
            self.hfile[f"{self.split_type}/prime_prty/data"][batch_start:batch_end, :2]
        )
        points = torch.tensor(
            self.hfile[f"{self.split_type}/muons_prty/individ_coords_norm/data"][
                batch_start:batch_end
            ]
        )
        hits_labels = torch.tensor(
            self.hfile[f"{self.split_type}/labels/data"][info["global_start"] : info["global_end"]]
        )
        tres = torch.tensor(
            self.hfile[f"{self.split_type}/t_res/data"][info["global_start"] : info["global_end"]]
        )

        raw_data = torch.cat([info["data"], hits_labels.unsqueeze(-1), tres.unsqueeze(-1)], dim=-1)
        data_x, _, mask = self._collate(info["event_starts"], raw_data)
        data_x = data_x[:, :, :-2]
        data_y = torch.cat([angles, points], dim=-1)

        return self.preprocessor(data_x, data_y, mask)


class BaikalDatasetEnergy(BaikalDataset):
    def __getitem__(self, idx: int):
        batch_start, batch_end = self._get_batch_indices(idx)
        info = self._load_event_data(batch_start, batch_end)

        prime_prty = torch.tensor(
            self.hfile[f"{self.split_type}/prime_prty/data"][batch_start:batch_end]
        )
        energy = prime_prty[:, 2]

        hits_labels = torch.tensor(
            self.hfile[f"{self.split_type}/labels/data"][info["global_start"] : info["global_end"]]
        )
        tres = torch.tensor(
            self.hfile[f"{self.split_type}/t_res/data"][info["global_start"] : info["global_end"]]
        )

        raw_data = torch.cat([info["data"], hits_labels.unsqueeze(-1), tres.unsqueeze(-1)], dim=-1)
        data_x, _, mask = self._collate(info["event_starts"], raw_data)
        data_x, labels, tres = data_x[:, :, :-2], data_x[:, :, -2], data_x[:, :, -1]

        return self.preprocessor(data_x, energy, mask, labels=labels, tres=tres)
