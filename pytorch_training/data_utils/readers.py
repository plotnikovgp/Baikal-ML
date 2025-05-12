import numpy as np
import h5py as h5
import torch
from torch.utils.data import Dataset
import logging
from .preprocessors import BasePreprocessor, BaseGraphPreprocessor
import math
import typing as tp

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
        events_amount: int = None,
        *args,
        **kwargs,
    ) -> None:
        """
        Args:
            data_file (str): path to .h5 file
            split_type (str): train/val/test
        """
        self.path_to_data_file = data_file
        self.hfile = h5.File(data_file, "r")
        self.split_type = split_type
        # NB: only for train dataset
        print(data_file, self.hfile.keys())
        try:
            self.events_amount = (
                self.hfile[self.split_type + "/ev_starts/data"].shape[0] - 1
            )
        except KeyError:
            self.events_amount = (
                self.hfile[self.split_type + "/data"].shape[0] - 1
            )  # old angles dataset

        if events_amount is not None and split_type == "train":
            self.events_amount = min(self.events_amount, events_amount)

        self.preprocessor = preprocessor
        self.batch_size = batch_size
        self.is_graph = is_graph

        if set_tres_stats:
            logging.info(f"counting tres mean and std for {self.split_type}...")
            self.tres_data = np.array(self.hfile[self.split_type + "/t_res/data"])
            tres_mean = self.tres_data.mean()
            tres_std = self.tres_data.std()
            self.preprocessor.set_stats(tres_mean, tres_std)
            logging.info("finished")

    def _collate(
        self,
        event_starts,
        raw_data_x,
        raw_data_y=None,
        max_length=MAX_SEQ_LEN,
        pad_y=False,
    ):
        batch_x = []
        batch_y = []
        seq_lengths = []
        global_start = event_starts[0]
        for i in range(len(event_starts) - 1):
            start = event_starts[i] - global_start
            end = event_starts[i + 1] - global_start
            length = min(end - start, max_length)
            x = raw_data_x[start : start + length]
            batch_x.append(x)
            seq_lengths.append(length)
            if raw_data_y is not None:
                y = torch.tensor(raw_data_y[start : start + length])
                batch_y.append(y)

        data_x = torch.nn.utils.rnn.pad_sequence(
            batch_x, batch_first=True, padding_value=0
        )
        data_y = (
            torch.nn.utils.rnn.pad_sequence(batch_y, batch_first=True, padding_value=0)
            if batch_y
            else None
        )
        mask = torch.zeros((len(seq_lengths), data_x.shape[1]), dtype=torch.bool)
        for i, length in enumerate(seq_lengths):
            mask[i, :length] = 1
        return data_x, data_y, mask

    def __len__(self):
        return (
            self.events_amount // self.batch_size
            if not self.is_graph
            else self.events_amount
        )

    def __getitem__(self, idx):
        batch_start_idx = idx * self.batch_size
        batch_end_idx = min(
            (idx + 1) * self.batch_size,
            len(self.hfile[self.split_type + "/ev_starts/data"]) - 1,
        )
        event_starts = self.hfile[self.split_type + "/ev_starts/data"][
            batch_start_idx : batch_end_idx + 1
        ]

        global_start = event_starts[0]
        global_end = event_starts[-1]
        raw_data = self.hfile[self.split_type + "/data/data"][global_start:global_end]
        labels = self.hfile[self.split_type + "/labels/data"][global_start:global_end]

        data_x, data_y, mask = self._collate(event_starts, raw_data, labels, pad_y=True)
        return self.preprocessor(data_x, data_y, mask)


class BaikalDatasetTres(BaikalDataset):
    def __getitem__(self, idx):
        batch_start_idx = idx * self.batch_size
        batch_end_idx = min(
            (idx + 1) * self.batch_size,
            len(self.hfile[self.split_type + "/ev_starts/data"]) - 1,
        )
        event_starts = self.hfile[self.split_type + "/ev_starts/data"][
            batch_start_idx : batch_end_idx + 1
        ]

        global_start = event_starts[0]
        global_end = event_starts[-1]
        raw_data = self.hfile[self.split_type + "/data/data"][global_start:global_end]
        tres = self.hfile[self.split_type + "/t_res/data"][global_start:global_end]

        data_x, _, mask = self._collate(event_starts, raw_data)

        return self.preprocessor(data_x, tres, mask)


class BaikalDatasetTresSingle(BaikalDataset):
    def __getitem__(self, idx):
        start, end = self.hfile[self.split_type + "/ev_starts/data"][idx : idx + 2]
        data = np.array(self.hfile[self.split_type + "/data/data"][start:end])
        tres = self.tres_data[start:end]

        data_x = torch.tensor(data, dtype=torch.float32)
        data_y = torch.tensor(tres, dtype=torch.float32)

        data_x, data_y = self.preprocessor(data_x, data_y)

        return data_x, data_y


class BaikalDatasetTrackCascade(BaikalDataset):
    def __getitem__(self, idx):
        batch_start_idx = idx * self.batch_size
        batch_end_idx = min(
            (idx + 1) * self.batch_size,
            len(self.hfile[self.split_type + "/ev_starts/data"]) - 1,
        )
        event_starts = self.hfile[self.split_type + "/ev_starts/data"][
            batch_start_idx : batch_end_idx + 1
        ]

        global_start = event_starts[0]
        global_end = event_starts[-1]
        raw_data = torch.tensor(
            self.hfile[self.split_type + "/data/data"][global_start:global_end]
        )
        tres = torch.tensor(
            self.hfile[self.split_type + "/t_res/data"][global_start:global_end]
        )

        labels = self.hfile[self.split_type + "/labels/data"][global_start:global_end]

        raw_data = torch.cat([raw_data, tres.reshape(-1, 1)], dim=-1)
        data_x, data_y, mask = self._collate(event_starts, raw_data, labels, pad_y=True)
        tres = data_x[:, :, -1].reshape(data_x.shape[0], -1)
        data_x = data_x[:, :, :-1]

        # additional_data = {
        #     "norm_params": self.hfile[self.split_type + "/norm_params/data"][global_start:global_end],
        # }
        return self.preprocessor(data_x, data_y, tres, mask)


class BaikalDatasetAngles(BaikalDataset):
    def __getitem__(self, idx):
        batch_start_idx = idx * self.batch_size
        batch_end_idx = min(
            (idx + 1) * self.batch_size,
            len(self.hfile[self.split_type + "/ev_starts/data"]) - 1,
        )
        event_starts = self.hfile[self.split_type + "/ev_starts/data"][
            batch_start_idx : batch_end_idx + 1
        ]
        prime_prty = torch.tensor(
            self.hfile[self.split_type + "/prime_prty/data"][
                batch_start_idx:batch_end_idx
            ]
        )
        global_start = event_starts[0]
        global_end = event_starts[-1]
        raw_data = torch.tensor(
            self.hfile[self.split_type + "/data/data"][global_start:global_end]
        )
        hits_labels = torch.tensor(
            self.hfile[self.split_type + "/labels/data"][global_start:global_end]
        )
        # is_track_hit = hits_labels < 0
        tres = torch.tensor(
            self.hfile[self.split_type + "/t_res/data"][global_start:global_end]
        )
        raw_data = torch.cat(
            [raw_data, hits_labels.reshape(-1, 1), tres.reshape(-1, 1)], dim=-1
        )
        data_x, _, mask = self._collate(event_starts, raw_data, None, pad_y=False)
        # data_y: [batch_size, 5]
        # mask = mask & (data_x[:, :, -1] < 0)
        data_x, labels, tres = data_x[:, :, :-2], data_x[:, :, -2], data_x[:, :, -1]
        res = self.preprocessor(data_x, prime_prty, mask, labels, tres)
        return res


class BaikalDatasetAnglesOld(BaikalDataset):
    def __getitem__(self, idx):
        batch_start_idx = idx * self.batch_size
        batch_end_idx = min(
            (idx + 1) * self.batch_size, len(self.hfile[self.split_type + "/data"]) - 1
        )

        angles = torch.tensor(
            self.hfile[self.split_type + "/ev_chars"][batch_start_idx:batch_end_idx]
        )

        batch_x = []
        seq_lengths = []
        for i in range(batch_start_idx, batch_end_idx):
            mask_len = int(np.sum(self.hfile[f"{self.split_type}/mask"][i], axis=-1))
            seq_lengths.append(mask_len)
            data = self.hfile[self.split_type + "/data"][i][:mask_len]
            batch_x.append(torch.tensor(data, dtype=torch.float32))
        data_x = torch.nn.utils.rnn.pad_sequence(
            batch_x, batch_first=True, padding_value=0
        )

        mask = torch.zeros((len(seq_lengths), data_x.shape[1]), dtype=torch.bool)
        for i, length in enumerate(seq_lengths):
            mask[i, :length] = 1
        return self.preprocessor(data_x, angles, mask)


class BaikalDatasetAnglesSingle(BaikalDataset):
    def __getitem__(self, idx):
        start, end = self.hfile[self.split_type + "/ev_starts/data"][idx : idx + 2]
        data = np.array(self.hfile[self.split_type + "/data/data"][start:end])
        thetha, phi = self.hfile[self.split_type + "/prime_prty/data"][idx][:2]

        thetha = float(thetha) * math.pi / 180
        phi = float(phi) * math.pi / 180
        vec = [
            math.sin(thetha) * math.cos(phi),
            math.sin(thetha) * math.sin(phi),
            math.cos(thetha),
        ]

        data_x = torch.tensor(data, dtype=torch.float32)
        data_y = torch.tensor(vec, dtype=torch.float32)
        data_y /= data_y.norm()
        return self.preprocessor(data_x, data_y)


class BaikalDatasetAnglesOldSingle(BaikalDataset):
    def __getitem__(self, idx):
        angles = torch.tensor(self.hfile[self.split_type + "/ev_chars"][idx])
        mask_len = int(np.sum(self.hfile[f"{self.split_type}/mask"][idx], axis=-1))
        data_x = torch.tensor(self.hfile[self.split_type + "/data"][idx][:mask_len])

        return self.preprocessor(data_x, angles)


class BaikalDatasetEnergy(BaikalDataset):
    def __getitem__(self, idx):
        batch_start_idx = idx * self.batch_size
        batch_end_idx = min(
            (idx + 1) * self.batch_size,
            len(self.hfile[self.split_type + "/ev_starts/data"]) - 1,
        )
        event_starts = self.hfile[self.split_type + "/ev_starts/data"][
            batch_start_idx : batch_end_idx + 1
        ]
        prime_prty = torch.tensor(
            self.hfile[self.split_type + "/prime_prty/data"][
                batch_start_idx:batch_end_idx
            ]
        )
        global_start = event_starts[0]
        global_end = event_starts[-1]
        raw_data = torch.tensor(
            self.hfile[self.split_type + "/data/data"][global_start:global_end]
        )
        hits_labels = torch.tensor(
            self.hfile[self.split_type + "/labels/data"][global_start:global_end]
        )

        energy = prime_prty[:, 2]
        # is_track_hit = hits_labels < 0
        tres = torch.tensor(
            self.hfile[self.split_type + "/t_res/data"][global_start:global_end]
        )
        raw_data = torch.cat(
            [raw_data, hits_labels.reshape(-1, 1), tres.reshape(-1, 1)], dim=-1
        )
        data_x, _, mask = self._collate(event_starts, raw_data, None, pad_y=False)
        # data_y: [batch_size, 5]
        # mask = mask & (data_x[:, :, -1] < 0)
        data_x, labels, tres = data_x[:, :, :-2], data_x[:, :, -2], data_x[:, :, -1]
        res = self.preprocessor(data_x, energy, mask, labels=labels, tres=tres)

        return res
