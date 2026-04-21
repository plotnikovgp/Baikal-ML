import typing as tp
from abc import ABC, abstractmethod

import h5py as h5
import numpy as np
import torch
import torch_geometric.nn as gnn
from torch_geometric.data import Data as GData

EPS = 1e-8


class DataPrefilter:
    def __init__(
        self,
        data_file: str | None = None,
        Q_lower_bound: float | None = None,
        Q_upper_bound: float | None = None,
        additive_gauss_noise_std: tp.Sequence[float] | None = None,
        mult_gauss_noise_fraction: float | None = None,
        norm_Q: bool = False,
        **kwargs,
    ):
        self.additive_gauss_noise_std = additive_gauss_noise_std
        self.mult_gauss_noise_fraction = mult_gauss_noise_fraction
        self.norm_Q = norm_Q
        self.Q_lower_bound = None
        self.Q_upper_bound = None

        if data_file:
            self.hfile = h5.File(data_file, "r")
            self.means = np.array(self.hfile["norm_param/mean"])
            self.stds = np.array(self.hfile["norm_param/std"])
        if Q_lower_bound is not None or Q_upper_bound is not None:
            assert data_file
            if Q_lower_bound is not None:
                self.Q_lower_bound = (Q_lower_bound - self.means[0]) / self.stds[0]
            if Q_upper_bound is not None:
                self.Q_upper_bound = (Q_upper_bound - self.means[0]) / self.stds[0]

    def __call__(self, data_x):
        if self.norm_Q:
            data_x[0] = (data_x[0] - self.means[0]) / self.stds[0]
        if self.Q_lower_bound is not None:
            data_x[0] = data_x[0].clamp(min=self.Q_lower_bound)
        if self.Q_upper_bound is not None:
            data_x[0] = data_x[0].clamp(max=self.Q_upper_bound)

        if self.additive_gauss_noise_std is not None:
            # data_x: [batch_size, 5], additive_gauss_noise_std: [5], add noise to each feature with corresponding std
            noise = torch.randn_like(data_x) * torch.tensor(
                self.additive_gauss_noise_std, device=data_x.device
            )
            data_x = data_x + noise
            data_x[:, :1] = 0
        if self.mult_gauss_noise_fraction is not None:
            data_x[0] = data_x[0] * (1 + (0, self.mult_gauss_noise_fraction, data_x[0].shape))
        return data_x

    def denormalize(self, data_x):
        return data_x * self.stds + self.means


class BasePreprocessor:
    def __init__(self, data_prefilter: DataPrefilter | None = None, *args, **kwargs):
        self.data_prefilter = data_prefilter
        self.training = True

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def __call__(self, x, y, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        return x, y


class NoiseSigPreprocessor(BasePreprocessor):
    def __init__(
        self,
        data_prefilter: DataPrefilter | None = None,
        tres_cut_for_track_hit: float = 20.0,
        z_mirror: bool = False,
        **kwargs,
    ):
        super().__init__(data_prefilter, **kwargs)
        self.tres_cut_for_track_hit = tres_cut_for_track_hit
        self.z_mirror = z_mirror

    def __call__(
        self, x: torch.Tensor, y: torch.Tensor, t_res: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.data_prefilter is not None:
            x = self.data_prefilter(x)
        if self.z_mirror and self.training and torch.rand(1).item() < 0.5:
            x = x.clone()
            x[:, :, 4] = -x[:, :, 4]
        signal_mask = torch.abs(t_res) < self.tres_cut_for_track_hit
        y[signal_mask] = 1
        y[~signal_mask] = 0
        y = y.long()
        return x, y, mask


class NoiseSigOriginalLabelsPreprocessor(BasePreprocessor):
    """Use original MC labels: positive label -> signal (1), negative -> noise (0)."""

    def __call__(
        self, x: torch.Tensor, y: torch.Tensor, t_res: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.data_prefilter is not None:
            x = self.data_prefilter(x)
        y = (y > 0).long()
        return x, y, mask


class NoiseSigOrLabelsPreprocessor(BasePreprocessor):
    """Signal = |t_res| < tres_cut OR original label > 0."""

    def __init__(
        self,
        data_prefilter: DataPrefilter | None = None,
        tres_cut_for_track_hit: float = 10.0,
        z_mirror: bool = False,
        **kwargs,
    ):
        super().__init__(data_prefilter, **kwargs)
        self.tres_cut_for_track_hit = tres_cut_for_track_hit
        self.z_mirror = z_mirror

    def __call__(
        self, x: torch.Tensor, y: torch.Tensor, t_res: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.data_prefilter is not None:
            x = self.data_prefilter(x)
        if self.z_mirror and self.training and torch.rand(1).item() < 0.5:
            x = x.clone()
            x[:, :, 4] = -x[:, :, 4]
        signal_mask = (torch.abs(t_res) < self.tres_cut_for_track_hit) | (y > 0)
        y[signal_mask] = 1
        y[~signal_mask] = 0
        y = y.long()
        return x, y, mask


class TresRegressionPreprocessor(BasePreprocessor):
    """Predict |t_res|. Noise hits capped at max_tres."""

    def __init__(
        self,
        data_prefilter: DataPrefilter | None = None,
        max_tres: float = 100.0,
        z_mirror: bool = False,
        **kwargs,
    ):
        super().__init__(data_prefilter, **kwargs)
        self.max_tres = max_tres
        self.z_mirror = z_mirror

    def __call__(
        self, x: torch.Tensor, y: torch.Tensor, t_res: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.data_prefilter is not None:
            x = self.data_prefilter(x)
        if self.z_mirror and self.training and torch.rand(1).item() < 0.5:
            x = x.clone()
            x[:, :, 4] = -x[:, :, 4]
        t_res = t_res.abs().clone()
        noise_mask = y < 0
        t_res[noise_mask] = t_res[noise_mask].clamp(max=self.max_tres)
        return x, t_res, mask


# class NoiseSigPreprocessor(BasePreprocessor):
#     def __init__(self, data_prefilter: DataPrefilter | None = None, tres_cut_for_track_hit: float = 20.0):
#         self.tres_cut_for_track_hit = tres_cut_for_track_hit


#     def __call__(
#         self, x: torch.Tensor, y: torch.Tensor, t_res: torch.Tensor, mask: torch.Tensor
#     ) -> tuple[torch.Tensor, torch.Tensor]:
#         y[(y != 0)] = 1
#         y = y.long()
#         return x, y, mask


class NoLabelsPreprocessor(BasePreprocessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.data_prefilter is not None:
            x = self.data_prefilter(x)
        return x, torch.zeros(x.shape[0], dtype=torch.float32), mask


class NoLabelsPerHitPreprocessor(BasePreprocessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.data_prefilter is not None:
            x = self.data_prefilter(x)
        batch_size, seq_len = x.shape[:2]
        y = torch.zeros(batch_size, seq_len, dtype=torch.float32)
        return x, y, mask


class TrackCascadePreprocessor(BasePreprocessor):
    def __init__(self, tres_cut):
        self.tres_cut = tres_cut

    def __call__(
        self, x: torch.Tensor, y: torch.Tensor, tres: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # print(x.shape, y.shape, mask.shape, tres.shape)
        y[y > 0] = 1  # cascade
        y[y < 0] = 0  # track
        y[torch.abs(tres) < self.tres_cut] = 0
        return x, y.unsqueeze(0).type(torch.LongTensor), mask


class TresPreprocessor(BasePreprocessor):
    def __init__(self):
        self.tres_mean = None
        self.tres_std = None

    def set_stats(self, tres_mean: float, tres_std: float):
        self.tres_mean = tres_mean
        self.tres_std = tres_std

    def __call__(
        self,
        x: torch.Tensor,
        tres: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        tres = (tres - self.tres_mean) / (self.tres_std + EPS)
        return x, tres, mask


class TresAndTrackCascadePreprocessor(TresPreprocessor):
    def __init__(self, tres_cut):
        super().__init__()
        self.tres_cut = tres_cut

    def __call__(
        self, x: torch.Tensor, y: torch.Tensor, tres: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        y[y > 0] = 1
        y[y < 0] = 0
        y[torch.abs(tres) < self.tres_cut] = 0
        tres = (tres - self.tres_mean) / (self.tres_std + EPS)
        labels_and_tres = torch.stack((y, tres), -1)
        return x, labels_and_tres


class AnglePreprocessor(BasePreprocessor):
    def __call__(
        self, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        y = y[:, :2]
        thetha = torch.deg2rad(y[:, 0])
        phi = torch.deg2rad(y[:, 1])
        angle = torch.zeros(y.shape[0], 3, dtype=torch.float32)
        angle[:, 0] = torch.sin(thetha) * torch.cos(phi)
        angle[:, 1] = torch.sin(thetha) * torch.sin(phi)
        angle[:, 2] = torch.cos(thetha)
        return x, angle, mask


class AnglePreprocessorWithTres(BasePreprocessor):
    def __init__(self, tres_cut: float, data_prefilter: DataPrefilter | None = None):
        self.tres_cut = tres_cut
        self.data_prefilter = data_prefilter

    def __call__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        mask: torch.Tensor,
        tres: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        y = y[:, :2]
        thetha = torch.deg2rad(y[:, 0])
        phi = torch.deg2rad(y[:, 1])
        # print(y.shape, thetha.min(), thetha.max(), phi.min(), phi.max())
        angle = torch.zeros(y.shape[0], 3, dtype=torch.float32)
        angle[:, 0] = torch.sin(thetha) * torch.cos(phi)
        angle[:, 1] = torch.sin(thetha) * torch.sin(phi)
        angle[:, 2] = torch.cos(thetha)

        if self.data_prefilter is not None:
            x = self.data_prefilter(x)
        mask = mask  # & (labels != 0) & track_hits
        mask[mask.sum(-1) == 0] = True
        return x, angle, mask


class EnergyPreprocessor(BasePreprocessor):
    def __init__(self, data_prefilter: DataPrefilter | None = None):
        self.data_prefilter = data_prefilter

    def __call__(
        self, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # print(y.shape, thetha.min(), thetha.max(), phi.min(), phi.max())
        energy = torch.log10(y)

        if self.data_prefilter is not None:
            x = self.data_prefilter(x)
        mask = mask
        mask[mask.sum(-1) == 0] = True
        return x, energy, mask


class DirectionPreprocessor(BasePreprocessor):
    def __call__(
        self, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        points = y[:, 2:5]
        thetha = torch.deg2rad(y[:, 0])
        phi = torch.deg2rad(y[:, 1])
        # print(y.shape, thetha.min(), thetha.max(), phi.min(), phi.max())
        angle = torch.zeros(y.shape[0], 3, dtype=torch.float32)
        angle[:, 0] = torch.sin(thetha) * torch.cos(phi)
        angle[:, 1] = torch.sin(thetha) * torch.sin(phi)
        angle[:, 2] = torch.cos(thetha)
        angle_and_point = torch.cat((angle, points), -1)
        return x, angle_and_point, mask


class AngleAndTrackCascadePreprocessor(TrackCascadePreprocessor):
    def __call__(
        self,
        x: torch.Tensor,
        track_cascade_labels: torch.Tensor,
        angles: torch.Tensor,
        tres: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        track_cascade_labels[track_cascade_labels > 0] = 1
        track_cascade_labels[track_cascade_labels < 0] = 0
        track_cascade_labels[torch.abs(tres) < self.tres_cut] = 0
        return x, (track_cascade_labels.unsqueeze(0), angles)


class BaseGraphPreprocessor(ABC):
    def __init__(self, n_neighbours: int):
        self.n_neighbours = n_neighbours

    @abstractmethod
    def __call__(self, *args, **kwargs) -> GData:
        pass


class AngleGraphPreprocessor(BaseGraphPreprocessor):
    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> GData:
        # y = y[:2]
        # thetha = torch.deg2rad(y[0])
        # phi = torch.deg2rad(y[1])
        # angle = torch.zeros(3, dtype=torch.float32)
        # angle[0] = torch.sin(thetha) * torch.cos(phi)
        # angle[1] = torch.sin(thetha) * torch.sin(phi)
        # angle[2] = torch.cos(thetha)
        # y = angle
        # x[:, 0] = 0
        edge_index = gnn.knn_graph(x[:, 1], k=self.n_neighbours)
        graph = GData(x=x, edge_index=edge_index, y=y)
        return graph


class NoiseSigGraphPreprocessor(BaseGraphPreprocessor):
    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> GData:
        y[y != 0] = 1
        edge_index = gnn.knn_graph(x[:, 1], k=self.n_neighbours)
        graph = GData(x=x, edge_index=edge_index, y=y)
        return graph


class TrackCascadeGraphPreprocessor(BaseGraphPreprocessor):
    def __init__(self, n_neighbours: int, tres_cut: float):
        super().__init__(n_neighbours)
        self.tres_cut = tres_cut

    def __call__(self, x: torch.Tensor, y: torch.Tensor, tres: torch.Tensor) -> GData:
        y[y > 0] = 1  # cascade
        y[y < 0] = 0  # track
        y[torch.abs(tres) < self.tres_cut] = 0

        edge_index = gnn.knn_graph(x[:, 1], k=self.n_neighbours)
        graph = GData(x=x, edge_index=edge_index, y=y)
        return graph


class TresGraphPreprocessor(BaseGraphPreprocessor):
    def __init__(self, n_neighbours: int):
        super().__init__(n_neighbours)
        self.tres_mean = None
        self.tres_std = None

    def set_stats(self, tres_mean: float, tres_std: float):
        self.tres_mean = tres_mean
        self.tres_std = tres_std

    def __call__(self, x: torch.Tensor, tres: torch.Tensor) -> GData:
        assert self.tres_mean is not None and self.tres_std is not None, (
            "stats for preproccesor weren't not set"
        )
        tres = (tres - self.tres_mean) / (self.tres_std + EPS)
        edge_index = gnn.knn_graph(x[:, 1], k=self.n_neighbours)
        graph = GData(x=x, edge_index=edge_index, y=tres)
        return graph


class TresAndTrackCascadeGraphPreprocessor(TresGraphPreprocessor):
    def __init__(self, n_neighbours: int, tres_cut: float):
        super().__init__(n_neighbours)
        self.tres_cut = tres_cut

    def __call__(self, x: torch.Tensor, y: torch.Tensor, tres: torch.Tensor) -> GData:
        y[y > 0] = 1
        y[y < 0] = 0
        y[torch.abs(tres) < self.tres_cut] = 0
        tres = (tres - self.tres_mean) / (self.tres_std + EPS)
        labels_and_tres = torch.stack((y, tres), dim=-1)
        edge_index = gnn.knn_graph(x[:, 1], k=self.n_neighbours)
        graph = GData(x=x, edge_index=edge_index, y=labels_and_tres)
        return graph
