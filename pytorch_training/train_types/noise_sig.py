from typing import Any, Dict

import torch

from data_utils import BaikalDataset, BaikalDatasetSingle
from data_utils.preprocessors import (
    DataPrefilter,
    NoiseSigGraphPreprocessor,
    NoiseSigOriginalLabelsPreprocessor,
    NoiseSigOrLabelsPreprocessor,
    NoiseSigPreprocessor,
    TresRegressionPreprocessor,
)
from metrics import BinaryClassificationMetrics, RegressionMetrics

from .base import BaseTrainType, DomainAdaptationMixin


class NoiseSigTrainType(BaseTrainType):
    @property
    def name(self) -> str:
        return "noise_sig"

    def __init__(self, train_params: Dict[str, Any], device: str = "cuda"):
        super().__init__(train_params, device)
        self.is_classification = True
        self._criterion = torch.nn.CrossEntropyLoss()

    def get_dataset_type(self):
        if self.is_graph:
            return BaikalDatasetSingle
        return BaikalDataset

    def get_preprocessor(self, config: Dict[str, Any]):
        prefilter = DataPrefilter(**(config.get("data_prefilter_params", {})))
        tres_cut = config.get("tres_cut_for_track_hit", 20.0)
        z_mirror = config.get("z_mirror", False)
        if self.is_graph:
            return NoiseSigGraphPreprocessor(prefilter, config["knn_neighbours"])
        return NoiseSigPreprocessor(prefilter, tres_cut_for_track_hit=tres_cut, z_mirror=z_mirror)

    def get_criterion(self):
        return self._criterion

    def get_metrics_function(self):
        return BinaryClassificationMetrics(min_recall=0.9)


class NoiseSigDomainAdaptationTrainType(DomainAdaptationMixin, BaseTrainType):
    @property
    def name(self) -> str:
        return "noise_sig_domain_adaptation"

    def __init__(self, train_params: Dict[str, Any], device: str = "cuda"):
        super().__init__(train_params, device)
        self.is_classification = True
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.domain_loss_fn = torch.nn.CrossEntropyLoss()
        tt_params = train_params.get("train_type", train_params)
        self.domain_adaptation_loss_k = tt_params.get("domain_adaptation_loss_k", 0.1)
        self.label_dataset_name = tt_params.get("label_dataset_name", None)

    def get_dataset_type(self):
        if self.is_graph:
            return BaikalDatasetSingle
        return BaikalDataset

    def get_preprocessor(self, config: Dict[str, Any]):
        prefilter = DataPrefilter(**(config.get("data_prefilter_params", {})))
        if self.is_graph:
            return NoiseSigGraphPreprocessor(prefilter, config["knn_neighbours"])
        return NoiseSigPreprocessor(prefilter)

    def get_criterion(self):
        dataset_names = self.dataset_names
        label_dataset_name = self.label_dataset_name
        loss_fn = self.loss_fn
        domain_loss_fn = self.domain_loss_fn
        domain_adaptation_loss_k = self.domain_adaptation_loss_k

        def criterion(y_pred, y_true, domain_pred=None, domain_true=None):
            res = {}

            if domain_true is None:
                noise_sig_loss = loss_fn(y_pred, y_true)
                res["loss"] = noise_sig_loss
                res["noise_sig_loss"] = noise_sig_loss
                return res

            domain_true_event = domain_true[0] if isinstance(domain_true, tuple) else domain_true
            is_labeled_batch = domain_true_event[0].item() == dataset_names.index(
                label_dataset_name
            )

            if is_labeled_batch:
                noise_sig_loss = loss_fn(y_pred, y_true)
            else:
                noise_sig_loss = torch.tensor(0.0, device=y_pred.device)

            domain_loss = domain_loss_fn(domain_pred, domain_true_event)
            total_loss = noise_sig_loss + domain_adaptation_loss_k * domain_loss

            res["loss"] = total_loss
            res["noise_sig_loss"] = noise_sig_loss.detach()
            res["domain_loss"] = domain_loss.detach()
            return res

        return criterion

    def get_metrics_function(self):
        return BinaryClassificationMetrics(min_recall=0.9)

    def get_train_kwargs(self) -> Dict[str, Any]:
        kwargs = super().get_train_kwargs()
        kwargs["is_domain_adaptation"] = True
        return kwargs

    def process_batch(self, model, data, dataset_idx=None) -> Dict[str, torch.Tensor]:
        return self._process_batch_with_domain(model, data, dataset_idx)


class NoiseSigOriginalLabelsTrainType(BaseTrainType):
    """Noise/signal classification using original MC labels (no tres cut)."""

    @property
    def name(self) -> str:
        return "noise_sig_original_labels"

    def __init__(self, train_params: Dict[str, Any], device: str = "cuda"):
        super().__init__(train_params, device)
        self.is_classification = True
        self._criterion = torch.nn.CrossEntropyLoss()

    def get_dataset_type(self):
        return BaikalDataset

    def get_preprocessor(self, config: Dict[str, Any]):
        prefilter = DataPrefilter(**(config.get("data_prefilter_params", {})))
        return NoiseSigOriginalLabelsPreprocessor(prefilter)

    def get_criterion(self):
        return self._criterion

    def get_metrics_function(self):
        return BinaryClassificationMetrics(min_recall=0.9)


class NoiseSigOrLabelsTrainType(BaseTrainType):
    """Noise/signal: signal = |tres| < tres_cut OR original label > 0."""

    @property
    def name(self) -> str:
        return "noise_sig_or_labels"

    def __init__(self, train_params: Dict[str, Any], device: str = "cuda"):
        super().__init__(train_params, device)
        self.is_classification = True
        self._criterion = torch.nn.CrossEntropyLoss()

    def get_dataset_type(self):
        return BaikalDataset

    def get_preprocessor(self, config: Dict[str, Any]):
        prefilter = DataPrefilter(**(config.get("data_prefilter_params", {})))
        tres_cut = config.get("tres_cut_for_track_hit", 10.0)
        z_mirror = config.get("z_mirror", False)
        return NoiseSigOrLabelsPreprocessor(
            prefilter, tres_cut_for_track_hit=tres_cut, z_mirror=z_mirror
        )

    def get_criterion(self):
        return self._criterion

    def get_metrics_function(self):
        return BinaryClassificationMetrics(min_recall=0.9)


class NoiseSigOriginalLabelsDomainAdaptationTrainType(DomainAdaptationMixin, BaseTrainType):
    """Original-labels noise/signal + domain adaptation on experimental data."""

    @property
    def name(self) -> str:
        return "noise_sig_original_labels_da"

    def __init__(self, train_params: Dict[str, Any], device: str = "cuda"):
        super().__init__(train_params, device)
        self.is_classification = True
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.domain_loss_fn = torch.nn.CrossEntropyLoss()
        tt_params = train_params.get("train_type", train_params)
        self.domain_adaptation_loss_k = tt_params.get("domain_adaptation_loss_k", 0.1)
        self.label_dataset_name = tt_params.get("label_dataset_name", None)

    def get_dataset_type(self):
        return BaikalDataset

    def get_preprocessor(self, config: Dict[str, Any]):
        prefilter = DataPrefilter(**(config.get("data_prefilter_params", {})))
        return NoiseSigOriginalLabelsPreprocessor(prefilter)

    def get_criterion(self):
        dataset_names = self.dataset_names
        label_dataset_name = self.label_dataset_name
        loss_fn = self.loss_fn
        domain_loss_fn = self.domain_loss_fn
        domain_adaptation_loss_k = self.domain_adaptation_loss_k

        def criterion(y_pred, y_true, domain_pred=None, domain_true=None):
            res = {}

            if domain_true is None:
                noise_sig_loss = loss_fn(y_pred, y_true)
                res["loss"] = noise_sig_loss
                res["noise_sig_loss"] = noise_sig_loss
                return res

            domain_true_event = domain_true[0] if isinstance(domain_true, tuple) else domain_true
            is_labeled_batch = domain_true_event[0].item() == dataset_names.index(
                label_dataset_name
            )

            if is_labeled_batch:
                noise_sig_loss = loss_fn(y_pred, y_true)
            else:
                noise_sig_loss = torch.tensor(0.0, device=y_pred.device)

            domain_loss = domain_loss_fn(domain_pred, domain_true_event)
            total_loss = noise_sig_loss + domain_adaptation_loss_k * domain_loss

            res["loss"] = total_loss
            res["noise_sig_loss"] = noise_sig_loss.detach()
            res["domain_loss"] = domain_loss.detach()
            return res

        return criterion

    def get_metrics_function(self):
        return BinaryClassificationMetrics(min_recall=0.9)

    def get_train_kwargs(self) -> Dict[str, Any]:
        kwargs = super().get_train_kwargs()
        kwargs["is_domain_adaptation"] = True
        return kwargs

    def process_batch(self, model, data, dataset_idx=None) -> Dict[str, torch.Tensor]:
        return self._process_batch_with_domain(model, data, dataset_idx)


class SignalFocusedMSELoss(torch.nn.Module):
    """MSE for signal hits (|t_res| < threshold), log-damped for noise hits.

    For |t_true| < threshold: loss = (pred - true)^2
    For |t_true| >= threshold: loss = threshold^2 * log(1 + (pred - true)^2 / threshold^2)

    The log branch is differentiable everywhere, matches MSE at the boundary,
    and grows slowly for large residuals so noisy hits don't dominate.
    """

    def __init__(self, threshold: float = 10.0):
        super().__init__()
        self.threshold = threshold

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        residual_sq = (y_pred - y_true) ** 2
        signal = y_true < self.threshold
        t2 = self.threshold**2
        loss = torch.where(
            signal,
            residual_sq,
            t2 * torch.log1p(residual_sq / t2),
        )
        return loss.mean()


class BoundaryWeightedMSELoss(torch.nn.Module):
    """MSE loss that upweights hits near the signal/noise threshold.

    weight(t) = 1 + boost * exp(-(|t| - threshold)^2 / (2*sigma^2))

    Hits with |t_res_true| near `threshold` get up to (1+boost)x weight.
    Hits far from the boundary get weight ~1.
    """

    def __init__(self, threshold: float = 10.0, sigma: float = 5.0, boost: float = 4.0):
        super().__init__()
        self.threshold = threshold
        self.sigma = sigma
        self.boost = boost

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        dist_to_boundary = (y_true.abs() - self.threshold).abs()
        w = 1.0 + self.boost * torch.exp(-0.5 * (dist_to_boundary / self.sigma) ** 2)
        return (w * (y_pred - y_true) ** 2).mean()


class TresRegressionTrainType(BaseTrainType):
    """Predict t_res directly (regression). Noise hits capped at max_tres.
    Uses boundary-weighted MSE to focus on the signal/noise threshold region.
    Reports both MSE and derived binary noise/signal metrics."""

    @property
    def name(self) -> str:
        return "tres_regression"

    def __init__(self, train_params: Dict[str, Any], device: str = "cuda"):
        super().__init__(train_params, device)
        self.is_classification = False
        tt = train_params.get("train_type", train_params)
        self.tres_cut = tt.get("tres_cut_for_binary", 10.0)
        threshold = float(self.tres_cut)
        sigma = tt.get("boundary_sigma", 5.0)
        boost = tt.get("boundary_boost", 4.0)
        self._criterion = BoundaryWeightedMSELoss(
            threshold=threshold,
            sigma=sigma,
            boost=boost,
        )

    def get_dataset_type(self):
        return BaikalDataset

    def get_preprocessor(self, config: Dict[str, Any]):
        prefilter = DataPrefilter(**(config.get("data_prefilter_params", {})))
        max_tres = config.get("max_tres", 100.0)
        z_mirror = config.get("z_mirror", False)
        return TresRegressionPreprocessor(prefilter, max_tres=max_tres, z_mirror=z_mirror)

    def get_criterion(self):
        return self._criterion

    def get_metrics_function(self):
        return TresWithBinaryMetrics(tres_cut=self.tres_cut)

    def _reshape_outputs(self, y_true, output, mask):
        y_true = y_true.reshape(-1)
        output = output.reshape(-1, output.shape[-1]).squeeze()
        mask_flat = mask.reshape(-1)
        y_true = y_true[mask_flat != 0]
        output = output[mask_flat != 0]
        return y_true, output, mask


class TresRegressionSoftLossTrainType(BaseTrainType):
    """Like TresRegressionTrainType but uses SignalFocusedMSELoss:
    MSE for signal hits, log-damped for noise hits."""

    @property
    def name(self) -> str:
        return "tres_regression_soft"

    def __init__(self, train_params: Dict[str, Any], device: str = "cuda"):
        super().__init__(train_params, device)
        self.is_classification = False
        tt = train_params.get("train_type", train_params)
        self.tres_cut = tt.get("tres_cut_for_binary", 10.0)
        self._criterion = SignalFocusedMSELoss(threshold=float(self.tres_cut))

    def get_dataset_type(self):
        return BaikalDataset

    def get_preprocessor(self, config: Dict[str, Any]):
        prefilter = DataPrefilter(**(config.get("data_prefilter_params", {})))
        max_tres = config.get("max_tres", 100.0)
        z_mirror = config.get("z_mirror", False)
        return TresRegressionPreprocessor(prefilter, max_tres=max_tres, z_mirror=z_mirror)

    def get_criterion(self):
        return self._criterion

    def get_metrics_function(self):
        return TresWithBinaryMetrics(tres_cut=self.tres_cut)

    def _reshape_outputs(self, y_true, output, mask):
        y_true = y_true.reshape(-1)
        output = output.reshape(-1, output.shape[-1]).squeeze()
        mask_flat = mask.reshape(-1)
        y_true = y_true[mask_flat != 0]
        output = output[mask_flat != 0]
        return y_true, output, mask


class TresWithBinaryMetrics:
    """Compute regression metrics on |t_res| AND derived binary classification metrics.
    Targets are already absolute. Signal if predicted < tres_cut."""

    def __init__(self, tres_cut: float = 10.0):
        self.tres_cut = tres_cut
        self._reg = RegressionMetrics()
        self._bin = BinaryClassificationMetrics(min_recall=0.9)

    def __call__(self, y_pred, y_true, **kwargs):
        import numpy as np

        y_pred_np = np.array(y_pred, dtype=np.float32)
        y_true_np = np.array(y_true, dtype=np.float32)

        metrics = self._reg(y_pred, y_true, **kwargs)

        true_signal = (y_true_np < self.tres_cut).astype(np.int32)
        signal_prob = 1.0 - np.minimum(np.clip(y_pred_np, 0, None) / self.tres_cut, 1.0)

        if len(np.unique(true_signal)) > 1:
            bin_metrics = self._bin(signal_prob, true_signal)
            metrics.update({"bin_" + k: v for k, v in bin_metrics.items()})

        return metrics
