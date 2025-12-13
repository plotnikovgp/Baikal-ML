from typing import Any, Dict

import torch

from data_utils import BaikalDatasetTrackCascade, BaikalDatasetTrackCascadeSingle
from data_utils.preprocessors import TrackCascadeGraphPreprocessor, TrackCascadePreprocessor
from metrics import TrackCascadeMetrics

from .base import BaseTrainType, DomainAdaptationMixin


class TrackCascadeTrainType(BaseTrainType):
    @property
    def name(self) -> str:
        return "track_cascade"

    def __init__(self, train_params: Dict[str, Any], device: str = "cuda"):
        super().__init__(train_params, device)
        self.is_classification = True
        self._criterion = torch.nn.CrossEntropyLoss()

    def get_dataset_type(self):
        if self.is_graph:
            return BaikalDatasetTrackCascadeSingle
        return BaikalDatasetTrackCascade

    def get_preprocessor(self, config: Dict[str, Any]):
        if self.is_graph:
            return TrackCascadeGraphPreprocessor(config["knn_neighbours"])
        return TrackCascadePreprocessor()

    def get_criterion(self):
        return self._criterion

    def get_metrics_function(self):
        return TrackCascadeMetrics()


class TrackCascadeDomainAdaptationTrainType(DomainAdaptationMixin, BaseTrainType):
    @property
    def name(self) -> str:
        return "track_cascade_domain_adaptation"

    def __init__(self, train_params: Dict[str, Any], device: str = "cuda"):
        super().__init__(train_params, device)
        self.is_classification = True
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.domain_loss_fn = torch.nn.CrossEntropyLoss()
        self.domain_adaptation_loss_k = train_params.get("domain_adaptation_loss_k", 0.1)
        self.label_dataset_name = train_params.get("label_dataset_name", None)

    def get_dataset_type(self):
        if self.is_graph:
            return BaikalDatasetTrackCascadeSingle
        return BaikalDatasetTrackCascade

    def get_preprocessor(self, config: Dict[str, Any]):
        if self.is_graph:
            return TrackCascadeGraphPreprocessor(config["knn_neighbours"])
        return TrackCascadePreprocessor()

    def get_criterion(self):
        dataset_names = self.dataset_names
        label_dataset_name = self.label_dataset_name
        loss_fn = self.loss_fn
        domain_loss_fn = self.domain_loss_fn
        domain_adaptation_loss_k = self.domain_adaptation_loss_k

        def criterion(y_pred, y_true, domain_pred=None, domain_true=None):
            res = {}

            if domain_true is None:
                loss = loss_fn(y_pred, y_true)
                res["loss"] = loss
                return res

            domain_true_event = domain_true[0] if isinstance(domain_true, tuple) else domain_true
            is_labeled_batch = domain_true_event[0].item() == dataset_names.index(
                label_dataset_name
            )

            if is_labeled_batch:
                track_cascade_loss = loss_fn(y_pred, y_true)
            else:
                track_cascade_loss = torch.tensor(0.0, device=y_pred.device)

            domain_loss = domain_loss_fn(domain_pred, domain_true_event)
            total_loss = track_cascade_loss + domain_adaptation_loss_k * domain_loss

            res["loss"] = total_loss
            res["track_cascade_loss"] = track_cascade_loss.detach()
            res["domain_loss"] = domain_loss.detach()
            return res

        return criterion

    def get_metrics_function(self):
        return TrackCascadeMetrics()

    def get_train_kwargs(self) -> Dict[str, Any]:
        kwargs = super().get_train_kwargs()
        kwargs["is_domain_adaptation"] = True
        return kwargs

    def process_batch(self, model, data, dataset_idx=None) -> Dict[str, torch.Tensor]:
        return self._process_batch_with_domain(model, data, dataset_idx)
