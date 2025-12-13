from typing import Any, Dict

import torch

from data_utils import BaikalDatasetTrackCascade, BaikalDatasetTres
from data_utils.preprocessors import (
    TresAndTrackCascadeGraphPreprocessor,
    TresAndTrackCascadePreprocessor,
    TresGraphPreprocessor,
    TresPreprocessor,
)
from metrics import RegressionMetrics, TresAndTrackCascadeMetrics

from .base import BaseTrainType, DomainAdaptationMixin


class TresTrainType(BaseTrainType):
    @property
    def name(self) -> str:
        return "tres"

    def __init__(self, train_params: Dict[str, Any], device: str = "cuda"):
        super().__init__(train_params, device)
        self._criterion = torch.nn.MSELoss()

    def get_dataset_type(self):
        return BaikalDatasetTres

    def get_preprocessor(self, config: Dict[str, Any]):
        if self.is_graph:
            return TresGraphPreprocessor(config["knn_neighbours"])
        return TresPreprocessor()

    def get_criterion(self):
        return self._criterion

    def get_metrics_function(self):
        return RegressionMetrics()


class TresDomainAdaptationTrainType(DomainAdaptationMixin, BaseTrainType):
    @property
    def name(self) -> str:
        return "tres_domain_adaptation"

    def __init__(self, train_params: Dict[str, Any], device: str = "cuda"):
        super().__init__(train_params, device)
        self.domain_adaptation_loss_k = train_params.get("domain_adaptation_loss_k", 0.1)
        self.label_dataset_name = train_params.get("label_dataset_name", None)
        self.tres_loss_ = torch.nn.MSELoss()

    def get_dataset_type(self):
        return BaikalDatasetTres

    def get_preprocessor(self, config: Dict[str, Any]):
        if self.is_graph:
            return TresGraphPreprocessor(config["knn_neighbours"])
        return TresPreprocessor()

    def get_criterion(self):
        dataset_names = self.dataset_names
        label_dataset_name = self.label_dataset_name
        tres_loss_ = self.tres_loss_
        domain_adaptation_loss_k = self.domain_adaptation_loss_k

        def criterion(y_pred, y_true, domain_pred=None, domain_true=None):
            if y_pred.shape != y_true.shape:
                assert y_pred.size(0) == y_true.size(0)
                y_pred = y_pred.reshape(-1)
                y_true = y_true.reshape(-1)
            assert label_dataset_name is not None
            y_pred = y_pred.squeeze()
            y_true = y_true.squeeze()

            if domain_true is None:
                return {"loss": tres_loss_(y_pred, y_true)}

            domain_true_event, domain_true_hit = domain_true

            label_idx = dataset_names.index(label_dataset_name)
            mask = domain_true_hit == label_idx
            if mask.sum() > 0:
                tres_loss = tres_loss_(y_pred[mask], y_true[mask])
            else:
                tres_loss = torch.tensor(0.0, device=y_pred.device)

            domain_loss = torch.nn.functional.cross_entropy(domain_pred, domain_true_event)
            total_loss = tres_loss + domain_adaptation_loss_k * domain_loss
            return {
                "loss": total_loss,
                "tres_loss": tres_loss.detach(),
                "domain_loss": domain_loss.detach(),
            }

        return criterion

    def get_metrics_function(self):
        return RegressionMetrics()

    def get_train_kwargs(self) -> Dict[str, Any]:
        kwargs = super().get_train_kwargs()
        kwargs["is_domain_adaptation"] = True
        return kwargs

    def process_batch(self, model, data, dataset_idx=None) -> Dict[str, torch.Tensor]:
        return self._process_batch_with_domain(model, data, dataset_idx)


class TresAndTrackCascadeTrainType(BaseTrainType):
    @property
    def name(self) -> str:
        return "tres_and_track_cascade"

    def __init__(self, train_params: Dict[str, Any], device: str = "cuda"):
        super().__init__(train_params, device)
        self.tres_mse_coef = train_params.get("tres_mse_coef", 1.0)
        self.ce_ = torch.nn.CrossEntropyLoss()
        self.mse_ = torch.nn.MSELoss()

    def get_dataset_type(self):
        return BaikalDatasetTrackCascade

    def get_preprocessor(self, config: Dict[str, Any]):
        if self.is_graph:
            return TresAndTrackCascadeGraphPreprocessor(
                config["knn_neighbours"], config["tres_cut"]
            )
        return TresAndTrackCascadePreprocessor(config["tres_cut"])

    def get_criterion(self):
        ce_ = self.ce_
        mse_ = self.mse_
        tres_mse_coef = self.tres_mse_coef

        def criterion(y_pred, y_true):
            ce_loss = ce_(y_pred[:, :-1], y_true[:, 0].long())
            mse_loss = mse_(y_pred[:, -1], y_true[:, -1])
            return ce_loss + tres_mse_coef * mse_loss

        return criterion

    def get_metrics_function(self):
        return TresAndTrackCascadeMetrics()

    def _reshape_outputs(self, y_true, output, mask):
        y_true = y_true.reshape(-1, y_true.shape[-1])
        output = output.reshape(-1, output.shape[-1]).squeeze()
        return y_true, output, mask

    def _compute_predictions(self, output, y_true) -> torch.Tensor:
        y_pred = torch.zeros_like(y_true)
        y_pred[:, 0] = torch.sigmoid(output[:, 1])
        y_pred[:, 1] = output[:, -1]
        return y_pred
