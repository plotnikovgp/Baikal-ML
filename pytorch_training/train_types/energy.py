import math
from pathlib import Path
from typing import Any, Dict

import torch

from data_utils import BaikalDatasetEnergy
from data_utils.preprocessors import DataPrefilter, EnergyPreprocessor
from metrics import EnergyMetrics

from .base import BaseTrainType, DomainAdaptationMixin


class EnergyReconstructionTrainType(BaseTrainType):
    @property
    def name(self) -> str:
        return "energy_reconstruction"

    def __init__(self, train_params: Dict[str, Any], device: str = "cuda", save_dir: Path = None):
        super().__init__(train_params, device)
        self.save_dir = save_dir
        self.is_val_mode = train_params.get("val_mode", False)
        self._criterion = torch.nn.MSELoss()

    def get_dataset_type(self):
        return BaikalDatasetEnergy

    def get_preprocessor(self, config: Dict[str, Any]):
        data_prefilter = DataPrefilter(**(config.get("data_prefilter_params", {})))
        return EnergyPreprocessor(data_prefilter)

    def get_criterion(self):
        return self._criterion

    def get_metrics_function(self):
        return EnergyMetrics(save_preds=self.is_val_mode, save_dir=self.save_dir)

    def _process_batch(self, model, data, dataset_idx=None) -> Dict[str, torch.Tensor]:
        if len(data) == 4:
            data = data[:3]

        x, y_true, mask = data
        x = x.to(self.device)
        y_true = y_true.to(self.device) if y_true is not None else None
        mask = mask.to(self.device)

        output = model(x, mask)

        return {"output": output.squeeze(), "y_pred": output.squeeze(), "y_true": y_true.squeeze()}


class EnergyReconstructionDomainAdaptationTrainType(DomainAdaptationMixin, BaseTrainType):
    @property
    def name(self) -> str:
        return "energy_reconstruction_domain_adaptation"

    def __init__(self, train_params: Dict[str, Any], device: str = "cuda", save_dir: Path = None):
        super().__init__(train_params, device)
        self.save_dir = save_dir
        self.is_val_mode = train_params.get("val_mode", False)
        self.domain_adaptation_loss_k = train_params.get("domain_adaptation_loss_k", 0.1)
        self.label_dataset_name = train_params.get("label_dataset_name", None)
        self._setup_loss()

    def _setup_loss(self):
        if self.train_params.get("use_cosh_loss", False):

            def log_cosh_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
                def _log_cosh(x: torch.Tensor) -> torch.Tensor:
                    return x + torch.nn.functional.softplus(-2.0 * x) - math.log(2.0)

                return torch.mean(_log_cosh(y_pred - y_true))

            self.energy_loss_ = log_cosh_loss
        else:
            self.energy_loss_ = torch.nn.MSELoss()

    def get_dataset_type(self):
        return BaikalDatasetEnergy

    def get_preprocessor(self, config: Dict[str, Any]):
        data_prefilter = DataPrefilter(**(config.get("data_prefilter_params", {})))
        return EnergyPreprocessor(data_prefilter)

    def get_criterion(self):
        dataset_names = self.dataset_names
        label_dataset_name = self.label_dataset_name
        energy_loss_ = self.energy_loss_
        domain_adaptation_loss_k = self.domain_adaptation_loss_k

        def criterion(y_pred, y_true, domain_pred=None, domain_true=None):
            assert label_dataset_name is not None
            y_pred = y_pred.squeeze()
            y_true = y_true.squeeze()

            if domain_true is None:
                return {"loss": energy_loss_(y_pred, y_true)}

            domain_true_event = domain_true[0] if isinstance(domain_true, tuple) else domain_true
            mask = domain_true_event == dataset_names.index(label_dataset_name)
            if mask.sum() > 0:
                energy_loss = energy_loss_(y_pred[mask], y_true[mask])
            else:
                energy_loss = torch.tensor(0.0, device=y_pred.device)

            domain_loss = torch.nn.functional.cross_entropy(domain_pred, domain_true_event)
            total_loss = energy_loss + domain_adaptation_loss_k * domain_loss
            return {
                "loss": total_loss,
                "energy_loss": energy_loss.detach(),
                "domain_loss": domain_loss.detach(),
            }

        return criterion

    def get_metrics_function(self):
        return EnergyMetrics(save_preds=self.is_val_mode, save_dir=self.save_dir)

    def get_train_kwargs(self) -> Dict[str, Any]:
        kwargs = super().get_train_kwargs()
        kwargs["is_domain_adaptation"] = True
        return kwargs

    def process_batch(self, model, data, dataset_idx=None) -> Dict[str, torch.Tensor]:
        if isinstance(data, tuple) and len(data) > 3:
            dataset_idx = data[-1]
            data = data[:-1]

        if len(data) == 4:
            data = data[:3]

        x, y_true, mask = data
        x = x.to(self.device)
        y_true = y_true.to(self.device) if y_true is not None else None
        mask = mask.to(self.device)

        output, domain_pred = model(x, mask)

        batch_size = x.shape[0]
        domain_true = torch.full((batch_size,), dataset_idx, dtype=torch.long, device=self.device)

        return {
            "output": output.squeeze(),
            "y_pred": output.squeeze(),
            "y_true": y_true.squeeze(),
            "domain_pred": domain_pred,
            "domain_true": (domain_true, domain_true),
        }
