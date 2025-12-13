from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional

import torch
import torch_geometric


class BaseTrainType(ABC):
    def __init__(self, train_params: Dict[str, Any], device: str = "cuda"):
        self.train_params = train_params
        self.device = device
        self.is_graph = train_params.get("is_graph", False)
        self.is_classification = False
        self.dataset_names: Optional[list] = None

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def get_dataset_type(self):
        pass

    @abstractmethod
    def get_preprocessor(self, config: Dict[str, Any]):
        pass

    @abstractmethod
    def get_criterion(self) -> Callable:
        pass

    @abstractmethod
    def get_metrics_function(self):
        pass

    def get_train_kwargs(self) -> Dict[str, Any]:
        return {
            "is_classification": self.is_classification,
            "is_domain_adaptation": False,
        }

    def set_dataset_names(self, names: list):
        self.dataset_names = names

    def process_batch(self, model, data, dataset_idx=None) -> Dict[str, torch.Tensor]:
        if self.is_graph:
            return self._process_graph_batch(model, data)
        return self._process_batch(model, data, dataset_idx)

    def _process_graph_batch(self, model, data) -> Dict[str, torch.Tensor]:
        if isinstance(data, list):
            data = data[0]
        data = data.to(self.device)
        data.edge_index = torch_geometric.utils.sort_edge_index(data.edge_index)
        y_true = data.y

        output = model(data.x, data.edge_index, data.batch).squeeze()
        y_pred = self._compute_predictions(output, y_true)

        return {"output": output, "y_pred": y_pred, "y_true": y_true}

    def _process_batch(self, model, data, dataset_idx=None) -> Dict[str, torch.Tensor]:
        if isinstance(data, tuple) and len(data) == 2:
            if isinstance(data[0], tuple):
                data = data[0]
            elif not isinstance(data[1], torch.Tensor) or data[1].dim() == 0:
                data = data[0]

        if isinstance(data, tuple) and len(data) > 3:
            data = data[:3]

        x, y_true, mask = data
        x = x.to(self.device)
        y_true = y_true.to(self.device) if y_true is not None else None
        mask = mask.to(self.device)

        output = model(x, mask)

        y_true, output, mask = self._reshape_outputs(y_true, output, mask)
        y_pred = self._compute_predictions(output, y_true)

        return {"output": output, "y_pred": y_pred, "y_true": y_true}

    def _reshape_outputs(self, y_true, output, mask):
        y_true = y_true.reshape(-1)
        output = output.reshape(-1, output.shape[-1]).squeeze()
        mask_flat = mask.reshape(-1)
        y_true = y_true[mask_flat != 0]
        output = output[mask_flat != 0]
        return y_true, output, mask

    def _compute_predictions(self, output, y_true) -> torch.Tensor:
        if self.is_classification:
            return torch.sigmoid(output[:, 1])
        return output


class DomainAdaptationMixin:
    def _process_batch_with_domain(self, model, data, dataset_idx) -> Dict[str, torch.Tensor]:
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

        y_true, output, domain_true = self._reshape_with_domain(y_true, output, mask, domain_true)
        y_pred = self._compute_predictions(output, y_true)

        return {
            "output": output,
            "y_pred": y_pred,
            "y_true": y_true,
            "domain_pred": domain_pred,
            "domain_true": domain_true,
        }

    def _reshape_with_domain(self, y_true, output, mask, domain_true):
        batch_size, seq_len = mask.shape
        mask_flat = mask.reshape(-1)
        valid_mask = mask_flat != 0

        if len(y_true.shape) > 1:
            y_true = y_true.reshape(-1)

        if len(output.shape) > 1:
            output = output.reshape(-1, output.shape[-1]).squeeze()
        else:
            output = output.squeeze()

        domain_true_event = domain_true
        domain_true_hit = domain_true.unsqueeze(1).expand(batch_size, seq_len).reshape(-1)

        if y_true.shape[0] == mask_flat.shape[0]:
            y_true = y_true[valid_mask]
        if output.shape[0] == mask_flat.shape[0]:
            output = output[valid_mask]
        domain_true_hit = domain_true_hit[valid_mask]

        return y_true, output, (domain_true_event, domain_true_hit)
