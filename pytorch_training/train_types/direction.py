from pathlib import Path
from typing import Any, Dict

import torch

from data_utils import BaikalDatasetAngles
from data_utils.preprocessors import AnglePreprocessorWithTres, DataPrefilter
from metrics import DirectionMetrics

from .base import BaseTrainType


class DirectionTrainType(BaseTrainType):
    @property
    def name(self) -> str:
        return "direction"

    def __init__(self, train_params: Dict[str, Any], device: str = "cuda", save_dir: Path = None):
        super().__init__(train_params, device)
        self.save_dir = save_dir
        self.is_val_mode = train_params.get("val_mode", False)
        self._criterion = torch.nn.MSELoss()

    def get_dataset_type(self):
        return BaikalDatasetAngles

    def get_preprocessor(self, config: Dict[str, Any]):
        data_prefilter = DataPrefilter(**(config.get("data_prefilter_params", {})))
        return AnglePreprocessorWithTres(config["tres_cut"], data_prefilter)

    def get_criterion(self):
        return self._criterion

    def get_metrics_function(self):
        return DirectionMetrics(save_preds=self.is_val_mode, save_dir=self.save_dir)

    def _process_batch(self, model, data, dataset_idx=None) -> Dict[str, torch.Tensor]:
        if len(data) == 4:
            data = data[:3]

        x, y_true, mask = data
        x = x.to(self.device)
        y_true = y_true.to(self.device) if y_true is not None else None
        mask = mask.to(self.device)

        output = model(x, mask)

        norms = output[:, :3].norm(dim=-1, keepdim=True)
        normalized_values = output[:, :3] / norms
        output = torch.cat((normalized_values, output[:, 3:]), dim=-1)

        return {"output": output, "y_pred": output, "y_true": y_true}
