from pathlib import Path
from typing import Any, Dict

import torch

from data_utils import (
    BaikalDatasetAngles,
    BaikalDatasetAnglesAndTrackCascade,
    BaikalDatasetAnglesOld,
    BaikalDatasetAnglesOldSingle,
    BaikalDatasetAnglesSingle,
)
from data_utils.preprocessors import (
    AngleAndTrackCascadePreprocessor,
    AngleGraphPreprocessor,
    AnglePreprocessor,
    AnglePreprocessorWithTres,
    DataPrefilter,
)
from metrics import AngleReconstructionMetrics, BinaryClassificationMetrics
from training.losses import (
    CosSimLoss,
    MAELoss,
    MSELoss,
    NLLUncertaintyLoss,
    RMSEVonMisesFisher3DLoss,
    VonMisesFisher3DLoss,
)

from .base import BaseTrainType, DomainAdaptationMixin


class AngleReconstructionTrainType(BaseTrainType):
    @property
    def name(self) -> str:
        return "angle_reconstruction"

    def __init__(self, train_params: Dict[str, Any], device: str = "cuda", save_dir: Path = None):
        super().__init__(train_params, device)
        self.save_dir = save_dir
        self.is_val_mode = train_params.get("val_mode", False)
        self.use_old = "old" in train_params.get("train_type", "")
        self._setup_criterion()

    def _setup_criterion(self):
        if self.train_params.get("use_vmf_loss", False):
            self._criterion = VonMisesFisher3DLoss()
        elif self.train_params.get("use_vmf_rmse_loss", False):
            self._criterion = RMSEVonMisesFisher3DLoss()
        elif self.train_params.get("use_mae_loss", False):
            self._criterion = MAELoss()
        elif self.train_params.get("use_mse_loss", False):
            self._criterion = MSELoss()
        else:
            self._criterion = MAELoss()

    def get_dataset_type(self):
        if self.is_graph:
            return BaikalDatasetAnglesOldSingle if self.use_old else BaikalDatasetAnglesSingle
        return BaikalDatasetAnglesOld if self.use_old else BaikalDatasetAngles

    def get_preprocessor(self, config: Dict[str, Any]):
        if self.use_old:
            return AnglePreprocessor()
        if self.is_graph:
            return AngleGraphPreprocessor(config["knn_neighbours"])
        data_prefilter = DataPrefilter(**(config.get("data_prefilter_params", {})))
        return AnglePreprocessorWithTres(config["tres_cut"], data_prefilter)

    def get_criterion(self):
        return self._criterion

    def get_metrics_function(self):
        return AngleReconstructionMetrics(save_preds=self.is_val_mode, save_dir=self.save_dir)

    def _process_batch(self, model, data, dataset_idx=None) -> Dict[str, torch.Tensor]:
        if len(data) == 4:
            data = data[:3]

        x, y_true, mask = data
        x = x.to(self.device)
        y_true = y_true.to(self.device) if y_true is not None else None
        mask = mask.to(self.device)

        output = model(x, mask)
        output = output / output.norm(dim=1, keepdim=True)

        return {"output": output, "y_pred": output, "y_true": y_true}


class AngleReconstructionDomainAdaptationTrainType(DomainAdaptationMixin, BaseTrainType):
    @property
    def name(self) -> str:
        return "angle_reconstruction_domain_adaptation"

    def __init__(self, train_params: Dict[str, Any], device: str = "cuda", save_dir: Path = None):
        super().__init__(train_params, device)
        self.save_dir = save_dir
        self.is_val_mode = train_params.get("val_mode", False)
        self.domain_adaptation_loss_k = train_params.get("domain_adaptation_loss_k", 0.0)
        self.label_dataset_name = train_params.get("label_dataset_name", None)
        self.predict_sigma = train_params.get("predict_sigma", False)
        self.nll_loss_k = train_params.get("nll_loss_k", 0.0)
        self.angle_loss_k = train_params.get("angle_loss_k", 1.0)
        self._setup_losses()

    def _setup_losses(self):
        if self.train_params.get("use_vmf_loss", False):
            self.loss_fn = VonMisesFisher3DLoss()
        elif self.train_params.get("use_vmf_rmse_loss", False):
            self.loss_fn = RMSEVonMisesFisher3DLoss()
        elif self.train_params.get("use_mae_loss", False):
            self.loss_fn = MAELoss()
        elif self.train_params.get("use_mse_loss", False):
            self.loss_fn = MSELoss()
        elif self.train_params.get("use_cos_loss", False):
            self.loss_fn = CosSimLoss()
        else:
            raise ValueError("Unknown loss function for angle_reconstruction_domain_adaptation")
        self.nll_loss_fn = NLLUncertaintyLoss(pred_size=3)

    def get_dataset_type(self):
        if self.is_graph:
            return BaikalDatasetAnglesSingle
        return BaikalDatasetAngles

    def get_preprocessor(self, config: Dict[str, Any]):
        if self.is_graph:
            return AngleGraphPreprocessor(config["knn_neighbours"])
        data_prefilter = DataPrefilter(**(config.get("data_prefilter_params", {})))
        return AnglePreprocessorWithTres(config["tres_cut"], data_prefilter)

    def get_criterion(self):
        dataset_names = self.dataset_names
        label_dataset_name = self.label_dataset_name
        loss_fn = self.loss_fn
        nll_loss_fn = self.nll_loss_fn
        domain_adaptation_loss_k = self.domain_adaptation_loss_k
        predict_sigma = self.predict_sigma
        nll_loss_k = self.nll_loss_k
        angle_loss_k = self.angle_loss_k

        def criterion(y_pred, y_true, domain_pred=None, domain_true=None, is_train=False):
            res = {}

            if domain_true is None:
                mask = torch.ones(y_true.shape[0], dtype=torch.bool)
            else:
                mask = domain_true == dataset_names.index(label_dataset_name)

            if is_train:
                assert (domain_true == 0).all()

            if predict_sigma and mask.sum() > 0:
                nll_loss = nll_loss_fn(y_pred[mask], y_true[mask])
                y_pred = y_pred[:, :3]
                y_true = y_true[:, :3]
                res["nll_loss"] = nll_loss
            else:
                nll_loss = torch.tensor(0.0, device=y_pred.device)

            if mask.sum() > 0:
                angle_loss = loss_fn(y_pred[mask], y_true[mask])
            else:
                angle_loss = torch.tensor(0.0, device=y_pred.device)

            if domain_pred is not None:
                domain_loss = torch.nn.functional.cross_entropy(domain_pred, domain_true)
            else:
                domain_loss = torch.tensor(0.0, device=y_pred.device)

            total_loss = (
                angle_loss_k * angle_loss
                + domain_adaptation_loss_k * domain_loss
                + nll_loss_k * nll_loss
            )
            res["loss"] = total_loss
            res["angle_loss"] = angle_loss.detach()
            res["domain_loss"] = domain_loss.detach()
            res["nll_loss"] = nll_loss.detach()
            return res

        return criterion

    def get_metrics_function(self):
        return AngleReconstructionMetrics(save_preds=self.is_val_mode, save_dir=self.save_dir)

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
        output = output / output.norm(dim=1, keepdim=True)

        batch_size = x.shape[0]
        domain_true = torch.full((batch_size,), dataset_idx, dtype=torch.long, device=self.device)

        return {
            "output": output,
            "y_pred": output,
            "y_true": y_true,
            "domain_pred": domain_pred,
            "domain_true": domain_true,
        }


class AngleAndTrackCascadeTrainType(BaseTrainType):
    @property
    def name(self) -> str:
        return "angle_and_track_cascade"

    def __init__(self, train_params: Dict[str, Any], device: str = "cuda", save_dir: Path = None):
        super().__init__(train_params, device)
        self.save_dir = save_dir
        self.is_val_mode = train_params.get("val_mode", False)
        self.mse_coef = train_params.get("mse_coef", 1.0)
        self.ce_ = torch.nn.CrossEntropyLoss()
        self.mse_ = torch.nn.MSELoss()

    def get_dataset_type(self):
        return BaikalDatasetAnglesAndTrackCascade

    def get_preprocessor(self, config: Dict[str, Any]):
        return AngleAndTrackCascadePreprocessor(config["tres_cut"])

    def get_criterion(self):
        ce_ = self.ce_
        mse_ = self.mse_
        mse_coef = self.mse_coef

        def criterion(output, y_true):
            ce_loss = ce_(output[:, :2], y_true[:, 0].long())
            mse_loss = mse_(output[:, 0, 2:].reshape(-1, 2), y_true[:, :2])
            return ce_loss + mse_coef * mse_loss

        return criterion

    def get_metrics_function(self):
        return BinaryClassificationMetrics(save_preds=self.is_val_mode, save_dir=self.save_dir)

    def _process_batch(self, model, data, dataset_idx=None) -> Dict[str, torch.Tensor]:
        if len(data) == 4:
            data = data[:3]

        x, y_true, mask = data
        x = x.to(self.device)
        y_true = y_true.to(self.device) if y_true is not None else None
        mask = mask.to(self.device)

        output = model(x, mask)

        y_pred = torch.zeros_like(y_true)
        y_pred[:, 1] = output[:, -2]
        y_pred[:, 2] = output[:, -1]
        y_pred[:, 0] = torch.sigmoid(output[:, 1])

        return {"output": output, "y_pred": y_pred, "y_true": y_true}
