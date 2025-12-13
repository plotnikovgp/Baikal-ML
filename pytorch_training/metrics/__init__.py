from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)

THRESHOLD = 0.5


class BaseMetrics:
    def __init__(self, save_preds=False, save_dir: Path = None, dataset_name: str = None):
        self.save_preds = save_preds
        self.save_dir = save_dir
        self.dataset_name = dataset_name
        self.data_to_save = {}

    def set_dataset_name(self, dataset_name: str):
        self.dataset_name = dataset_name

    def _calc_metrics(self, y_pred, y_true, **kwargs) -> dict:
        raise NotImplementedError

    def _save_preds(self):
        if not self.save_dir or not self.data_to_save:
            return
        data_dir = self.save_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        for name, data in self.data_to_save.items():
            np.savetxt(data_dir / f"{self.dataset_name}_{name}.txt", np.array(data))
        self.data_to_save = {}

    def __call__(self, y_pred, y_true, **kwargs) -> dict:
        metrics = self._calc_metrics(y_pred, y_true, **kwargs)
        if self.save_preds:
            self._save_preds()
        return metrics


def _extract_angles(vector):
    x, y, z = vector
    theta = np.arccos(z)
    phi = np.arctan2(y, x)
    return np.rad2deg(theta), np.rad2deg(phi)


class BinaryClassificationMetrics(BaseMetrics):
    def __init__(self, threshold=THRESHOLD, min_recall=None, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.min_recall = min_recall

    def _calc_metrics(self, y_pred_prob, y_true, **kwargs) -> dict:
        y_pred_prob = np.array(y_pred_prob, dtype=np.float32)
        y_true = np.array(y_true, dtype=np.int32)
        threshold = self.threshold

        if self.min_recall is not None:
            sorted_idx = np.argsort(y_pred_prob)[::-1]
            sorted_true = y_true[sorted_idx]
            sorted_prob = y_pred_prob[sorted_idx]

            total_pos = np.sum(y_true)
            if total_pos > 0:
                recalls = np.cumsum(sorted_true) / total_pos
                valid_idx = np.where(recalls >= self.min_recall)[0]
                if len(valid_idx) > 0:
                    threshold = sorted_prob[valid_idx[0]]

        y_pred = (y_pred_prob > threshold).astype(np.int32)

        return {
            "auc": roc_auc_score(y_true, y_pred_prob) if len(np.unique(y_true)) > 1 else 0.0,
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "threshold": float(threshold),
            "pos_class_ratio_pred": float(np.mean(y_pred)),
            "pos_class_ratio_true": float(np.mean(y_true)),
        }


class MulticlassClassificationMetrics(BaseMetrics):
    def _calc_metrics(self, y_pred_prob, y_true, **kwargs) -> dict:
        y_pred_prob = np.array(y_pred_prob, dtype=np.float32)
        y_true = np.array(y_true, dtype=np.int32)
        y_pred = np.argmax(y_pred_prob, axis=1)

        return {
            "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "accuracy": accuracy_score(y_true, y_pred),
        }


class RegressionMetrics(BaseMetrics):
    def _calc_metrics(self, y_pred, y_true, **kwargs) -> dict:
        y_pred = np.array(y_pred, dtype=np.float32)
        y_true = np.array(y_true, dtype=np.float32)
        diff = np.abs(y_true - y_pred)

        metrics = {
            "mae": mean_absolute_error(y_true, y_pred),
            "mse": mean_squared_error(y_true, y_pred),
            "mape": mean_absolute_percentage_error(y_true, y_pred),
        }
        if len(diff.shape) == 1:
            metrics["q50"] = float(np.quantile(diff, 0.5))
            metrics["q68"] = float(np.quantile(diff, 0.68))
        return metrics


class EnergyMetrics(BaseMetrics):
    def _calc_metrics(self, y_pred, y_true, **kwargs) -> dict:
        y_pred = np.array(y_pred, dtype=np.float32)
        y_true = np.array(y_true, dtype=np.float32)

        if self.save_preds:
            self.data_to_save = {"true_E": y_true, "pred_E": y_pred}

        return RegressionMetrics()._calc_metrics(y_pred, y_true)


class AngleReconstructionMetrics(BaseMetrics):
    def _calc_metrics(self, y_pred, y_true, **kwargs) -> dict:
        y_pred = np.array(y_pred, dtype=np.float32)
        y_true = np.array(y_true, dtype=np.float32)
        metrics = {}

        if y_pred.shape[1] == 6:
            log_sigma2 = y_pred[:, 3:]
            sigma_pred = np.sqrt(np.exp(log_sigma2))
            y_pred = y_pred[:, :3]
            sigma_true = (y_pred - y_true) ** 2
            sigma_metrics = RegressionMetrics()._calc_metrics(
                sigma_pred.mean(1), sigma_true.mean(1)
            )
            metrics.update({"sigma2_" + k: v for k, v in sigma_metrics.items()})

        kappa = None
        if y_pred.shape[1] == 4:
            kappa = y_pred[:, 3]
            y_pred = y_pred[:, :3]

        metrics.update(RegressionMetrics()._calc_metrics(y_pred, y_true))

        angles_true = np.array([_extract_angles(v) for v in y_true])
        angles_pred = np.array([_extract_angles(v) for v in y_pred])

        theta_true, phi_true = angles_true[:, 0], angles_true[:, 1]
        theta_pred, phi_pred = angles_pred[:, 0], angles_pred[:, 1]

        if not np.isnan(theta_pred).any() and not np.isnan(phi_pred).any():
            metrics["theta_mae"] = mean_absolute_error(theta_true, theta_pred)
            metrics["phi_mae"] = mean_absolute_error(phi_true, phi_pred)

            theta_res = np.abs(theta_true - theta_pred)
            phi_res = np.abs(phi_true - phi_pred)

            dot = np.clip(np.sum(y_true * y_pred, axis=1), -1.0, 1.0)
            dir_res = np.abs(np.rad2deg(np.arccos(dot)))

            metrics["theta_resolution_q50"] = np.quantile(theta_res, 0.5)
            metrics["theta_resolution_q68"] = np.quantile(theta_res, 0.68)
            metrics["phi_resolution_q50"] = np.quantile(phi_res, 0.5)
            metrics["phi_resolution_q68"] = np.quantile(phi_res, 0.68)
            metrics["dir_resolution_q50"] = np.quantile(dir_res, 0.5)
            metrics["dir_resolution_q68"] = np.quantile(dir_res, 0.68)

            if self.save_preds:
                self.data_to_save = {
                    "true_theta": theta_true,
                    "true_phi": phi_true,
                    "pred_theta": theta_pred,
                    "pred_phi": phi_pred,
                    "dir_resolution": dir_res,
                    "theta_resolution": theta_res,
                    "phi_resolution": phi_res,
                }

        if kappa is not None:
            cos_sim = np.sum(y_true * y_pred, axis=1)
            angular_error = np.arccos(np.clip(cos_sim, -1, 1))
            metrics["kappa_pearson"] = pearsonr(kappa, angular_error)[0]
            metrics["kappa_spearman"] = spearmanr(kappa, angular_error)[0]

        return {k: float(v) for k, v in metrics.items()}


class TrackCascadeMetrics(BaseMetrics):
    def __init__(self, threshold=THRESHOLD, min_recall=None, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.min_recall = min_recall

    def _calc_metrics(self, y_pred, y_true, **kwargs) -> dict:
        y_true = np.array(y_true, dtype=bool)

        cascade_metrics = BinaryClassificationMetrics(
            threshold=self.threshold, min_recall=self.min_recall
        )._calc_metrics(y_pred, y_true)

        threshold = cascade_metrics.get("threshold", self.threshold)
        y_pred_class = y_pred > threshold

        track_metrics = BinaryClassificationMetrics(threshold=1 - self.threshold)._calc_metrics(
            1 - y_pred, ~y_true
        )

        metrics = {k + "[cascade=1]": v for k, v in cascade_metrics.items()}
        metrics.update({k + "[track=1]": v for k, v in track_metrics.items()})
        metrics["n_cascade/n_track_pred"] = y_pred_class.sum() / max((~y_pred_class).sum(), 1)
        metrics["n_cascade/n_track_true"] = y_true.sum() / max((~y_true).sum(), 1)

        return {k: float(v) for k, v in metrics.items()}


class TresAndTrackCascadeMetrics(BaseMetrics):
    def __init__(self, min_recall=None, **kwargs):
        super().__init__(**kwargs)
        self.min_recall = min_recall

    def _calc_metrics(self, y_pred, y_true, **kwargs) -> dict:
        metrics = TrackCascadeMetrics(min_recall=self.min_recall)._calc_metrics(
            y_pred[:, 0].reshape(-1), y_true[:, 0].reshape(-1)
        )
        metrics.update(
            RegressionMetrics()._calc_metrics(y_pred[:, 1].reshape(-1), y_true[:, 1].reshape(-1))
        )
        return metrics


class DirectionMetrics(BaseMetrics):
    def _calc_distance(self, point1, point2, angle1, angle2):
        w = point1 - point2
        n = np.cross(angle1, angle2)
        norm_n = np.linalg.norm(n, axis=1, keepdims=True)
        parallel = norm_n < 1e-6
        dist_parallel = np.linalg.norm(np.cross(w, angle1), axis=1, keepdims=True)
        dist_normal = np.abs(np.sum(w * n, axis=1, keepdims=True)) / np.maximum(norm_n, 1e-6)
        return float(np.where(parallel, dist_parallel, dist_normal).mean())

    def _calc_metrics(self, y_pred, y_true, **kwargs) -> dict:
        y_pred = np.array(y_pred, dtype=np.float32)
        y_true = np.array(y_true, dtype=np.float32)

        angle_pred, angle_true = y_pred[:, :3], y_true[:, :3]
        point_pred, point_true = y_pred[:, 3:], y_true[:, 3:]

        metrics = AngleReconstructionMetrics()._calc_metrics(angle_pred, angle_true)
        metrics["direction_distance"] = self._calc_distance(
            point_pred, point_true, angle_pred, angle_true
        )

        return metrics
