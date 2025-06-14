import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    accuracy_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_percentage_error,
    mean_absolute_error,
)
import traceback
from scipy.spatial.distance import cosine as cosine_dist
import logging
from scipy.stats import spearmanr, pearsonr
from pathlib import Path
import matplotlib.pyplot as plt
from .plots import *
from .uncertainty_metrics import *
from data_utils.preprocessors import DataPrefilter

THRESHOLD = 0.5


class BaseMetrics:
    def __init__(
        self,
        save_preds=False,
        save_dir: Path | None = None,
        dataset_name: str | None = None,
        plot_metrics=False,
        **kwargs,
    ):
        self.save_preds = save_preds
        self.save_dir = save_dir
        self.dataset_name = dataset_name
        self.plot_metrics = plot_metrics
        self.data_to_save = {}
        self.dataset_name = dataset_name

    def set_dataset_name(self, dataset_name: str):
        self.dataset_name = dataset_name

    def set_plot_metrics(self, plot_metrics: bool):
        self.plot_metrics = plot_metrics

    def _calc_metrics(self, y_pred, y_true, **kwargs):
        pass

    def __call__(self, y_pred, y_true, **kwargs):
        pass

    def _plot(self, **kwargs):
        pass

    def _save_preds(self, **kwargs):
        for name, data in self.data_to_save.items():
            if not isinstance(data, np.ndarray):
                data = np.array(data)
            if not (self.save_dir / "data").exists():
                (self.save_dir / "data").mkdir(parents=True, exist_ok=True)
            np.savetxt(self.save_dir / "data" / f"{self.dataset_name}_{name}.txt", data)
        self.data_to_save = {}

    def __call__(self, y_pred, y_true, **kwargs):
        metrics = self._calc_metrics(y_pred, y_true, **kwargs)

        if self.plot_metrics:
            self._plot(**kwargs)

        if self.save_preds:
            self._save_preds(**kwargs)

        return metrics


def roc_auc_score_safe(y_true, y_pred):
    if np.unique(y_true).size == 1:
        return -1.0
    else:
        return roc_auc_score(y_true, y_pred)


def extract_angles(vector):
    x, y, z = vector
    theta = np.arccos(z)
    phi = np.arctan2(y, x)

    # Convert from radians to degrees
    theta_deg = np.rad2deg(theta)
    phi_deg = np.rad2deg(phi)

    return theta_deg, phi_deg


def binary_clf_metrics(y_pred_prob, y_true, threshold=THRESHOLD, min_recall=None):
    y_pred_prob = np.array(y_pred_prob, dtype=np.float32)
    y_true = np.array(y_true, dtype=np.int32)

    # If min_recall is specified, find the threshold that gives at least that recall
    if min_recall is not None:
        sorted_indices = np.argsort(y_pred_prob)
        sorted_indices = np.flip(sorted_indices)

        sorted_y_true = y_true[sorted_indices]
        sorted_y_pred_prob = y_pred_prob[sorted_indices]

        true_positives = np.cumsum(sorted_y_true)
        total_positives = np.sum(y_true)

        if total_positives > 0:
            recalls = true_positives / total_positives

            valid_indices = np.where(recalls >= min_recall)[0]
            if len(valid_indices) > 0:
                min_valid_index = valid_indices[0]
                threshold = sorted_y_pred_prob[min_valid_index]
                logging.info(
                    f"Using threshold {threshold} to achieve minimum recall of {min_recall}"
                )
            else:
                # If no threshold achieves min_recall, use the lowest threshold
                threshold = np.min(y_pred_prob) - 1e-6
                logging.warning(
                    f"Could not find threshold for min_recall={min_recall}, using {threshold}"
                )

    y_pred = np.array(y_pred_prob > threshold, dtype=np.int32)

    try:
        metrics = {
            "auc": roc_auc_score_safe(y_true, y_pred_prob),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "threshold": float(threshold),
        }
        return metrics
    except ValueError:
        traceback.print_exc()
        raise


def multiclass_clf_metrics(y_pred_prob, y_true, min_recall=None):
    y_pred_prob = np.array(y_pred_prob, dtype=np.float32)
    y_true = np.array(y_true, dtype=np.int32)
    if min_recall is not None:
        pass

    y_pred = np.argmax(y_pred_prob, axis=1)
    metrics = {
        "precision_macro": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "accuracy_m": accuracy_score(y_true, y_pred),
    }
    return metrics


def regression_metrics(y_pred, y_true):
    y_pred = np.array(y_pred, dtype=np.float32)
    y_true = np.array(y_true, dtype=np.float32)
    try:
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
    except ValueError:
        traceback.print_exc()
        raise


class EnergyMetrics(BaseMetrics):
    def _calc_metrics(self, y_pred, y_true, **kwargs):
        metrics = regression_metrics(y_pred, y_true)
        if self.save_preds:
            self.data_to_save = {
                "true_E": y_true,
                "pred_E": y_pred,
            }
        return metrics


class AngleReconstructionMetrics(BaseMetrics):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # self.uncertainty_metrics = AngleUncertaintyMetrics(**kwargs)
        self.uncertainty_metrics = AngleUncertaintyMetricsV2(**kwargs)
        self.data_transform = None

    def set_plot_metrics(self, plot_metrics: bool):
        super().set_plot_metrics(plot_metrics)
        self.uncertainty_metrics.set_plot_metrics(plot_metrics)

    def set_dataset_name(self, dataset_name: str):
        super().set_dataset_name(dataset_name)
        self.uncertainty_metrics.set_dataset_name(dataset_name)

    def _calc_metrics(
        self, y_pred, y_true, additional_data: dict | None = None, **kwargs
    ):
        if self.data_transform is not None:
            y_pred = self.data_transform[self.dataset_name].data_prefilter.postprocess(
                y_pred
            )
            y_true = self.data_transform[self.dataset_name].data_prefilter.postprocess(
                y_true
            )

        y_pred = np.array(y_pred, dtype=np.float32)
        y_true = np.array(y_true, dtype=np.float32)
        metrics = {}

        # TODO: fix this
        if y_pred.shape[1] == 6:
            uncertainty_metrics = self.uncertainty_metrics(y_pred, y_true)
            y_pred = y_pred[:, :3]
            y_true = y_true[:, :3]
            metrics.update(uncertainty_metrics)

        kappa = None  # vmf loss
        if y_pred.shape[1] == 4:
            kappa = y_pred[:, 3]
            y_pred = y_pred[:, :3]

        metrics.update(regression_metrics(y_pred, y_true))
        angles_true = np.array(
            [extract_angles(vec) for vec in y_true], dtype=np.float32
        )
        angles_pred = np.array(
            [extract_angles(vec) for vec in y_pred], dtype=np.float32
        )
        y_true_theta_angle, y_true_phi_angle = angles_true[:, 0], angles_true[:, 1]
        y_pred_theta_angle, y_pred_phi_angle = angles_pred[:, 0], angles_pred[:, 1]

        if (
            not np.isnan(y_pred_theta_angle).any()
            and not np.isnan(y_pred_phi_angle).any()
        ):
            metrics["theta_mae"] = mean_absolute_error(
                y_true_theta_angle, y_pred_theta_angle
            )
            metrics["phi_mae"] = mean_absolute_error(y_true_phi_angle, y_pred_phi_angle)
            theta_resolution = np.abs(y_true_theta_angle - y_pred_theta_angle)
            phi_resolution = np.abs(y_true_phi_angle - y_pred_phi_angle)

            dot_product = np.sum(y_true * y_pred, axis=1)
            dot_product = np.clip(dot_product, -1.0, 1.0)
            dir_resolution = np.abs(np.rad2deg(np.arccos(dot_product)))
            metrics["theta_resolution_q50"] = np.quantile(theta_resolution, 0.5)
            metrics["theta_resolution_q68"] = np.quantile(theta_resolution, 0.68)
            metrics["phi_resolution_q50"] = np.quantile(phi_resolution, 0.5)
            metrics["phi_resolution_q68"] = np.quantile(phi_resolution, 0.68)
            metrics["dir_resolution_q50"] = np.quantile(dir_resolution, 0.5)
            metrics["dir_resolution_q68"] = np.quantile(dir_resolution, 0.68)

            if self.save_preds:
                self.data_to_save = {
                    "true_theta": y_true_theta_angle,
                    "true_phi": y_true_phi_angle,
                    "pred_theta": y_pred_theta_angle,
                    "pred_phi": y_pred_phi_angle,
                    "dir_resolution": dir_resolution,
                    "theta_resolution": theta_resolution,
                    "phi_resolution": phi_resolution,
                }

        if kappa is not None:
            cos_sim = np.sum(y_true * y_pred, axis=1)
            angular_error = np.arccos(np.clip(cos_sim, -1, 1))
            metrics["kappa_pearson"] = pearsonr(kappa, angular_error)[0]
            metrics["kappa_spearman"] = spearmanr(kappa, angular_error)[0]
        return {k: float(v) for k, v in metrics.items()}


def calculate_distance(point1, point2, angle1, angle2):
    w = point1 - point2
    n = np.cross(angle1, angle2)
    norm_n = np.linalg.norm(n, axis=1, keepdims=True)

    # Handle cases where angles are nearly parallel
    parallel_mask = norm_n < 1e-6

    # For nearly parallel angles, use the alternative formula
    distance_parallel = np.linalg.norm(np.cross(w, angle1), axis=1, keepdims=True)

    # For non-parallel angles, use the standard formula
    distance_normal = np.abs(np.sum(w * n, axis=1, keepdims=True)) / np.maximum(
        norm_n, 1e-6
    )

    # Combine results based on the parallel mask
    distance_loss = np.where(parallel_mask, distance_parallel, distance_normal)

    return float(distance_loss.mean())


def direction_metrics(y_pred, y_true):
    MEAN = np.array(
        [
            1.6824790239334106,
            -4.7245340084600684e-08,
            0.9185771346092224,
            -0.2642837464809418,
            20.012351989746094,
        ]
    )
    STD = np.array([5.9653406, 1343.6559, 39.985634, 39.049866, 154.23203])
    y_pred = np.array(y_pred, dtype=np.float32)
    y_true = np.array(y_true, dtype=np.float32)
    angle_pred = y_pred[:, :3]
    angle_true = y_true[:, :3]

    point_pred = y_pred[:, 3:]
    point_true = y_true[:, 3:]
    angle_metrics = angle_reconstruction_metrics(angle_pred, angle_true)
    point_pred_init = point_pred * STD[2:] + MEAN[2:]
    point_true_init = point_true * STD[2:] + MEAN[2:]
    metrics = {k: v for k, v in angle_metrics.items()}
    metrics["direction_norm"] = calculate_distance(
        point_pred, point_true, angle_pred, angle_true
    )
    metrics["direction_meters"] = calculate_distance(
        point_pred_init, point_true_init, angle_pred, angle_true
    )
    return metrics


def cartesian_to_spherical_uncertainty(pred, pred_sigma2):
    """
    Convert Cartesian uncertainties to spherical (theta, phi) uncertainties
    """
    x, y, z = pred[:, 0], pred[:, 1], pred[:, 2]
    var_x, var_y, var_z = pred_sigma2[:, 0], pred_sigma2[:, 1], pred_sigma2[:, 2]

    # Calculate intermediate values
    r = np.sqrt(x**2 + y**2 + z**2)
    r_sq = r**2
    xy = np.sqrt(x**2 + y**2)  # Distance in xy-plane

    # Add small epsilon to prevent division by zero
    epsilon = 1e-10
    r = np.maximum(r, epsilon)
    xy = np.maximum(xy, epsilon)

    # Derivatives of theta with respect to x, y, z
    # theta = arccos(z/r)
    dtheta_dx = x * z / (r_sq * xy)
    dtheta_dy = y * z / (r_sq * xy)
    dtheta_dz = -xy / r_sq

    # Variance of theta using error propagation formula
    theta_var = (dtheta_dx**2 * var_x) + (dtheta_dy**2 * var_y) + (dtheta_dz**2 * var_z)

    # For phi uncertainty (derivative of arctan2(y, x) with respect to x, y)
    denom = x**2 + y**2
    denom = np.maximum(denom, epsilon)  # Prevent division by zero

    # Derivatives of phi with respect to x, y (z doesn't affect phi)
    dphi_dx = -y / denom
    dphi_dy = x / denom

    # Variance of phi using error propagation formula
    phi_var = (dphi_dx**2 * var_x) + (dphi_dy**2 * var_y)

    # Convert to degrees if needed
    rad_to_deg = 180.0 / np.pi
    theta_var_deg = theta_var * (rad_to_deg**2)
    phi_var_deg = phi_var * (rad_to_deg**2)

    return theta_var_deg, phi_var_deg


class AngleUncertaintyMetrics(BaseMetrics):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.plotter = AngleUncertaintyPlotter(save_dir=self.save_dir)

    def _plot(self, **kwargs):
        self.plotter.save_dir = self.save_dir / "plots" / self.dataset_name
        if not self.plotter.save_dir.exists():
            self.plotter.save_dir.mkdir(parents=True, exist_ok=True)
        self.plotter.plot(data=self.data_to_save, **kwargs)

    def _calc_metrics(self, y_pred_and_log_sigma, y_true):
        y_pred, log_pred_sigma2 = (
            y_pred_and_log_sigma[:, :3],
            y_pred_and_log_sigma[:, 3:],
        )

        y_pred = np.array(y_pred, dtype=np.float32)
        y_true = np.array(y_true, dtype=np.float32)
        predicted_sigma2 = np.exp(np.array(log_pred_sigma2, dtype=np.float32))
        true_sigma2 = (y_true - y_pred) ** 2

        x_sigma_mae = np.mean(np.abs(predicted_sigma2[:, 0] - true_sigma2[:, 0]))
        y_sigma_mae = np.mean(np.abs(predicted_sigma2[:, 1] - true_sigma2[:, 1]))
        z_sigma_mae = np.mean(np.abs(predicted_sigma2[:, 2] - true_sigma2[:, 2]))

        angles_true = np.array(
            [extract_angles(vec) for vec in y_true], dtype=np.float32
        )
        angles_pred = np.array(
            [extract_angles(vec) for vec in y_pred], dtype=np.float32
        )

        y_true_theta_angle, y_true_phi_angle = angles_true[:, 0], angles_true[:, 1]
        y_pred_theta_angle, y_pred_phi_angle = angles_pred[:, 0], angles_pred[:, 1]

        pred_theta_sigma2, pred_phi_sigma2 = cartesian_to_spherical_uncertainty(
            y_pred, predicted_sigma2
        )
        true_theta_sigma2, true_phi_sigma2 = cartesian_to_spherical_uncertainty(
            y_true, true_sigma2
        )

        true_theta_sigma2_v2 = (y_true_theta_angle - y_pred_theta_angle) ** 2
        true_phi_sigma2_v2 = (y_true_phi_angle - y_pred_phi_angle) ** 2

        theta_sigma_mae = np.mean(np.abs(pred_theta_sigma2 - true_theta_sigma2))
        phi_sigma_mae = np.mean(np.abs(pred_phi_sigma2 - true_phi_sigma2))

        self.data_to_save = {
            "y_pred": y_pred,
            "y_true": y_true,
            "true_theta": y_true_theta_angle,
            "true_phi": y_true_phi_angle,
            "pred_theta": y_pred_theta_angle,
            "pred_phi": y_pred_phi_angle,
            "pred_sigma2": predicted_sigma2,
            "true_sigma2": true_sigma2,
            "pred_theta_sigma2": pred_theta_sigma2,
            "pred_phi_sigma2": pred_phi_sigma2,
            "true_theta_sigma2": true_theta_sigma2,
            "true_phi_sigma2": true_phi_sigma2,
            "true_theta_sigma2_v2": true_theta_sigma2_v2,
            "true_phi_sigma2_v2": true_phi_sigma2_v2,
        }
        metrics = {
            "err_mae": (predicted_sigma2 - true_sigma2).mean(),
            "x_sigma_mae": x_sigma_mae,
            "y_sigma_mae": y_sigma_mae,
            "z_sigma_mae": z_sigma_mae,
            "theta_sigma_mae": theta_sigma_mae,
            "phi_sigma_mae": phi_sigma_mae,
            "theta_msll": calculate_msll(
                y_true_theta_angle, y_pred_theta_angle, pred_theta_sigma2
            ),
            "phi_msll": calculate_msll(
                y_true_phi_angle, y_pred_phi_angle, pred_phi_sigma2
            ),
            "theta_picp_68": calculate_picp(
                y_true_theta_angle,
                y_pred_theta_angle,
                pred_theta_sigma2,
                confidence=0.68,
            ),
            "phi_picp_68": calculate_picp(
                y_true_phi_angle, y_pred_phi_angle, pred_phi_sigma2, confidence=0.68
            ),
            "theta_mce": calculate_mce(
                y_true_theta_angle, y_pred_theta_angle, pred_theta_sigma2
            ),
            "phi_mce": calculate_mce(
                y_true_phi_angle, y_pred_phi_angle, pred_phi_sigma2
            ),
        }

        return {k: float(v) for k, v in metrics.items()}


class AngleUncertaintyMetricsV2(BaseMetrics):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.plotter = AngleUncertaintyPlotter(save_dir=self.save_dir)

    def _plot(self, **kwargs):
        self.plotter.save_dir = self.save_dir / "plots" / self.dataset_name
        if not self.plotter.save_dir.exists():
            self.plotter.save_dir.mkdir(parents=True, exist_ok=True)
        self.plotter.plot(data=self.data_to_save, **kwargs)

    def _calc_metrics(self, y_pred_and_log_sigma, y_true):
        y_pred, log_pred_theta_phi_sigma2 = (
            y_pred_and_log_sigma[:, :3],
            y_pred_and_log_sigma[:, 3:],
        )
        # This gives variances in radians^2 because your loss works in radians
        pred_theta_phi_sigma2 = np.exp(
            np.array(log_pred_theta_phi_sigma2, dtype=np.float32)
        )

        y_pred = np.array(y_pred, dtype=np.float32)

        def extract_angles_rad_vectorized(vectors):
            x, y, z = vectors[:, 0], vectors[:, 1], vectors[:, 2]
            theta = np.arccos(np.clip(z, -1.0, 1.0))
            phi = np.arctan2(y, x)

            return np.column_stack((theta, phi))

        angles_true_rad = extract_angles_rad_vectorized(y_true)
        angles_pred_rad = extract_angles_rad_vectorized(y_pred)
        y_true_theta_rad, y_true_phi_rad = angles_true_rad[:, 0], angles_true_rad[:, 1]
        y_pred_theta_rad, y_pred_phi_rad = angles_pred_rad[:, 0], angles_pred_rad[:, 1]

        theta_diff = y_true_theta_rad - y_pred_theta_rad
        phi_diff = y_true_phi_rad - y_pred_phi_rad
        # Handle circular nature of phi in radians
        phi_diff = np.arctan2(np.sin(phi_diff), np.cos(phi_diff))

        true_theta_sigma2 = theta_diff**2
        true_phi_sigma2 = phi_diff**2

        pred_theta_sigma2 = pred_theta_phi_sigma2[:, 0]
        pred_phi_sigma2 = pred_theta_phi_sigma2[:, 1]

        # Convert to degrees for display purposes only
        rad_to_deg = 180.0 / np.pi
        y_true_theta_angle = y_true_theta_rad * rad_to_deg
        y_true_phi_angle = y_true_phi_rad * rad_to_deg
        y_pred_theta_angle = y_pred_theta_rad * rad_to_deg
        y_pred_phi_angle = y_pred_phi_rad * rad_to_deg

        # Store data for plotting - can be in degrees for display
        self.data_to_save = {
            "y_pred": y_pred,
            "y_true": y_true,
            "true_theta": y_true_theta_angle,
            "true_phi": y_true_phi_angle,
            "pred_theta": y_pred_theta_angle,
            "pred_phi": y_pred_phi_angle,
            "pred_theta_sigma2": pred_theta_sigma2 * rad_to_deg**2,
            "pred_phi_sigma2": pred_phi_sigma2 * rad_to_deg**2,
            "true_theta_sigma2": true_theta_sigma2 * rad_to_deg**2,
            "true_phi_sigma2": true_phi_sigma2 * rad_to_deg**2,
        }

        metrics = {
            "theta_msll": calculate_msll(
                y_true_theta_angle, y_pred_theta_angle, pred_theta_sigma2
            ),
            "phi_msll": calculate_msll(
                y_true_phi_angle, y_pred_phi_angle, pred_phi_sigma2
            ),
            "theta_picp_68": calculate_picp(
                y_true_theta_angle,
                y_pred_theta_angle,
                pred_theta_sigma2,
                confidence=0.68,
            ),
            "phi_picp_68": calculate_picp(
                y_true_phi_angle, y_pred_phi_angle, pred_phi_sigma2, confidence=0.68
            ),
            "theta_mce": calculate_mce(
                y_true_theta_angle, y_pred_theta_angle, pred_theta_sigma2
            ),
            "phi_mce": calculate_mce(
                y_true_phi_angle, y_pred_phi_angle, pred_phi_sigma2
            ),
            "theta_sigma_mae": np.mean(np.abs(pred_theta_sigma2 - true_theta_sigma2)),
            "phi_sigma_mae": np.mean(np.abs(pred_phi_sigma2 - true_phi_sigma2)),
        }
        return {k: float(v) for k, v in metrics.items()}


def regression_and_clf_metrics(y_pred, y_true, min_recall=None):
    metrics = {}
    metrics.update(
        binary_clf_metrics(
            y_pred[:, :, 2].reshape(-1),
            y_true[:, :, 2].reshape(-1),
            min_recall=min_recall,
        )
    )
    metrics.update(
        regression_metrics(y_pred[:, 0, :2].reshape(-1), y_true[:, 0, :2].reshape(-1))
    )
    return metrics


def angle_and_track_cascade_metrics(
    y_pred, y_true, angles_pred, angles_true, min_recall=None
):
    metrics = {}
    metrics.update(
        binary_clf_metrics(
            y_pred.reshape(-1), y_true.reshape(-1), min_recall=min_recall
        )
    )
    metrics.update(angle_reconstruction_metrics(angles_pred, angles_true))
    return metrics


def track_cascade_clf_metrics(y_pred, y_true, threshold=THRESHOLD, min_recall=None):
    metrics = {}
    y_true = np.array(y_true, dtype=bool)

    # Use min_recall for the cascade metrics
    cascade_metrics = binary_clf_metrics(
        y_pred, y_true, threshold, min_recall=min_recall
    )
    y_pred_class = np.array(
        y_pred > cascade_metrics.get("threshold", threshold), dtype=bool
    )

    metrics = {k + "[cascade=1]": v for k, v in cascade_metrics.items()}

    # For track metrics (inverse of cascade), we don't use min_recall
    metrics.update(
        {
            k + "[track=1]": v
            for k, v in binary_clf_metrics(1 - y_pred, ~y_true, 1 - threshold).items()
        }
    )
    metrics.update(
        {"n_cascade/n_track_pred": y_pred_class.sum() / (~y_pred_class).sum()}
    )
    metrics.update({"n_cascade/n_track_true": y_true.sum() / (~y_true).sum()})

    return {k: float(v) for k, v in metrics.items()}


def tres_and_track_cascade_metrics(y_pred, y_true, min_recall=None):
    metrics = {}
    metrics.update(
        track_cascade_clf_metrics(
            y_pred[:, 0].reshape(-1), y_true[:, 0].reshape(-1), min_recall=min_recall
        )
    )
    metrics.update(
        regression_metrics(y_pred[:, 1].reshape(-1), y_true[:, 1].reshape(-1))
    )
    # metrics.update(
    #     angle_reconstruction_metrics(y_pred[:, :2].reshape(-1), y_true[:, :2].reshape(-1))
    # )
    return metrics


def dummy_metrics(y_pred, y_true):
    return {}
