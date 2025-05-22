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

    def set_dataset_name(self, dataset_name: str):
        self.dataset_name = dataset_name

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
            a = input("save? (y/n)")
            print(self.save_dir / "data" / f"{self.dataset_name}_{name}.txt")
            np.savetxt(self.save_dir / "data" / f"{self.dataset_name}_{name}.txt", data)
        self.data_to_save = {}

    def __call__(self, y_pred, y_true, **kwargs):
        metrics = self._calc_metrics(y_pred, y_true, **kwargs)

        if self.save_preds:
            self._save_preds(**kwargs)
        if self.plot_metrics:
            self._plot(**kwargs)

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


class AngleReconstructionMetrics(BaseMetrics):
    def _calc_metrics(
        self, y_pred, y_true, additional_data: dict | None = None, **kwargs
    ):
        y_pred = np.array(y_pred, dtype=np.float32)
        y_true = np.array(y_true, dtype=np.float32)
        metrics = {}

        # TODO: check by flag
        if y_pred.shape[1] == 6:
            log_sigma2_pred = np.array(y_pred[:, 3:], dtype=np.float32)
            sigma2_pred = np.exp(log_sigma2_pred)
            sigma_pred = np.sqrt(sigma2_pred)
            y_pred = y_pred[:, :3]
            sigma_true = (y_pred - y_true) ** 2

            sigma_metrics = regression_metrics(sigma_pred.mean(1), sigma_true.mean(1))
            metrics.update({"sigma2_" + k: v for k, v in sigma_metrics.items()})

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


def angle_uncertainty_metrics(y_pred_and_log_sigma, y_true):
    # Split predictions and log variances
    y_pred, log_sigma = y_pred_and_log_sigma[:, :3], y_pred_and_log_sigma[:, 3:]
    y_pred = np.array(y_pred, dtype=np.float32)
    y_true = np.array(y_true, dtype=np.float32)
    predicted_sigma = np.exp(
        np.array(log_sigma, dtype=np.float32) / 2
    )  # Convert to standard deviation

    # Extract angles from true and predicted vectors
    angles_true = np.array([extract_angles(vec) for vec in y_true], dtype=np.float32)
    angles_pred = np.array([extract_angles(vec) for vec in y_pred], dtype=np.float32)

    # Separate theta and phi angles
    y_true_theta_angle, y_true_phi_angle = angles_true[:, 0], angles_true[:, 1]
    y_pred_theta_angle, y_pred_phi_angle = angles_pred[:, 0], angles_pred[:, 1]

    # Compute directional error using dot product (same as in your function)
    # Normalize vectors to get pure directional error
    y_true_norm = np.linalg.norm(y_true, axis=1, keepdims=True)
    y_pred_norm = np.linalg.norm(y_pred, axis=1, keepdims=True)

    # Handle zero vectors
    valid_indices = (y_true_norm.flatten() > 1e-6) & (y_pred_norm.flatten() > 1e-6)

    # Initialize with maximum angle error (180 degrees)
    dir_resolution = np.full(len(y_true), 180.0)

    # Calculate directional error only for valid vectors
    if np.any(valid_indices):
        y_true_normalized = np.zeros_like(y_true)
        y_pred_normalized = np.zeros_like(y_pred)

        y_true_normalized[valid_indices] = (
            y_true[valid_indices] / y_true_norm[valid_indices]
        )
        y_pred_normalized[valid_indices] = (
            y_pred[valid_indices] / y_pred_norm[valid_indices]
        )

        dot_product = np.sum(y_true_normalized * y_pred_normalized, axis=1)
        dot_product = np.clip(dot_product, -1.0, 1.0)
        dir_resolution[valid_indices] = np.abs(
            np.rad2deg(np.arccos(dot_product[valid_indices]))
        )

    # Calculate angular differences
    theta_resolution = np.abs(y_true_theta_angle - y_pred_theta_angle)

    # For phi, handle the circular nature (wrapping around 360°)
    phi_resolution = np.minimum(
        np.abs(y_true_phi_angle - y_pred_phi_angle),
        360.0 - np.abs(y_true_phi_angle - y_pred_phi_angle),
    )

    # Convert Cartesian uncertainties to angular uncertainties
    # This is where we need to compute sigma for theta and phi based on predicted_sigma

    # Initialize arrays for angular uncertainties
    sigma_theta = np.zeros(len(y_pred))
    sigma_phi = np.zeros(len(y_pred))

    for i in range(len(y_pred)):
        # Skip if predicted vector is too small for meaningful angle calculation
        if np.linalg.norm(y_pred[i]) < 1e-6:
            sigma_theta[i] = 90.0  # Default large uncertainty
            sigma_phi[i] = 180.0  # Default large uncertainty
            continue

        # Normalize the prediction vector for angular calculations
        x, y, z = y_pred[i] / np.linalg.norm(y_pred[i])

        # Get predicted standard deviations for each component
        sigma_x, sigma_y, sigma_z = predicted_sigma[i]

        # Convert Cartesian uncertainties to spherical uncertainties
        # Using error propagation formulas

        # For theta uncertainty (inclination angle)
        # θ = arccos(z)
        # Partial derivatives: dθ/dz = -1/sqrt(1-z²)
        if abs(z) < 0.99:  # Avoid numerical issues near poles
            sigma_theta[i] = np.rad2deg(np.abs(sigma_z / np.sqrt(1 - z**2)))
        else:
            sigma_theta[i] = 90.0  # Large uncertainty near poles

        # For phi uncertainty (azimuth angle)
        # φ = arctan2(y, x)
        # Error propagation for arctan2: σ_φ² = (σ_y/(x²+y²))² + (σ_x·y/(x²+y²))²
        xy_norm_squared = x**2 + y**2
        if xy_norm_squared > 1e-6:  # Avoid division by near-zero
            sigma_phi[i] = np.rad2deg(
                np.sqrt(
                    (sigma_y / xy_norm_squared) ** 2
                    + (sigma_x * y / xy_norm_squared) ** 2
                )
            )
        else:
            sigma_phi[i] = 180.0  # Large uncertainty when on z-axis

    # Calculate sigma differences - how well the predicted uncertainty matches the actual error
    sigma_diff_theta_angle = np.abs(sigma_theta - theta_resolution)
    sigma_diff_phi_angle = np.abs(sigma_phi - phi_resolution)

    # Calculate sigma magnitude difference
    error_magnitude = np.linalg.norm(y_true - y_pred, axis=1)
    predicted_sigma_magnitude = np.linalg.norm(predicted_sigma, axis=1)
    sigma_diff = predicted_sigma_magnitude - error_magnitude

    # Calculate metrics
    dir_mae = np.mean(dir_resolution)
    dir_q50 = np.quantile(dir_resolution, 0.5)
    theta_q50 = np.quantile(sigma_diff_theta_angle, 0.5)
    phi_q50 = np.quantile(sigma_diff_phi_angle, 0.5)

    metrics = {
        "err_mae": (predicted_sigma - (y_true - y_pred) ** 2).mean(),
        "dir_mae": dir_mae,
        # "sigma_diff_mae": np.mean(np.abs(sigma_diff)),
        # "thetha_q50": theta_q50,
        # "phi_q50": phi_q50,
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
