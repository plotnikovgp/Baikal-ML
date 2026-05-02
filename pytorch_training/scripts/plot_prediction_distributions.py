"""
Script to plot prediction distributions for signal/noise classification on different datasets.
Generates histograms showing the distribution of predicted signal probabilities.

Usage:
    python plot_prediction_distributions.py --checkpoint <path_to_checkpoint> --config <path_to_config>
"""

import argparse
import hashlib
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from tqdm import tqdm

from data_utils import *
from metrics import BinaryClassificationMetrics
from models import load_model


def load_checkpoint_and_config(
    checkpoint_path, config_path="train_configs/noise_sig_da_k02_gr0.yaml"
):
    config_path = "train_configs/noise_sig_da_k02_gr0.yaml"
    """Load model from checkpoint and config"""
    if config_path is None:
        # Try to load config from checkpoint directory
        checkpoint_dir = Path(checkpoint_path).parent
        possible_configs = list(checkpoint_dir.glob("train_config*.yaml"))
        if possible_configs:
            config_path = possible_configs[0]
        else:
            raise ValueError(
                f"Config path not provided and no train_config*.yaml found in {checkpoint_dir}"
            )

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Load model
    model = load_model(config["model_type"], config["model_params"])

    # Load checkpoint
    state_dict = torch.load(checkpoint_path, map_location="cuda")
    model.load_state_dict(state_dict)
    model = model.to("cuda")
    model.eval()

    return model, config


def get_cache_key(checkpoint_path, dataset_name):
    """Generate a cache key based on checkpoint and dataset"""
    checkpoint_mtime = Path(checkpoint_path).stat().st_mtime
    key_str = f"{checkpoint_path}_{dataset_name}_{checkpoint_mtime}"
    return hashlib.md5(key_str.encode()).hexdigest()


def load_cached_predictions(cache_dir, cache_key):
    """Load predictions from cache if available"""
    cache_file = Path(cache_dir) / f"{cache_key}.pkl"
    if cache_file.exists():
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    return None


def save_predictions_to_cache(cache_dir, cache_key, predictions, labels):
    """Save predictions to cache"""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{cache_key}.pkl"
    with open(cache_file, "wb") as f:
        pickle.dump({"predictions": predictions, "labels": labels}, f)


def get_predictions(
    model, val_loader, is_domain_adaptation=False, is_classification=True, has_labels=True
):
    """Get all predictions from validation set"""
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data in tqdm(val_loader, desc="Getting predictions"):
            if len(data) == 4:
                data = data[:3]
            x, y_true, mask = data

            x = x.to("cuda")
            y_true = y_true.to("cuda")
            mask = mask.to("cuda")

            if is_domain_adaptation:
                output, domain_pred = model(x, mask)
            else:
                output = model(x, mask)

            # For per-hit predictions, reshape
            if len(output.shape) == 3:  # [batch, seq, classes]
                output_flat = output.reshape(-1, output.shape[-1])
                mask_flat = mask.reshape(-1)

                # Apply masking
                valid_mask = mask_flat != 0
                output_valid = output_flat[valid_mask]

                if has_labels:
                    y_true_flat = y_true.reshape(-1)
                    y_true_valid = y_true_flat[valid_mask]
                else:
                    y_true_valid = None
            else:
                output_valid = output
                y_true_valid = y_true if has_labels else None

            if is_classification and len(output_valid.shape) == 2:
                probs = torch.sigmoid(output_valid[:, 1])
            else:
                probs = torch.sigmoid(output_valid)

            all_preds.append(probs.cpu().numpy())
            if has_labels:
                all_labels.append(y_true_valid.cpu().numpy())

    if has_labels:
        return np.concatenate(all_preds), np.concatenate(all_labels)
    else:
        return np.concatenate(all_preds), None


def plot_distributions(predictions_dict, labels_dict, save_path="prediction_distributions.png"):
    """
    Plot prediction distributions for each dataset

    Args:
        predictions_dict: Dict[dataset_name] -> predictions array
        labels_dict: Dict[dataset_name] -> labels array
    """
    fig, axes = plt.subplots(1, len(predictions_dict), figsize=(6 * len(predictions_dict), 5))

    if len(predictions_dict) == 1:
        axes = [axes]

    for ax, (dataset_name, preds) in zip(axes, predictions_dict.items()):
        labels = labels_dict[dataset_name]

        # Filter signal and noise
        signal_preds = preds[labels == 1]
        noise_preds = preds[labels == 0]

        # Plot histograms
        bins = np.linspace(0, 1, 50)

        ax.hist(
            signal_preds,
            bins=bins,
            alpha=0.5,
            label="Signal (true)",
            color="blue",
            density=True,
            histtype="step",
            linewidth=2,
        )
        ax.hist(
            noise_preds,
            bins=bins,
            alpha=0.5,
            label="Noise (true)",
            color="red",
            density=True,
            histtype="step",
            linewidth=2,
        )

        ax.set_xlabel("Predicted Signal Probability", fontsize=12)
        ax.set_ylabel("Normalized Density", fontsize=12)
        ax.set_title(
            f"{dataset_name}\n(Signal: {len(signal_preds)}, Noise: {len(noise_preds)})", fontsize=14
        )
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_xlim(0, 1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {save_path}")
    plt.close()


def compute_kl_divergence(p, q, bins=50):
    """Compute KL divergence between two distributions"""
    hist_p, _ = np.histogram(p, bins=bins, range=(0, 1), density=True)
    hist_q, _ = np.histogram(q, bins=bins, range=(0, 1), density=True)

    # Add small epsilon to avoid log(0)
    eps = 1e-10
    hist_p = hist_p + eps
    hist_q = hist_q + eps

    # Normalize
    hist_p = hist_p / hist_p.sum()
    hist_q = hist_q / hist_q.sum()

    return np.sum(hist_p * np.log(hist_p / hist_q))


def plot_all_predictions_comparison(
    predictions_dict,
    predictions_unlabeled_dict,
    model_name="model",
    save_path="all_predictions.png",
):
    """
    Simple comparison of ALL predicted probabilities (signal+noise combined) across datasets

    Args:
        predictions_dict: Dict[dataset_name] -> predictions array (labeled data)
        predictions_unlabeled_dict: Dict[dataset_name] -> predictions array (unlabeled data)
        model_name: Name of the model for plot titles
    """
    plt.figure(figsize=(12, 7))

    bins = np.linspace(0, 1, 50)

    for dataset_name, preds in predictions_dict.items():
        plt.hist(
            preds,
            bins=bins,
            alpha=0.6,
            label=f"{dataset_name} (MC)",
            density=True,
            histtype="stepfilled",
            linewidth=2.5,
            edgecolor="blue",
            color="lightblue",
        )

    for dataset_name, preds in predictions_unlabeled_dict.items():
        plt.hist(
            preds,
            bins=bins,
            alpha=0.6,
            label=f"{dataset_name} (Exp)",
            density=True,
            histtype="stepfilled",
            linewidth=2.5,
            edgecolor="orange",
            color="wheat",
        )

    plt.xlabel("Predicted Signal Probability", fontsize=14)
    plt.ylabel("Normalized Density", fontsize=14)
    plt.title(f"All Hits Prediction Distribution - {model_name}", fontsize=16, fontweight="bold")
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    plt.xlim(0, 1)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {save_path}")
    plt.close()


def plot_prediction_comparison(
    predictions_dict,
    predictions_unlabeled_dict,
    model_name="model",
    save_path="prediction_comparison.png",
):
    """
    Plot comprehensive comparison of predicted probability distributions across datasets

    Args:
        predictions_dict: Dict[dataset_name] -> predictions array (labeled data)
        predictions_unlabeled_dict: Dict[dataset_name] -> predictions array (unlabeled data)
        model_name: Name of the model for plot titles
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    bins = np.linspace(0, 1, 50)

    # Get data for both domains
    labeled_name = list(predictions_dict.keys())[0] if predictions_dict else None
    unlabeled_name = (
        list(predictions_unlabeled_dict.keys())[0] if predictions_unlabeled_dict else None
    )

    labeled_preds = predictions_dict[labeled_name] if labeled_name else np.array([])
    unlabeled_preds = predictions_unlabeled_dict[unlabeled_name] if unlabeled_name else np.array([])

    # 1. Histogram (PDF)
    ax1 = axes[0, 0]
    if len(labeled_preds) > 0:
        ax1.hist(
            labeled_preds,
            bins=bins,
            alpha=0.6,
            label=f"{labeled_name} (MC)",
            density=True,
            histtype="stepfilled",
            linewidth=2,
            edgecolor="blue",
            color="lightblue",
        )
    if len(unlabeled_preds) > 0:
        ax1.hist(
            unlabeled_preds,
            bins=bins,
            alpha=0.6,
            label=f"{unlabeled_name} (Exp)",
            density=True,
            histtype="stepfilled",
            linewidth=2,
            edgecolor="orange",
            color="wheat",
        )

    ax1.set_xlabel("Predicted Signal Probability", fontsize=12)
    ax1.set_ylabel("Normalized Density", fontsize=12)
    ax1.set_title(f"PDF - {model_name}", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)
    ax1.set_xlim(0, 1)

    # 2. Cumulative Distribution Function (CDF)
    ax2 = axes[0, 1]
    if len(labeled_preds) > 0:
        sorted_labeled = np.sort(labeled_preds)
        cdf_labeled = np.arange(1, len(sorted_labeled) + 1) / len(sorted_labeled)
        ax2.plot(
            sorted_labeled, cdf_labeled, label=f"{labeled_name} (MC)", linewidth=2.5, color="blue"
        )
    if len(unlabeled_preds) > 0:
        sorted_unlabeled = np.sort(unlabeled_preds)
        cdf_unlabeled = np.arange(1, len(sorted_unlabeled) + 1) / len(sorted_unlabeled)
        ax2.plot(
            sorted_unlabeled,
            cdf_unlabeled,
            label=f"{unlabeled_name} (Exp)",
            linewidth=2.5,
            color="orange",
            linestyle="--",
        )

    ax2.set_xlabel("Predicted Signal Probability", fontsize=12)
    ax2.set_ylabel("Cumulative Probability", fontsize=12)
    ax2.set_title(f"CDF - {model_name}", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=11)
    ax2.grid(alpha=0.3)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)

    # 3. Q-Q Plot
    ax3 = axes[1, 0]
    if len(labeled_preds) > 0 and len(unlabeled_preds) > 0:
        quantiles = np.linspace(0, 1, 100)
        q_labeled = np.quantile(labeled_preds, quantiles)
        q_unlabeled = np.quantile(unlabeled_preds, quantiles)

        ax3.scatter(q_labeled, q_unlabeled, alpha=0.5, s=20)
        ax3.plot([0, 1], [0, 1], "r--", linewidth=2, label="Perfect alignment")
        ax3.set_xlabel(f"{labeled_name} (MC) Quantiles", fontsize=12)
        ax3.set_ylabel(f"{unlabeled_name} (Exp) Quantiles", fontsize=12)
        ax3.set_title(f"Q-Q Plot - {model_name}", fontsize=14, fontweight="bold")
        ax3.legend(fontsize=11)
        ax3.grid(alpha=0.3)
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)
        ax3.set_aspect("equal")

    # 4. Statistics Summary
    ax4 = axes[1, 1]
    ax4.axis("off")

    stats_text = f"Distribution Statistics - {model_name}\n" + "=" * 50 + "\n\n"

    if len(labeled_preds) > 0:
        stats_text += f"{labeled_name} (MC - Labeled):\n"
        stats_text += f"  Mean:   {labeled_preds.mean():.4f}\n"
        stats_text += f"  Median: {np.median(labeled_preds):.4f}\n"
        stats_text += f"  Std:    {labeled_preds.std():.4f}\n"
        stats_text += f"  Q25:    {np.quantile(labeled_preds, 0.25):.4f}\n"
        stats_text += f"  Q75:    {np.quantile(labeled_preds, 0.75):.4f}\n"
        stats_text += f"  Samples: {len(labeled_preds):,}\n\n"

    if len(unlabeled_preds) > 0:
        stats_text += f"{unlabeled_name} (Exp - Unlabeled):\n"
        stats_text += f"  Mean:   {unlabeled_preds.mean():.4f}\n"
        stats_text += f"  Median: {np.median(unlabeled_preds):.4f}\n"
        stats_text += f"  Std:    {unlabeled_preds.std():.4f}\n"
        stats_text += f"  Q25:    {np.quantile(unlabeled_preds, 0.25):.4f}\n"
        stats_text += f"  Q75:    {np.quantile(unlabeled_preds, 0.75):.4f}\n"
        stats_text += f"  Samples: {len(unlabeled_preds):,}\n\n"

    if len(labeled_preds) > 0 and len(unlabeled_preds) > 0:
        kl_div = compute_kl_divergence(labeled_preds, unlabeled_preds)
        kl_div_rev = compute_kl_divergence(unlabeled_preds, labeled_preds)
        js_div = 0.5 * (kl_div + kl_div_rev)

        stats_text += "Divergence Metrics:\n"
        stats_text += f"  KL(MC || Exp):  {kl_div:.4f}\n"
        stats_text += f"  KL(Exp || MC):  {kl_div_rev:.4f}\n"
        stats_text += f"  JS Divergence:  {js_div:.4f}\n"
        stats_text += "\nLower divergence = more similar distributions\n"

    ax4.text(
        0.1,
        0.9,
        stats_text,
        transform=ax4.transAxes,
        fontsize=11,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {save_path}")
    plt.close()


def plot_signal_to_noise_ratio(
    predictions_dict, labels_dict, model_name="model", save_path="signal_noise_ratio.png"
):
    """
    Plot signal-to-noise ratio at different classification thresholds

    Shows how the ratio of signal to noise hits changes as threshold increases.
    Higher S/N ratio = better classification quality at that threshold.

    Args:
        predictions_dict: Dict[dataset_name] -> predictions array (labeled data)
        labels_dict: Dict[dataset_name] -> labels array (labeled data)
        model_name: Name of the model for plot titles
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    thresholds = np.linspace(0, 1, 100)

    # Left plot: Signal-to-Noise Ratio vs Threshold
    ax1 = axes[0]
    for dataset_name, preds in predictions_dict.items():
        labels = labels_dict[dataset_name]

        signal_ratios = []
        for t in thresholds:
            above_threshold = preds >= t
            n_signal = np.sum((labels == 1) & above_threshold)
            n_noise = np.sum((labels == 0) & above_threshold)

            # Compute S/N ratio (add small epsilon to avoid division by zero)
            ratio = n_signal / (n_noise + 1e-10)
            signal_ratios.append(ratio)

        ax1.plot(
            thresholds,
            signal_ratios,
            label=f"{dataset_name}",
            linewidth=2.5,
            marker="o",
            markersize=4,
            markevery=10,
        )

    ax1.set_xlabel("Classification Threshold", fontsize=14)
    ax1.set_ylabel("Signal-to-Noise Ratio", fontsize=14)
    ax1.set_title(f"Signal/Noise Ratio vs Threshold - {model_name}", fontsize=16, fontweight="bold")
    ax1.legend(fontsize=12)
    ax1.grid(alpha=0.3)
    ax1.set_yscale("log")  # Log scale for better visualization
    ax1.axhline(y=1, color="r", linestyle="--", alpha=0.5, label="S/N = 1")

    # Right plot: Signal Purity (fraction of hits above threshold that are signal)
    ax2 = axes[1]
    for dataset_name, preds in predictions_dict.items():
        labels = labels_dict[dataset_name]

        purities = []
        for t in thresholds:
            above_threshold = preds >= t
            n_signal = np.sum((labels == 1) & above_threshold)
            n_total = np.sum(above_threshold)

            # Compute purity (signal fraction)
            purity = n_signal / (n_total + 1e-10)
            purities.append(purity)

        ax2.plot(
            thresholds,
            purities,
            label=f"{dataset_name}",
            linewidth=2.5,
            marker="s",
            markersize=4,
            markevery=10,
        )

    ax2.set_xlabel("Classification Threshold", fontsize=14)
    ax2.set_ylabel("Signal Purity (Signal / Total)", fontsize=14)
    ax2.set_title(f"Signal Purity vs Threshold - {model_name}", fontsize=16, fontweight="bold")
    ax2.legend(fontsize=12)
    ax2.grid(alpha=0.3)
    ax2.set_ylim(0, 1.05)
    ax2.axhline(y=0.5, color="r", linestyle="--", alpha=0.5, label="50% purity")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {save_path}")
    plt.close()


def plot_precision_recall_vs_threshold(
    predictions_dict_no_da,
    labels_dict_no_da,
    predictions_dict_da,
    labels_dict_da,
    save_path="precision_recall_threshold.png",
):
    """
    Plot precision and recall vs threshold for both no-DA and DA models

    Args:
        predictions_dict_no_da: Dict[dataset_name] -> predictions array (no-DA model)
        labels_dict_no_da: Dict[dataset_name] -> labels array (no-DA model)
        predictions_dict_da: Dict[dataset_name] -> predictions array (DA model)
        labels_dict_da: Dict[dataset_name] -> labels array (DA model)
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    thresholds = np.linspace(0, 1, 100)

    # Left plot: Precision vs Threshold
    ax1 = axes[0]
    for (name, preds), labels in zip(predictions_dict_no_da.items(), labels_dict_no_da.values()):
        precisions = []
        for t in thresholds:
            above_threshold = preds >= t
            n_signal = np.sum((labels == 1) & above_threshold)
            n_total = np.sum(above_threshold)
            precision = n_signal / (n_total + 1e-10)
            precisions.append(precision)

        ax1.plot(
            thresholds,
            precisions,
            label="No DA",
            linewidth=2.5,
            marker="o",
            markersize=3,
            markevery=10,
            linestyle="-",
            color="blue",
        )

    for (name, preds), labels in zip(predictions_dict_da.items(), labels_dict_da.values()):
        precisions = []
        for t in thresholds:
            above_threshold = preds >= t
            n_signal = np.sum((labels == 1) & above_threshold)
            n_total = np.sum(above_threshold)
            precision = n_signal / (n_total + 1e-10)
            precisions.append(precision)

        ax1.plot(
            thresholds,
            precisions,
            label="With DA",
            linewidth=2.5,
            marker="s",
            markersize=3,
            markevery=10,
            linestyle="--",
            color="orange",
        )

    ax1.set_xlabel("Classification Threshold", fontsize=14)
    ax1.set_ylabel("Precision (Signal Purity)", fontsize=14)
    ax1.set_title("Precision vs Threshold", fontsize=16, fontweight="bold")
    ax1.legend(fontsize=12)
    ax1.grid(alpha=0.3)
    ax1.set_ylim(0, 1.05)

    # Right plot: Recall vs Threshold
    ax2 = axes[1]
    for (name, preds), labels in zip(predictions_dict_no_da.items(), labels_dict_no_da.values()):
        recalls = []
        total_signal = np.sum(labels == 1)
        for t in thresholds:
            above_threshold = preds >= t
            n_signal_found = np.sum((labels == 1) & above_threshold)
            recall = n_signal_found / (total_signal + 1e-10)
            recalls.append(recall)

        ax2.plot(
            thresholds,
            recalls,
            label="No DA",
            linewidth=2.5,
            marker="o",
            markersize=3,
            markevery=10,
            linestyle="-",
            color="blue",
        )

    for (name, preds), labels in zip(predictions_dict_da.items(), labels_dict_da.values()):
        recalls = []
        total_signal = np.sum(labels == 1)
        for t in thresholds:
            above_threshold = preds >= t
            n_signal_found = np.sum((labels == 1) & above_threshold)
            recall = n_signal_found / (total_signal + 1e-10)
            recalls.append(recall)

        ax2.plot(
            thresholds,
            recalls,
            label="With DA",
            linewidth=2.5,
            marker="s",
            markersize=3,
            markevery=10,
            linestyle="--",
            color="orange",
        )

    ax2.set_xlabel("Classification Threshold", fontsize=14)
    ax2.set_ylabel("Recall (Signal Efficiency)", fontsize=14)
    ax2.set_title("Recall vs Threshold", fontsize=16, fontweight="bold")
    ax2.legend(fontsize=12)
    ax2.grid(alpha=0.3)
    ax2.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {save_path}")
    plt.close()


def plot_signal_vs_threshold(
    predictions_dict, labels_dict, predictions_unlabeled_dict, save_path="signal_vs_threshold.png"
):
    """
    Plot comparison of predictions across datasets for domain adaptation analysis

    Args:
        predictions_dict: Dict[dataset_name] -> predictions array (labeled data)
        labels_dict: Dict[dataset_name] -> labels array (labeled data)
        predictions_unlabeled_dict: Dict[dataset_name] -> predictions array (unlabeled data)
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    thresholds = np.linspace(0, 1, 100)

    # Left plot: Signal detection rate on labeled data
    ax1 = axes[0]
    for dataset_name, preds in predictions_dict.items():
        labels = labels_dict[dataset_name]
        signal_preds = preds[labels == 1]

        signal_counts = [np.sum(signal_preds >= t) for t in thresholds]

        ax1.plot(
            thresholds,
            signal_counts,
            label=f"{dataset_name} (labeled)",
            linewidth=2.5,
            marker="o",
            markersize=4,
            markevery=10,
        )

    ax1.set_xlabel("Classification Threshold", fontsize=14)
    ax1.set_ylabel("Number of Signal Hits Above Threshold", fontsize=14)
    ax1.set_title("Signal Detection on Labeled Data", fontsize=16)
    ax1.legend(fontsize=12)
    ax1.grid(alpha=0.3)

    # Right plot: Normalized prediction distribution on ALL datasets (labeled + unlabeled)
    ax2 = axes[1]

    for dataset_name, preds in predictions_dict.items():
        hit_counts = np.array([np.sum(preds >= t) for t in thresholds])
        normalized_counts = hit_counts / len(preds)  # Normalize by total hits
        ax2.plot(
            thresholds,
            normalized_counts,
            label=f"{dataset_name} (labeled)",
            linewidth=2.5,
            marker="o",
            markersize=4,
            markevery=10,
            linestyle="-",
        )

    for dataset_name, preds in predictions_unlabeled_dict.items():
        hit_counts = np.array([np.sum(preds >= t) for t in thresholds])
        normalized_counts = hit_counts / len(preds)  # Normalize by total hits
        ax2.plot(
            thresholds,
            normalized_counts,
            label=f"{dataset_name} (unlabeled)",
            linewidth=2.5,
            marker="s",
            markersize=4,
            markevery=10,
            linestyle="--",
        )

    ax2.set_xlabel("Classification Threshold", fontsize=14)
    ax2.set_ylabel("Fraction of Hits Classified as Signal", fontsize=14)
    ax2.set_title("Model Predictions Across Domains (Normalized)", fontsize=16)
    ax2.legend(fontsize=12)
    ax2.grid(alpha=0.3)
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {save_path}")
    plt.close()


def load_model_predictions(checkpoint_path, config, cache_dir, no_cache, batch_size, model_name):
    """Helper function to load a model and get predictions"""
    import copy

    from data_utils import BaikalDataset, BaikalDatasetNoLabels
    from data_utils.dataloaders import create_multi_dataset_dataloader
    from train import create_preprocessor

    print(f"\n{'=' * 60}")
    print(f"Loading {model_name} model from {checkpoint_path}")
    print(f"{'=' * 60}")

    # If config is a dict, we use it; if it's a string (path), we load it; if None, auto-detect
    if isinstance(config, str):
        model, loaded_config = load_checkpoint_and_config(checkpoint_path, config)
        config = loaded_config
    elif isinstance(config, dict):
        model, loaded_config = load_checkpoint_and_config(checkpoint_path, None)
        # Use provided config, not the one from checkpoint dir
    else:
        model, loaded_config = load_checkpoint_and_config(checkpoint_path, None)
        config = loaded_config

    # Use provided config or loaded config
    if not isinstance(config, dict):
        config = loaded_config
    else:
        config = copy.deepcopy(config)

    # Determine preprocessor based on train_type
    train_type = config["train_type"]
    is_domain_adaptation = "domain_adaptation" in train_type
    is_graph = config.get("is_graph", False)

    # Preprocess dataset configs to convert string DatasetTypes to actual classes
    for i, dc in enumerate(config["dataset_configs"]):
        if "DatasetType" not in dc:
            dc["DatasetType"] = BaikalDataset
        elif dc["DatasetType"] == "no_labels":
            dc["DatasetType"] = BaikalDatasetNoLabels
        else:
            raise ValueError(f"Unknown dataset type: {dc['DatasetType']}")

        dc["preprocessor"] = create_preprocessor(train_type, dc.get("is_graph", is_graph), dc)

    dataloader_dict = create_multi_dataset_dataloader(
        dataset_configs=config["dataset_configs"],
        batch_size=batch_size,
        return_datasets=False,
    )

    # Get predictions for each dataset
    predictions_dict = {}
    labels_dict = {}
    predictions_unlabeled_dict = {}

    dataset_names = [dc["name"] for dc in config["dataset_configs"]]

    # Use train loader for exp_data (unlabeled), val loader for sig_noise_2020 (labeled)
    val_loaders = []
    for train_loader, val_loader, name in zip(
        dataloader_dict["train"], dataloader_dict["val"], dataset_names
    ):
        if "exp" in name.lower():
            val_loaders.append(train_loader)
            print(f"Using TRAIN loader for {name}")
        else:
            val_loaders.append(val_loader)
            print(f"Using VAL loader for {name}")

    for i, (loader, name) in enumerate(zip(val_loaders, dataset_names)):
        dataset_type = config["dataset_configs"][i].get("DatasetType", None)
        is_unlabeled = dataset_type == BaikalDatasetNoLabels or dataset_type == "no_labels"

        print(f"\nProcessing {name}...")

        # Try to load from cache
        cache_key = get_cache_key(checkpoint_path, name)
        cached = None if no_cache else load_cached_predictions(cache_dir, cache_key)

        if cached is not None:
            print("  Loaded from cache")
            preds = cached["predictions"]
            labels = cached.get("labels", None)
        else:
            preds, labels = get_predictions(
                model,
                loader,
                is_domain_adaptation=is_domain_adaptation,
                is_classification=True,
                has_labels=not is_unlabeled,
            )

            if not no_cache:
                save_predictions_to_cache(cache_dir, cache_key, preds, labels)
                print("  Saved to cache")

        if is_unlabeled:
            predictions_unlabeled_dict[name] = preds
            print(f"  Total hits: {len(preds)}")
            print(f"  Mean prediction: {preds.mean():.4f}")
            print(f"  Median prediction: {np.median(preds):.4f}")
        else:
            predictions_dict[name] = preds
            labels_dict[name] = labels
            metrics = BinaryClassificationMetrics(min_recall=0.9)(preds, labels)
            print(f"  Metrics: {metrics}")
            signal_count = (labels == 1).sum()
            noise_count = (labels == 0).sum()
            print(f"  Signal samples: {signal_count}")
            print(f"  Noise samples: {noise_count}")
            print(f"  Mean prediction: {preds.mean():.4f}")
            print(f"  Mean for signal: {preds[labels == 1].mean():.4f}")
            print(f"  Mean for noise: {preds[labels == 0].mean():.4f}")

    return predictions_dict, labels_dict, predictions_unlabeled_dict


def main():
    parser = argparse.ArgumentParser(description="Plot prediction distributions")
    parser.add_argument(
        "--checkpoint_no_da", type=str, required=True, help="Path to no-DA model checkpoint"
    )
    parser.add_argument(
        "--checkpoint_da", type=str, required=True, help="Path to DA model checkpoint"
    )
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument(
        "--output", type=str, default="prediction_distributions.png", help="Output plot filename"
    )
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for inference")
    parser.add_argument(
        "--cache_dir", type=str, default="prediction_cache", help="Directory to cache predictions"
    )
    parser.add_argument("--no_cache", action="store_true", help="Disable caching")

    args = parser.parse_args()

    # Load config once
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Create dataloaders
    from data_utils import BaikalDataset, BaikalDatasetNoLabels
    from data_utils.dataloaders import create_multi_dataset_dataloader
    from train import create_preprocessor

    # Determine preprocessor based on train_type
    train_type = config["train_type"]
    is_domain_adaptation = "domain_adaptation" in train_type
    is_graph = config.get("is_graph", False)

    # Preprocess dataset configs to convert string DatasetTypes to actual classes
    for i, dc in enumerate(config["dataset_configs"]):
        if "DatasetType" not in dc:
            dc["DatasetType"] = BaikalDataset
        elif dc["DatasetType"] == "no_labels":
            dc["DatasetType"] = BaikalDatasetNoLabels
        else:
            raise ValueError(f"Unknown dataset type: {dc['DatasetType']}")

        dc["preprocessor"] = create_preprocessor(train_type, dc.get("is_graph", is_graph), dc)

    dataloader_dict = create_multi_dataset_dataloader(
        dataset_configs=config["dataset_configs"],
        batch_size=args.batch_size,
        return_datasets=False,
    )

    # Get predictions for each dataset
    predictions_dict = {}
    labels_dict = {}
    predictions_unlabeled_dict = {}

    dataset_names = [dc["name"] for dc in config["dataset_configs"]]
    val_loaders = dataloader_dict["val"]

    for i, (loader, name) in enumerate(zip(val_loaders, dataset_names)):
        from data_utils import BaikalDatasetNoLabels

        dataset_type = config["dataset_configs"][i].get("DatasetType", None)
        is_unlabeled = dataset_type == BaikalDatasetNoLabels or dataset_type == "no_labels"

        print(f"\nProcessing {name}...")

        # Try to load from cache
        cache_key = get_cache_key(args.checkpoint, name)
        cached = None if args.no_cache else load_cached_predictions(args.cache_dir, cache_key)

        if cached is not None:
            print("  Loaded from cache")
            preds = cached["predictions"]
            labels = cached.get("labels", None)
        else:
            preds, labels = get_predictions(
                model,
                loader,
                is_domain_adaptation=is_domain_adaptation,
                is_classification=True,
                has_labels=not is_unlabeled,
            )

            if not args.no_cache:
                save_predictions_to_cache(args.cache_dir, cache_key, preds, labels)
                print("  Saved to cache")

        if is_unlabeled:
            predictions_unlabeled_dict[name] = preds
            print(f"  Total hits: {len(preds)}")
            print(f"  Mean prediction: {preds.mean():.4f}")
            print(f"  Median prediction: {np.median(preds):.4f}")
        else:
            predictions_dict[name] = preds
            labels_dict[name] = labels
            metrics = BinaryClassificationMetrics(min_recall=0.9)(preds, labels)
            print(f"  Metrics: {metrics}")
            # Print statistics
            signal_count = (labels == 1).sum()
            noise_count = (labels == 0).sum()
            print(f"  Signal samples: {signal_count}")
            print(f"  Noise samples: {noise_count}")
            print(f"  Mean prediction: {preds.mean():.4f}")
            print(f"  Mean for signal: {preds[labels == 1].mean():.4f}")
            print(f"  Mean for noise: {preds[labels == 0].mean():.4f}")

    # Determine model name
    if args.model_name:
        model_name = args.model_name
    else:
        # Auto-detect from checkpoint path
        checkpoint_dir = Path(args.checkpoint).parent.name
        if "no_da" in checkpoint_dir or "baseline" in checkpoint_dir:
            model_name = "No DA (Baseline)"
        elif "da" in checkpoint_dir:
            model_name = "With DA"
        else:
            model_name = checkpoint_dir

    print(f"Model name: {model_name}")

    # Plot all visualizations
    print("\nGenerating plots...")

    # 1. Distribution plot for labeled data only (signal vs noise)
    if predictions_dict:
        plot_distributions(predictions_dict, labels_dict, save_path=args.output)

    # 2. Simple comparison of ALL predictions (signal+noise combined)
    all_preds_plot_path = args.output.replace(".png", "_all_predictions.png")
    plot_all_predictions_comparison(
        predictions_dict,
        predictions_unlabeled_dict,
        model_name=model_name,
        save_path=all_preds_plot_path,
    )

    # 3. Comprehensive comparison with statistics
    comparison_plot_path = args.output.replace(".png", "_comparison.png")
    plot_prediction_comparison(
        predictions_dict,
        predictions_unlabeled_dict,
        model_name=model_name,
        save_path=comparison_plot_path,
    )

    # 4. Threshold analysis plot
    threshold_plot_path = args.output.replace(".png", "_threshold.png")
    plot_signal_vs_threshold(
        predictions_dict, labels_dict, predictions_unlabeled_dict, save_path=threshold_plot_path
    )

    # 5. Signal-to-noise ratio plot
    snr_plot_path = args.output.replace(".png", "_snr.png")
    plot_signal_to_noise_ratio(
        predictions_dict, labels_dict, model_name=model_name, save_path=snr_plot_path
    )

    # 6. Precision/Recall vs threshold (requires two models - skip for now, will be added with comparison script)
    # pr_plot_path = args.output.replace('.png', '_precision_recall.png')
    # plot_precision_recall_vs_threshold(predictions_dict, labels_dict, predictions_dict, labels_dict, save_path=pr_plot_path)

    print("Done!")


if __name__ == "__main__":
    main()
