"""
Script to compare two models (no-DA vs DA) on the same plots.
Generates comparison plots with both models' predictions side-by-side.

Usage:
    python plot_comparison.py --checkpoint_no_da <path> --checkpoint_da <path> --config <path> --output <path>
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import umap
import yaml
from scipy.stats import entropy
from tqdm import tqdm

from data_utils import *
from data_utils.dataloaders import create_multi_dataset_dataloader
from metrics import BinaryClassificationMetrics
from models import load_model
from train import create_preprocessor


def compute_kl_divergence(p, q, bins=100, range_val=(0, 0.3)):
    """Compute KL divergence between two distributions"""
    hist_p, _ = np.histogram(p, bins=bins, range=range_val, density=True)
    hist_q, _ = np.histogram(q, bins=bins, range=range_val, density=True)

    eps = 1e-10
    hist_p = hist_p + eps
    hist_q = hist_q + eps

    hist_p = hist_p / hist_p.sum()
    hist_q = hist_q / hist_q.sum()

    return np.sum(hist_p * np.log(hist_p / hist_q))


def load_checkpoint_and_config(checkpoint_path, config_path):
    """Load model from checkpoint and config"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model = load_model(config["model_type"], config["model_params"])
    state_dict = torch.load(checkpoint_path, map_location="cuda")
    model.load_state_dict(state_dict)
    model = model.to("cuda")
    model.eval()

    return model, config


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

            # Get probabilities for positive class (signal)
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


def get_hidden_states(model, val_loader, max_samples=10000):
    """Extract hidden states from the encoder for UMAP visualization"""
    all_hidden_states = []
    total_samples = 0

    with torch.no_grad():
        for data in tqdm(val_loader, desc="Extracting hidden states"):
            if total_samples >= max_samples:
                break

            if len(data) == 4:
                data = data[:3]
            x, y_true, mask = data

            x = x.to("cuda")
            mask = mask.to("cuda")

            # Get hidden states (use return_hidden_states if available)
            try:
                output, domain_pred, hidden_states = model(x, mask, return_hidden_states=True)
            except (TypeError, AttributeError):
                _, hidden_states = model.encoder(x, mask)

            # Average pool over sequence dimension to get per-event features
            # hidden_states shape: [batch, seq_len, hidden_dim]
            mask_expanded = mask.unsqueeze(-1)  # [batch, seq_len, 1]
            masked_hidden = hidden_states * mask_expanded
            seq_lengths = mask.sum(dim=1, keepdim=True).clamp(min=1)  # [batch, 1]
            pooled_hidden = masked_hidden.sum(dim=1) / seq_lengths  # [batch, hidden_dim]

            all_hidden_states.append(pooled_hidden.cpu().numpy())

            total_samples += len(x)

    hidden_states = np.concatenate(all_hidden_states, axis=0)[:max_samples]

    # Return hidden states and dummy labels (not needed for UMAP domain visualization)
    return hidden_states, None


def load_model_predictions(checkpoint_path, config, model_name):
    """Helper function to load a model and get predictions"""
    print(f"\n{'=' * 60}")
    print(f"Loading {model_name} from {Path(checkpoint_path).name}")
    print(f"{'=' * 60}")
    model, _ = load_checkpoint_and_config(checkpoint_path, config)

    # Load config
    with open(config, "r") as f:
        cfg = yaml.safe_load(f)

    train_type = cfg["train_type"]
    is_domain_adaptation = "domain_adaptation" in train_type
    is_graph = cfg.get("is_graph", False)

    # Preprocess dataset configs
    for i, dc in enumerate(cfg["dataset_configs"]):
        if "DatasetType" not in dc:
            dc["DatasetType"] = BaikalDataset
        elif dc["DatasetType"] == "no_labels":
            dc["DatasetType"] = BaikalDatasetNoLabels

        dc["preprocessor"] = create_preprocessor(train_type, dc.get("is_graph", is_graph), dc)

    dataloader_dict = create_multi_dataset_dataloader(
        dataset_configs=cfg["dataset_configs"],
        batch_size=128,
        return_datasets=False,
    )

    predictions_dict = {}
    labels_dict = {}
    predictions_unlabeled_dict = {}

    dataset_names = [dc["name"] for dc in cfg["dataset_configs"]]
    val_loaders = dataloader_dict["val"]

    for i, (loader, name) in enumerate(zip(val_loaders, dataset_names)):
        dataset_type = cfg["dataset_configs"][i].get("DatasetType", None)
        is_unlabeled = dataset_type == BaikalDatasetNoLabels or dataset_type == "no_labels"

        print(f"Processing {name}...")
        preds, labels = get_predictions(
            model,
            loader,
            is_domain_adaptation=is_domain_adaptation,
            is_classification=True,
            has_labels=not is_unlabeled,
        )

        if is_unlabeled:
            predictions_unlabeled_dict[name] = preds
        else:
            predictions_dict[name] = preds
            labels_dict[name] = labels
            metrics = BinaryClassificationMetrics(min_recall=0.9)(preds, labels)
            print(
                f"  AUC: {metrics['auc']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}"
            )

    # Also extract hidden states for UMAP
    hidden_states_dict = {}
    hidden_labels_dict = {}

    print("\nExtracting hidden states for UMAP...")
    for i, (loader, name) in enumerate(zip(val_loaders, dataset_names)):
        print(f"  {name}...")
        hidden, labels_h = get_hidden_states(model, loader, max_samples=5000)
        hidden_states_dict[name] = hidden
        hidden_labels_dict[name] = labels_h

    return (
        predictions_dict,
        labels_dict,
        predictions_unlabeled_dict,
        hidden_states_dict,
        hidden_labels_dict,
    )


def plot_signal_distributions_comparison(
    preds_no_da,
    labels_no_da,
    preds_da,
    labels_da,
    preds_unlabeled_no_da,
    preds_unlabeled_da,
    save_path,
):
    """
    Plot 1: Compare exp_data vs sig_noise_2020 distributions for No-DA and DA models
    Shows how domain adaptation affects the alignment between MC and Exp data
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))

    bins = np.linspace(0, 1.0, 60)

    # Get all predictions (not just signal)
    mc_name = list(preds_no_da.keys())[0] if preds_no_da else None
    exp_name = list(preds_unlabeled_no_da.keys())[0] if preds_unlabeled_no_da else None

    if mc_name and exp_name:
        # No-DA model predictions
        mc_preds_no_da = preds_no_da[mc_name]
        exp_preds_no_da = preds_unlabeled_no_da[exp_name]

        # DA model predictions
        mc_preds_da = preds_da[mc_name]
        exp_preds_da = preds_unlabeled_da[exp_name]

        # Plot using step histograms (cleaner for log scale)
        ax.hist(
            mc_preds_no_da,
            bins=bins,
            label=f"{mc_name} - No DA (n={len(mc_preds_no_da):,})",
            density=True,
            histtype="step",
            linewidth=2.5,
            color="blue",
            linestyle="-",
            alpha=0.8,
        )
        ax.hist(
            exp_preds_no_da,
            bins=bins,
            label=f"{exp_name} - No DA (n={len(exp_preds_no_da):,})",
            density=True,
            histtype="step",
            linewidth=2.5,
            color="blue",
            linestyle="--",
            alpha=0.8,
        )

        ax.hist(
            mc_preds_da,
            bins=bins,
            label=f"{mc_name} - With DA (n={len(mc_preds_da):,})",
            density=True,
            histtype="step",
            linewidth=2.5,
            color="orange",
            linestyle="-",
            alpha=0.8,
        )
        ax.hist(
            exp_preds_da,
            bins=bins,
            label=f"{exp_name} - With DA (n={len(exp_preds_da):,})",
            density=True,
            histtype="step",
            linewidth=2.5,
            color="orange",
            linestyle="--",
            alpha=0.8,
        )

    ax.set_xlabel("Predicted Signal Probability", fontsize=14)
    ax.set_ylabel("Normalized Density (log scale)", fontsize=14)
    ax.set_title("Distribution Comparison: MC vs Exp (No-DA vs DA)", fontsize=16, fontweight="bold")
    ax.legend(fontsize=11, loc="best")
    ax.grid(alpha=0.3, which="both", linestyle=":")
    ax.set_xlim(0, 1.0)
    ax.set_yscale("log")
    ax.set_ylim(bottom=1e-3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()


def plot_umap_comparison(hidden_no_da, hidden_da, save_path):
    """
    Plot UMAP visualization of hidden states
    Left: No-DA, Right: With DA
    Colors: Blue=MC, Orange=Exp
    Shows how domain adaptation affects feature space alignment
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Get dataset names
    mc_name = [k for k in hidden_no_da.keys() if "sig_noise" in k or "mc" in k.lower()][0]
    exp_name = [k for k in hidden_no_da.keys() if "exp" in k.lower()][0]

    # Combine MC and Exp data for each model
    hidden_combined_no_da = np.concatenate([hidden_no_da[mc_name], hidden_no_da[exp_name]], axis=0)
    labels_no_da = np.array(
        ["MC"] * len(hidden_no_da[mc_name]) + ["Exp"] * len(hidden_no_da[exp_name])
    )

    hidden_combined_da = np.concatenate([hidden_da[mc_name], hidden_da[exp_name]], axis=0)
    labels_da = np.array(["MC"] * len(hidden_da[mc_name]) + ["Exp"] * len(hidden_da[exp_name]))

    # Left plot: No-DA
    print("Computing UMAP for No-DA model...")
    reducer_no_da = umap.UMAP(n_neighbors=30, min_dist=0.1, n_components=2, random_state=42)
    embedding_no_da = reducer_no_da.fit_transform(hidden_combined_no_da)

    ax = axes[0]
    mc_mask = labels_no_da == "MC"
    exp_mask = labels_no_da == "Exp"

    ax.scatter(
        embedding_no_da[mc_mask, 0],
        embedding_no_da[mc_mask, 1],
        c="blue",
        label=f"{mc_name} (MC)",
        alpha=0.5,
        s=10,
        edgecolors="none",
    )
    ax.scatter(
        embedding_no_da[exp_mask, 0],
        embedding_no_da[exp_mask, 1],
        c="orange",
        label=f"{exp_name} (Exp)",
        alpha=0.5,
        s=10,
        edgecolors="none",
    )

    ax.set_xlabel("UMAP 1", fontsize=14)
    ax.set_ylabel("UMAP 2", fontsize=14)
    ax.set_title("No Domain Adaptation", fontsize=16, fontweight="bold")
    ax.legend(fontsize=12, markerscale=3)
    ax.grid(alpha=0.3)

    # Right plot: DA
    print("Computing UMAP for DA model...")
    reducer_da = umap.UMAP(n_neighbors=30, min_dist=0.1, n_components=2, random_state=42)
    embedding_da = reducer_da.fit_transform(hidden_combined_da)

    ax = axes[1]
    mc_mask = labels_da == "MC"
    exp_mask = labels_da == "Exp"

    ax.scatter(
        embedding_da[mc_mask, 0],
        embedding_da[mc_mask, 1],
        c="blue",
        label=f"{mc_name} (MC)",
        alpha=0.3,
        s=10,
        edgecolors="none",
    )
    ax.scatter(
        embedding_da[exp_mask, 0],
        embedding_da[exp_mask, 1],
        c="orange",
        label=f"{exp_name} (Exp)",
        alpha=0.3,
        s=10,
        edgecolors="none",
    )

    ax.set_xlabel("UMAP 1", fontsize=14)
    ax.set_ylabel("UMAP 2", fontsize=14)
    ax.set_title("With Domain Adaptation", fontsize=16, fontweight="bold")
    ax.legend(fontsize=12, markerscale=3)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()


def plot_signal_purity_comparison(
    preds_no_da,
    labels_no_da,
    preds_da,
    labels_da,
    preds_unlabeled_no_da,
    preds_unlabeled_da,
    save_path,
):
    """
    Plot 2: Signal purity vs threshold
    Left: no-DA, Right: with DA
    Each plot has two lines: sig_noise_2020 (MC) and exp_data (Exp)
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    thresholds = np.linspace(0, 1, 100)

    # Left plot: No-DA model
    ax1 = axes[0]
    for name, preds in preds_no_da.items():
        labels = labels_no_da[name]
        purities = []
        for t in thresholds:
            above_threshold = preds >= t
            n_signal = np.sum((labels == 1) & above_threshold)
            n_total = np.sum(above_threshold)
            purity = n_signal / (n_total + 1e-10)
            purities.append(purity)
        ax1.plot(
            thresholds,
            purities,
            label=f"{name} (MC)",
            linewidth=2.5,
            marker="o",
            markersize=2,
            markevery=10,
        )

    # For unlabeled data, we can't compute true purity, so skip

    ax1.set_xlabel("Classification Threshold", fontsize=14)
    ax1.set_ylabel("Signal Purity (Signal / Total)", fontsize=14)
    ax1.set_title("No Domain Adaptation", fontsize=16, fontweight="bold")
    ax1.legend(fontsize=12)
    ax1.grid(alpha=0.3)
    ax1.set_ylim(0, 1.05)
    ax1.axhline(y=0.5, color="r", linestyle="--", alpha=0.3)

    # Right plot: DA model
    ax2 = axes[1]
    for name, preds in preds_da.items():
        labels = labels_da[name]
        purities = []
        for t in thresholds:
            above_threshold = preds >= t
            n_signal = np.sum((labels == 1) & above_threshold)
            n_total = np.sum(above_threshold)
            purity = n_signal / (n_total + 1e-10)
            purities.append(purity)
        ax2.plot(
            thresholds,
            purities,
            label=f"{name} (MC)",
            linewidth=2.5,
            marker="o",
            markersize=2,
            markevery=10,
        )

    ax2.set_xlabel("Classification Threshold", fontsize=14)
    ax2.set_ylabel("Signal Purity (Signal / Total)", fontsize=14)
    ax2.set_title("With Domain Adaptation", fontsize=16, fontweight="bold")
    ax2.legend(fontsize=12)
    ax2.grid(alpha=0.3)
    ax2.set_ylim(0, 1.05)
    ax2.axhline(y=0.5, color="r", linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()


def plot_signal_noise_density_grid(
    preds_no_da,
    labels_no_da,
    preds_da,
    labels_da,
    preds_unlabeled_no_da,
    preds_unlabeled_da,
    save_path,
):
    """
    Plot: 2 plots comparing MC vs Exp distributions
    Left: No-DA, Right: With DA
    Each plot shows MC (all hits) vs Exp (all hits)
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    bins = np.linspace(0, 0.3, 60)

    # Get dataset names
    mc_name = list(preds_no_da.keys())[0] if preds_no_da else None
    exp_name = list(preds_unlabeled_no_da.keys())[0] if preds_unlabeled_no_da else None

    # Rename sig_noise_2020 to mc_2020 for display
    mc_display_name = mc_name.replace("sig_noise", "mc") if mc_name else "MC"

    # Left plot: No-DA (MC vs Exp, all hits)
    if mc_name and exp_name:
        ax = axes[0]
        mc_preds = preds_no_da[mc_name]
        exp_preds = preds_unlabeled_no_da[exp_name]

        ax.hist(
            mc_preds,
            bins=bins,
            label=f"{mc_display_name} (n={len(mc_preds):,})",
            density=True,
            histtype="step",
            color="blue",
            linewidth=2.5,
            alpha=0.8,
        )
        ax.hist(
            exp_preds,
            bins=bins,
            label=f"{exp_name} (n={len(exp_preds):,})",
            density=True,
            histtype="step",
            color="orange",
            linewidth=2.5,
            alpha=0.8,
            linestyle="--",
        )

        ax.set_xlabel("Predicted Signal Probability", fontsize=14)
        ax.set_ylabel("Normalized Density (log scale)", fontsize=14)
        ax.set_title("No Domain Adaptation", fontsize=16, fontweight="bold")
        ax.legend(fontsize=12)
        ax.grid(alpha=0.3, which="both", linestyle=":")
        ax.set_xlim(0, 0.3)
        ax.set_yscale("log")
        ax.set_ylim(bottom=1e-3)

    # Right plot: DA (MC vs Exp, all hits)
    if mc_name and exp_name:
        ax = axes[1]
        mc_preds = preds_da[mc_name]
        exp_preds = preds_unlabeled_da[exp_name]

        ax.hist(
            mc_preds,
            bins=bins,
            label=f"{mc_display_name} (n={len(mc_preds):,})",
            density=True,
            histtype="step",
            color="blue",
            linewidth=2.5,
            alpha=0.8,
        )
        ax.hist(
            exp_preds,
            bins=bins,
            label=f"{exp_name} (n={len(exp_preds):,})",
            density=True,
            histtype="step",
            color="orange",
            linewidth=2.5,
            alpha=0.8,
            linestyle="--",
        )

        ax.set_xlabel("Predicted Signal Probability", fontsize=14)
        ax.set_ylabel("Normalized Density (log scale)", fontsize=14)
        ax.set_title("With Domain Adaptation", fontsize=16, fontweight="bold")
        ax.legend(fontsize=12)
        ax.grid(alpha=0.3, which="both", linestyle=":")
        ax.set_xlim(0, 0.3)
        ax.set_yscale("log")
        ax.set_ylim(bottom=1e-3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Compare two models (No-DA vs DA)")
    parser.add_argument("--checkpoint_no_da", type=str, required=True)
    parser.add_argument("--checkpoint_da", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output", type=str, default="plots/comparison.png")

    args = parser.parse_args()

    # Load both models and get predictions + hidden states
    (
        preds_no_da,
        labels_no_da,
        preds_unlabeled_no_da,
        hidden_no_da,
        hidden_labels_no_da,
    ) = load_model_predictions(args.checkpoint_no_da, args.config, "No-DA")

    preds_da, labels_da, preds_unlabeled_da, hidden_da, hidden_labels_da = load_model_predictions(
        args.checkpoint_da, args.config, "DA"
    )
    for key in preds_no_da.keys():
        min_len = min(len(preds_no_da[key]), len(preds_da[key]))
        preds_no_da[key] = preds_no_da[key][:min_len]
        preds_da[key] = preds_da[key][:min_len]
    for key in preds_unlabeled_no_da.keys():
        min_len = min(len(preds_unlabeled_no_da[key]), len(preds_unlabeled_da[key]))
        preds_unlabeled_no_da[key] = preds_unlabeled_no_da[key][:min_len]
        preds_unlabeled_da[key] = preds_unlabeled_da[key][:min_len]

    for key in hidden_no_da.keys():
        min_len = min(len(hidden_no_da[key]), len(hidden_da[key]))
        hidden_no_da[key] = hidden_no_da[key][:min_len]
        hidden_da[key] = hidden_da[key][:min_len]

    # hidden_labels_no_da and hidden_labels_da are None, so skip them

    # Compute KL divergence
    print("\n" + "=" * 80)
    print("KL DIVERGENCE ANALYSIS: MC vs Exp Distributions")
    print("=" * 80)

    mc_name = list(preds_no_da.keys())[0] if preds_no_da else None
    exp_name = list(preds_unlabeled_no_da.keys())[0] if preds_unlabeled_no_da else None

    if mc_name and exp_name:
        # No-DA model
        mc_preds_no_da = preds_no_da[mc_name]
        exp_preds_no_da = preds_unlabeled_no_da[exp_name]

        kl_no_da_mc_exp = compute_kl_divergence(mc_preds_no_da, exp_preds_no_da)
        kl_no_da_exp_mc = compute_kl_divergence(exp_preds_no_da, mc_preds_no_da)
        js_no_da = 0.5 * (kl_no_da_mc_exp + kl_no_da_exp_mc)

        print("\n📊 WITHOUT Domain Adaptation:")
        print(f"   KL(MC || Exp):  {kl_no_da_mc_exp:.6f}")
        print(f"   KL(Exp || MC):  {kl_no_da_exp_mc:.6f}")
        print(f"   JS Divergence:  {js_no_da:.6f}")

        # DA model
        mc_preds_da = preds_da[mc_name]
        exp_preds_da = preds_unlabeled_da[exp_name]

        kl_da_mc_exp = compute_kl_divergence(mc_preds_da, exp_preds_da)
        kl_da_exp_mc = compute_kl_divergence(exp_preds_da, mc_preds_da)
        js_da = 0.5 * (kl_da_mc_exp + kl_da_exp_mc)

        print("\n📊 WITH Domain Adaptation:")
        print(f"   KL(MC || Exp):  {kl_da_mc_exp:.6f}")
        print(f"   KL(Exp || MC):  {kl_da_exp_mc:.6f}")
        print(f"   JS Divergence:  {js_da:.6f}")

        print("\n" + "=" * 80)
        print("🎯 DOMAIN ADAPTATION EFFECTIVENESS:")
        print("=" * 80)
        improvement = js_no_da - js_da
        pct_improvement = improvement / js_no_da * 100
        print(f"   JS Divergence:  {js_no_da:.6f} → {js_da:.6f}")
        print(f"   Absolute Reduction: {improvement:.6f}")
        print(f"   Relative Improvement: {pct_improvement:.1f}%")
        if js_da < js_no_da:
            print("   ✅ Domain Adaptation SUCCESSFULLY brings distributions closer!")
        else:
            print("   ❌ Domain Adaptation INCREASES divergence")
        print("=" * 80)

    # Generate comparison plots
    print("\n" + "=" * 80)
    print("GENERATING COMPARISON PLOTS")
    print("=" * 80)

    # Plot 1: MC vs Exp distribution comparison (all hits)
    plot1_path = args.output.replace(".png", "_distributions.png")
    plot_signal_distributions_comparison(
        preds_no_da,
        labels_no_da,
        preds_da,
        labels_da,
        preds_unlabeled_no_da,
        preds_unlabeled_da,
        plot1_path,
    )

    # Plot 2: 2x2 grid of signal/noise densities
    plot2_path = args.output.replace(".png", "_density_grid.png")
    plot_signal_noise_density_grid(
        preds_no_da,
        labels_no_da,
        preds_da,
        labels_da,
        preds_unlabeled_no_da,
        preds_unlabeled_da,
        plot2_path,
    )

    # Plot 3: UMAP visualization of hidden states
    plot3_path = args.output.replace(".png", "_umap.png")
    plot_umap_comparison(hidden_no_da, hidden_da, plot3_path)

    print("\n✓ All comparison plots generated!")


if __name__ == "__main__":
    main()
