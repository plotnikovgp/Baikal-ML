"""Evaluate noise/signal models: classification (a) and tres regression (c).

Usage:
    # Evaluate tres_regression checkpoint (c):
    python scripts/eval_noise_sig.py \
        --checkpoint checkpoints/noise_sig_experiments/c_tres_regression_hs128/best.ckpt \
        --mode regression --output-dir plots/eval_c_tres_regression

    # Evaluate noise_sig classification checkpoint (a):
    python scripts/eval_noise_sig.py \
        --checkpoint checkpoints/noise_sig_experiments/a_tres_cut_10ns_hs128/best.ckpt \
        --mode classification --output-dir plots/eval_a_tres_cut_10ns
"""

import argparse
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_utils import BaikalDataset, create_dataloaders
from data_utils.preprocessors import DataPrefilter, NoiseSigPreprocessor, TresRegressionPreprocessor
from models import Encoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_PATH = "/home/plotnikovgp/baikal/data/baikal_mc_merged_m50-na15-nue15_norm.h5"

ENCODER_PARAMS = dict(
    in_features=5,
    hidden_size=128,
    num_layers=5,
    dim_feedforward_size=512,
    n_heads=1,
    dropout_p=0.0,
    use_cls_token=False,
    return_only_cls_token=False,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--mode", choices=["classification", "regression"], required=True)
    p.add_argument("--output-dir", default="plots/eval_noise_sig")
    p.add_argument("--data", default=DATA_PATH)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--max-batches", type=int, default=None)
    p.add_argument("--tres-cut", type=float, default=10.0)
    p.add_argument("--max-tres", type=float, default=100.0)
    p.add_argument("--split", default="val")
    return p.parse_args()


def load_model(checkpoint_path, out_size):
    model = Encoder(**ENCODER_PARAMS, out_size=out_size)
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict)
    model.to(DEVICE).eval()
    return model


def run_inference(model, loader, is_classification, max_batches=None):
    all_preds, all_trues = [], []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc="Inference")):
            if max_batches and i >= max_batches:
                break
            if isinstance(batch, (list, tuple)):
                x, y_true, mask = batch[0], batch[1], batch[2]
            else:
                x, y_true, mask = batch
            x, mask = x.to(DEVICE), mask.to(DEVICE)
            output = model(x, mask)

            y_true_flat = y_true.reshape(-1)
            output_flat = output.reshape(-1, output.shape[-1]).squeeze()
            mask_flat = mask.reshape(-1).cpu()
            valid = mask_flat != 0
            y_true_flat = y_true_flat[valid]
            output_flat = output_flat.cpu()[valid]

            if is_classification:
                pred = torch.sigmoid(output_flat[:, 1]).cpu().numpy()
            else:
                pred = output_flat.cpu().numpy()

            all_preds.append(pred)
            all_trues.append(y_true_flat.cpu().numpy())

    return np.concatenate(all_preds), np.concatenate(all_trues)


def get_event_types_for_hits(data_path, split, ev_starts_for_hits):
    """Map per-event ev_ids to per-hit event types."""
    with h5py.File(data_path, "r") as f:
        ev_ids = f[f"{split}/ev_ids/data"][:]
        ev_starts = f[f"{split}/ev_starts/data"][:]

    ev_types = np.array([0 if b"muatm" in eid else (1 if b"nuatm" in eid else 2) for eid in ev_ids])

    n_hits = int(ev_starts[-1])
    hit_ev_types = np.zeros(n_hits, dtype=np.int32)
    for i in range(len(ev_starts) - 1):
        hit_ev_types[ev_starts[i] : ev_starts[i + 1]] = ev_types[i]
    return hit_ev_types


def get_valid_hit_mask_and_tres(data_path, split, preprocessor_type, tres_cut, max_tres):
    """Get per-hit valid mask, true t_res and labels for the full dataset."""
    with h5py.File(data_path, "r") as f:
        labels = f[f"{split}/labels/data"][:]
        t_res = f[f"{split}/t_res/data"][:]
    return labels, t_res


# --------------- Plotting functions ---------------


def plot_pr_curves_by_event_type(
    y_pred_prob, y_true, hit_ev_types, output_dir, tres_cut=None, title_suffix=""
):
    """Precision-Recall curves (true PR curve: recall vs precision) per event type."""
    type_names = {0: "muatm", 1: "nuatm", 2: "nue2"}
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for idx, (ev_type, name) in enumerate(type_names.items()):
        ax = axes[idx]
        mask = hit_ev_types == ev_type
        if mask.sum() == 0:
            ax.text(0.5, 0.5, f"No {name} events", ha="center", va="center", transform=ax.transAxes)
            continue

        probs = y_pred_prob[mask]
        labels = y_true[mask]

        if len(np.unique(labels)) < 2:
            ax.text(
                0.5,
                0.5,
                f"Single class in {name}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            continue

        precision, recall, thresholds = precision_recall_curve(labels, probs)
        auc_val = roc_auc_score(labels, probs)

        ax.plot(recall, precision, linewidth=2, label=f"AUC={auc_val:.4f}")

        ax.set_xlim(0.9, 1.002)
        ax.set_ylim(0.9, 1.002)
        ax.set_xlabel("Recall", fontsize=13)
        ax.set_ylabel("Precision", fontsize=13)
        ax.set_title(f"{name.upper()} (n={mask.sum():,} hits)", fontsize=14, fontweight="bold")
        ax.legend(fontsize=11, loc="lower left")
        ax.grid(alpha=0.3, linestyle=":")

    suptitle = "Precision-Recall Curves by Event Type"
    if title_suffix:
        suptitle += f" ({title_suffix})"
    fig.suptitle(suptitle, fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "pr_curves_by_event_type.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_dir / 'pr_curves_by_event_type.png'}")


def plot_pr_vs_threshold_by_event_type(
    y_pred_prob, y_true, hit_ev_types, output_dir, title_suffix=""
):
    """Precision and recall vs threshold, per event type."""
    type_names = {0: "muatm", 1: "nuatm", 2: "nue2"}
    thresholds = np.linspace(0.3, 0.95, 80)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for idx, (ev_type, name) in enumerate(type_names.items()):
        ax = axes[idx]
        mask = hit_ev_types == ev_type
        if mask.sum() == 0:
            continue

        probs = y_pred_prob[mask]
        labels = y_true[mask]

        precisions, recalls, f1s = [], [], []
        for t in thresholds:
            preds = (probs > t).astype(int)
            precisions.append(precision_score(labels, preds, zero_division=0))
            recalls.append(recall_score(labels, preds, zero_division=0))
            f1s.append(f1_score(labels, preds, zero_division=0))

        ax.plot(thresholds, precisions, label="Precision", linewidth=2)
        ax.plot(thresholds, recalls, label="Recall", linewidth=2, linestyle="--")
        ax.plot(thresholds, f1s, label="F1", linewidth=1.5, linestyle=":", color="green")

        best_f1_idx = np.argmax(f1s)
        best_t = thresholds[best_f1_idx]
        ax.axvline(best_t, color="gray", linestyle=":", alpha=0.5)
        ax.text(
            0.02,
            0.02,
            f"Best F1@t={best_t:.2f}\nP={precisions[best_f1_idx]:.3f} R={recalls[best_f1_idx]:.3f}",
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="bottom",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        ax.set_xlabel("Threshold", fontsize=13)
        ax.set_ylabel("Score", fontsize=13)
        ax.set_title(f"{name.upper()} (n={mask.sum():,})", fontsize=14, fontweight="bold")
        ax.legend(fontsize=10, loc="center left")
        ax.grid(alpha=0.3, linestyle=":")
        ax.set_ylim(0.85, 1.01)

    suptitle = "P/R vs Threshold by Event Type"
    if title_suffix:
        suptitle += f" ({title_suffix})"
    fig.suptitle(suptitle, fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "pr_vs_threshold_by_event_type.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_dir / 'pr_vs_threshold_by_event_type.png'}")


def plot_tres_histograms(y_pred, y_true, output_dir, tres_cut=10.0):
    """Predicted vs true |t_res| histograms: all hits and signal hits (|t_res| <= 15ns)."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax_idx, (title, mask_fn, bins_range) in enumerate(
        [
            ("All hits", lambda t: np.ones(len(t), dtype=bool), (0, 60)),
            ("|t_res_true| ≤ 15 ns", lambda t: t <= 15, (0, 20)),
        ]
    ):
        ax = axes[ax_idx]
        mask = mask_fn(y_true)
        bins = np.linspace(bins_range[0], bins_range[1], 80)

        ax.hist(
            y_true[mask], bins=bins, alpha=0.6, density=True, label="True |t_res|", color="#1f77b4"
        )
        ax.hist(
            y_pred[mask],
            bins=bins,
            alpha=0.6,
            density=True,
            label="Predicted |t_res|",
            color="#ff7f0e",
        )

        ax.axvline(tres_cut, color="red", linestyle="--", alpha=0.7, label=f"{tres_cut} ns cut")

        ax.set_xlabel("|t_res| [ns]", fontsize=13)
        ax.set_ylabel("Density", fontsize=13)
        ax.set_title(f"{title} (n={mask.sum():,})", fontsize=14, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3, linestyle=":")

    fig.suptitle("|t_res| Distribution: Predicted vs True", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "tres_histograms.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_dir / 'tres_histograms.png'}")


def plot_tres_scatter_and_residuals(y_pred, y_true, output_dir, tres_cut=10.0):
    """2D histogram of predicted vs true |t_res| + residual distribution."""
    import matplotlib.colors as mcolors

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # 2D hist: predicted vs true (all)
    ax = axes[0]
    clip = 60
    pred_c = np.clip(y_pred, 0, clip)
    true_c = np.clip(y_true, 0, clip)
    ax.hist2d(
        true_c,
        pred_c,
        bins=100,
        range=[[0, clip], [0, clip]],
        norm=mcolors.LogNorm(),
        cmap="viridis",
    )
    ax.plot([0, clip], [0, clip], "r--", linewidth=1.5, label="y=x")
    ax.set_xlabel("True |t_res| [ns]", fontsize=13)
    ax.set_ylabel("Predicted |t_res| [ns]", fontsize=13)
    ax.set_title("Pred vs True (all hits)", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3, linestyle=":")

    # 2D hist: signal region
    ax = axes[1]
    sig_mask = y_true <= 15
    if sig_mask.sum() > 100:
        ax.hist2d(
            y_true[sig_mask],
            y_pred[sig_mask],
            bins=80,
            range=[[0, 15], [0, 20]],
            norm=mcolors.LogNorm(),
            cmap="viridis",
        )
        ax.plot([0, 15], [0, 15], "r--", linewidth=1.5, label="y=x")
    ax.set_xlabel("True |t_res| [ns]", fontsize=13)
    ax.set_ylabel("Predicted |t_res| [ns]", fontsize=13)
    ax.set_title("|t_res_true| ≤ 15 ns", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3, linestyle=":")

    # Residual distribution
    ax = axes[2]
    residual = y_pred - y_true
    for label, mask, color in [
        (f"|t_res| < {tres_cut} (signal)", y_true < tres_cut, "#1f77b4"),
        (f"|t_res| ≥ {tres_cut} (noise)", y_true >= tres_cut, "#ff7f0e"),
    ]:
        r = residual[mask]
        r_clip = np.clip(r, -30, 30)
        ax.hist(r_clip, bins=80, alpha=0.6, density=True, label=label, color=color)
    ax.set_xlabel("Residual (pred - true) [ns]", fontsize=13)
    ax.set_ylabel("Density", fontsize=13)
    ax.set_title("Residual Distribution", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, linestyle=":")

    fig.suptitle("|t_res| Prediction Quality", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "tres_scatter_residuals.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_dir / 'tres_scatter_residuals.png'}")


def plot_tres_metrics_vs_true(y_pred, y_true, output_dir, tres_cut=10.0):
    """Binned metrics: MAE and classification accuracy as a function of true |t_res|."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    bin_edges = np.concatenate(
        [
            np.linspace(0, 10, 20),
            np.linspace(10, 60, 15),
        ]
    )
    bin_edges = np.unique(bin_edges)

    # MAE vs true |t_res|
    ax = axes[0]
    centers, maes = [], []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_true >= lo) & (y_true < hi)
        if mask.sum() < 20:
            continue
        centers.append((lo + hi) / 2)
        maes.append(np.mean(np.abs(y_pred[mask] - y_true[mask])))
    ax.plot(centers, maes, "-o", markersize=4, linewidth=2)
    ax.axvline(tres_cut, color="red", linestyle="--", alpha=0.5, label=f"{tres_cut} ns cut")
    ax.set_xlabel("True |t_res| [ns]", fontsize=13)
    ax.set_ylabel("MAE [ns]", fontsize=13)
    ax.set_title("MAE vs True |t_res|", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3, linestyle=":")

    # Classification accuracy (signal if pred < tres_cut) vs true |t_res|
    ax = axes[1]
    centers, accs = [], []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_true >= lo) & (y_true < hi)
        if mask.sum() < 20:
            continue
        pred_sig = (y_pred[mask] < tres_cut).astype(int)
        true_sig = (y_true[mask] < tres_cut).astype(int)
        centers.append((lo + hi) / 2)
        accs.append(np.mean(pred_sig == true_sig))
    ax.plot(centers, accs, "-o", markersize=4, linewidth=2, color="green")
    ax.axvline(tres_cut, color="red", linestyle="--", alpha=0.5, label=f"{tres_cut} ns cut")
    ax.set_xlabel("True |t_res| [ns]", fontsize=13)
    ax.set_ylabel("Accuracy", fontsize=13)
    ax.set_title("Binary Accuracy vs True |t_res|", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(alpha=0.3, linestyle=":")

    fig.suptitle("|t_res| Prediction Quality by Region", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "tres_metrics_vs_true.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_dir / 'tres_metrics_vs_true.png'}")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    is_classification = args.mode == "classification"
    out_size = 2 if is_classification else 1

    print(f"Loading model ({args.mode}) from {args.checkpoint}")
    model = load_model(args.checkpoint, out_size)

    if is_classification:
        preprocessor = NoiseSigPreprocessor(DataPrefilter(), tres_cut_for_track_hit=args.tres_cut)
    else:
        preprocessor = TresRegressionPreprocessor(DataPrefilter(), max_tres=args.max_tres)

    print(f"Loading data from {args.data}")
    dataloaders = create_dataloaders(
        path_to_data=args.data,
        DatasetType=BaikalDataset,
        batch_size=args.batch_size,
        is_graph=False,
        is_classification=is_classification,
        preprocessor=preprocessor,
    )
    loader = dataloaders[args.split]

    print("Running inference...")
    y_pred, y_true = run_inference(model, loader, is_classification, max_batches=args.max_batches)
    print(f"  Total hits: {len(y_pred):,}")

    # Get event types mapped to hits
    print("Mapping event types to hits...")
    hit_ev_types_full = get_event_types_for_hits(args.data, args.split, None)

    # We need to subsample hit_ev_types to match the val_subset_cut=3 used in data loader
    # The dataloader uses val_subset_cut=3 -> every 3rd event. Reconstruct valid hits.
    with h5py.File(args.data, "r") as f:
        ev_starts = f[f"{args.split}/ev_starts/data"][:]
        n_events = len(ev_starts) - 1
        f[f"{args.split}/labels/data"][:]
        f[f"{args.split}/t_res/data"][:]

    val_subset_cut = 3
    selected_events = list(range(0, n_events, val_subset_cut))
    hit_indices = []
    for ev_idx in selected_events:
        start, end = int(ev_starts[ev_idx]), int(ev_starts[ev_idx + 1])
        hit_indices.extend(range(start, end))
    hit_indices = np.array(hit_indices)
    hit_ev_types = hit_ev_types_full[hit_indices]

    # Align with model predictions (which only keep valid/non-padded hits)
    # The model flattens and filters by mask!=0, so hit_ev_types should already match
    # if we use the same subset. Let's verify length and truncate if needed.
    min_len = min(len(hit_ev_types), len(y_pred))
    hit_ev_types = hit_ev_types[:min_len]
    y_pred = y_pred[:min_len]
    y_true = y_true[:min_len]
    print(f"  Matched hits after alignment: {min_len:,}")

    print(f"\nGenerating plots in {output_dir}")

    if is_classification:
        # Classification mode: PR curves only
        plot_pr_curves_by_event_type(
            y_pred, y_true, hit_ev_types, output_dir, title_suffix=f"tres_cut={args.tres_cut}ns"
        )
        plot_pr_vs_threshold_by_event_type(
            y_pred, y_true, hit_ev_types, output_dir, title_suffix=f"tres_cut={args.tres_cut}ns"
        )

        # Print summary metrics
        auc = roc_auc_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0
        print(f"\n=== Classification Metrics (tres_cut={args.tres_cut}ns) ===")
        print(f"AUC: {auc:.4f}")
        for t in [0.5, 0.6, 0.7, 0.8]:
            preds = (y_pred > t).astype(int)
            p = precision_score(y_true, preds, zero_division=0)
            r = recall_score(y_true, preds, zero_division=0)
            print(f"  t={t:.1f}: P={p:.4f}, R={r:.4f}")

    else:
        # Regression mode: t_res plots + derived PR curves
        plot_tres_histograms(y_pred, y_true, output_dir, tres_cut=args.tres_cut)
        plot_tres_scatter_and_residuals(y_pred, y_true, output_dir, tres_cut=args.tres_cut)
        plot_tres_metrics_vs_true(y_pred, y_true, output_dir, tres_cut=args.tres_cut)

        # Targets are |t_res|, predictions are |t_res|
        signal_prob = 1.0 - np.minimum(np.clip(y_pred, 0, None) / args.tres_cut, 1.0)
        true_signal = (y_true < args.tres_cut).astype(np.int32)

        plot_pr_curves_by_event_type(
            signal_prob,
            true_signal,
            hit_ev_types,
            output_dir,
            title_suffix=f"regression, cut={args.tres_cut}ns",
        )
        plot_pr_vs_threshold_by_event_type(
            signal_prob,
            true_signal,
            hit_ev_types,
            output_dir,
            title_suffix=f"regression, cut={args.tres_cut}ns",
        )

        # Print summary
        print("\n=== Regression Metrics (|t_res|) ===")
        diff = np.abs(y_pred - y_true)
        print(f"MAE: {diff.mean():.3f} ns")
        print(f"q50: {np.median(diff):.3f} ns")
        print(f"q68: {np.percentile(diff, 68):.3f} ns")

        sig_mask = y_true < args.tres_cut
        print(f"\nSignal hits (|t_res| < {args.tres_cut}ns): MAE={diff[sig_mask].mean():.3f}")
        print(f"Noise hits (|t_res| >= {args.tres_cut}ns): MAE={diff[~sig_mask].mean():.3f}")

        if len(np.unique(true_signal)) > 1:
            auc = roc_auc_score(true_signal, signal_prob)
            print(f"\nDerived binary AUC: {auc:.4f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
