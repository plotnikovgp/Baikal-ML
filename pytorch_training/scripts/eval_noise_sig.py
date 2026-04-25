"""Evaluate noise/signal models: classification, regression, and OR-label training.

Usage:
    # Legacy tres-only labels (|t_res| < cut):
    python scripts/eval_noise_sig.py \
        --checkpoint checkpoints/noise_sig_experiments/a_tres_cut_10ns_hs128/best.ckpt \
        --mode classification --signal-definition tres \
        --data /path/to/mc_merged.h5 --output-dir plots/eval_a

    # h / i models: signal = |t_res| < cut OR label>0 (same as training):
    python scripts/eval_noise_sig.py \
        --checkpoint checkpoints/noise_sig_experiments/h_tres10_or_labels_old_2020_hs128/best.ckpt \
        --mode classification --signal-definition or_labels --tres-cut 10 \
        --data /home2/ivkhar/Baikal/data/normed/baikal_2020_sig-noise_mid-eq_normed.h5 \
        --output-dir plots/eval_h_or_labels

    # Regression (t_res):
    python scripts/eval_noise_sig.py \
        --checkpoint checkpoints/noise_sig_experiments/c_tres_regression_hs128/best.ckpt \
        --mode regression --output-dir plots/eval_c_tres_regression
"""

import argparse
import sys
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_utils import BaikalDataset, create_dataloaders
from data_utils import readers as _readers
from data_utils.preprocessors import (
    DataPrefilter,
    NoiseSigOriginalLabelsPreprocessor,
    NoiseSigOrLabelsPreprocessor,
    NoiseSigPreprocessor,
    TresRegressionPreprocessor,
)
from models import Encoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_PATH = "/home/plotnikovgp/baikal/data/baikal_mc_merged_m50-na15-nue15_norm.h5"

TYPE_NAMES = {0: "muatm", 1: "nuatm", 2: "nue2"}
TYPE_COLORS = {0: "#d62728", 1: "#ff7f0e", 2: "#9467bd"}

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "figure.titlesize": 15,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": ":",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--mode", choices=["classification", "regression"], required=True)
    p.add_argument(
        "--signal-definition",
        choices=["tres", "or_labels", "labels_only"],
        default="tres",
        help=(
            "tres: |t_res|<tres-cut only. "
            "or_labels: |t_res|<tres-cut OR |label|>0. "
            "labels_only: |label|>0 (ignores tres)."
        ),
    )
    p.add_argument("--output-dir", default="plots/eval_noise_sig")
    p.add_argument("--data", default=DATA_PATH)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--max-batches", type=int, default=None)
    p.add_argument("--tres-cut", type=float, default=10.0)
    p.add_argument("--max-tres", type=float, default=100.0)
    p.add_argument("--split", default="val")
    p.add_argument(
        "--val-subset-cut",
        type=int,
        default=3,
        help="Match training val subset (every N-th val event).",
    )
    p.add_argument("--hidden-size", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=5)
    p.add_argument("--dim-feedforward", type=int, default=512)
    p.add_argument("--n-heads", type=int, default=1)
    p.add_argument(
        "--pr-bar-threshold",
        type=float,
        default=0.5,
        help="Threshold for P/R bar chart (classification only; by particle type).",
    )
    p.add_argument(
        "--max-seq-len",
        type=int,
        default=2048,
        help="Override MAX_SEQ_LEN from data_utils.readers. Default 2048 keeps ~100%% of events intact at eval; training used 256.",
    )
    p.add_argument(
        "--min-event-signal-hits",
        type=int,
        default=1,
        help="Macro mode: min GT signal hits to include event in the per-event recall average.",
    )
    p.add_argument(
        "--max-events-per-type",
        type=int,
        default=None,
        help="Stop after collecting this many events of each particle type (muatm/nuatm/nue2).",
    )
    p.add_argument(
        "--out-size",
        type=int,
        default=None,
        help="Override model out_size (default: 2 for classification, 1 for regression, "
        "3 for multi-task classification+tres). Cls prob is taken from output[:,1].",
    )
    return p.parse_args()


def load_model(checkpoint_path, out_size, enc_kwargs: dict):
    model = Encoder(**enc_kwargs, out_size=out_size)
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict)
    model.to(DEVICE).eval()
    return model


def build_encoder_params(args):
    return dict(
        in_features=5,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dim_feedforward_size=args.dim_feedforward,
        n_heads=args.n_heads,
        dropout_p=0.0,
        use_cls_token=False,
        return_only_cls_token=False,
    )


def build_preprocessor(args):
    prefilter = DataPrefilter()
    if args.mode == "regression":
        return TresRegressionPreprocessor(prefilter, max_tres=args.max_tres)
    if args.signal_definition == "or_labels":
        p = NoiseSigOrLabelsPreprocessor(
            prefilter, tres_cut_for_track_hit=args.tres_cut, z_mirror=False
        )
    elif args.signal_definition == "labels_only":
        p = NoiseSigOriginalLabelsPreprocessor(prefilter)
    else:
        p = NoiseSigPreprocessor(prefilter, tres_cut_for_track_hit=args.tres_cut, z_mirror=False)
    p.eval()
    return p


def run_inference(model, loader, is_classification, max_batches=None):
    """Run inference. Assumes loader batches match event order on disk
    (create_dataloaders uses shuffle=False and subsamples val deterministically)."""
    all_preds, all_trues, all_event_sizes = [], [], []
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

            mask_cpu = mask.cpu()
            output_flat = output.reshape(-1, output.shape[-1]).squeeze(-1).cpu()
            y_flat = y_true.reshape(-1).cpu()
            mask_flat = mask_cpu.reshape(-1) != 0

            if is_classification:
                pred = torch.sigmoid(output_flat[mask_flat][:, 1]).numpy()
            else:
                pred = output_flat[mask_flat].numpy()

            all_preds.append(pred)
            all_trues.append(y_flat[mask_flat].numpy())
            all_event_sizes.append(mask_cpu.sum(dim=1).numpy().astype(np.int64))

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_trues)
    event_sizes = np.concatenate(all_event_sizes)
    return y_pred, y_true, event_sizes


def run_inference_with_type_cap(
    model,
    loader,
    is_classification,
    ev_types_full: np.ndarray,
    cap_per_type: int | None,
    max_batches: int | None = None,
):
    """Same as run_inference but also filters events to at most cap_per_type per type.
    ev_types_full gives the type for every event in the dataloader's iteration order."""
    all_preds, all_trues, all_event_sizes, all_ev_types = [], [], [], []
    type_counts = {0: 0, 1: 0, 2: 0}
    ev_global_idx = 0

    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc="Inference")):
            if max_batches and i >= max_batches:
                break
            if isinstance(batch, (list, tuple)):
                x, y_true, mask = batch[0], batch[1], batch[2]
            else:
                x, y_true, mask = batch

            bs = mask.shape[0]
            batch_types = ev_types_full[ev_global_idx : ev_global_idx + bs]
            ev_global_idx += bs

            if cap_per_type is not None:
                keep = np.array(
                    [type_counts[int(t)] < cap_per_type for t in batch_types], dtype=bool
                )
                if not keep.any():
                    if all(type_counts[t] >= cap_per_type for t in (0, 1, 2)):
                        break
                    continue
            else:
                keep = np.ones(bs, dtype=bool)

            x, mask = x.to(DEVICE), mask.to(DEVICE)
            output = model(x, mask)

            mask_cpu = mask.cpu()
            output_flat = output.reshape(-1, output.shape[-1]).squeeze(-1).cpu()
            y_flat = y_true.reshape(-1).cpu()
            mask_flat = mask_cpu.reshape(-1) != 0

            sizes_batch = mask_cpu.sum(dim=1).numpy().astype(np.int64)
            per_event_pred = output_flat[mask_flat]
            per_event_true = y_flat[mask_flat]

            splits = np.cumsum(sizes_batch)
            offsets = np.concatenate([[0], splits])

            for j in np.flatnonzero(keep):
                t = int(batch_types[j])
                if cap_per_type is not None and type_counts[t] >= cap_per_type:
                    continue
                s, f = offsets[j], offsets[j + 1]
                if s == f:
                    type_counts[t] += 1
                    all_event_sizes.append(0)
                    all_ev_types.append(t)
                    continue
                chunk = per_event_pred[s:f]
                if is_classification:
                    pred = torch.sigmoid(chunk[:, 1]).numpy()
                else:
                    pred = chunk.numpy()
                all_preds.append(pred)
                all_trues.append(per_event_true[s:f].numpy())
                all_event_sizes.append(int(f - s))
                all_ev_types.append(t)
                type_counts[t] += 1

            if cap_per_type is not None and all(type_counts[t] >= cap_per_type for t in (0, 1, 2)):
                print(f"  cap reached for all types: {type_counts}")
                break

    y_pred = np.concatenate(all_preds) if all_preds else np.array([])
    y_true = np.concatenate(all_trues) if all_trues else np.array([])
    event_sizes = np.array(all_event_sizes, dtype=np.int64)
    ev_types = np.array(all_ev_types, dtype=np.int32)
    print(f"  per-type counts: {type_counts}")
    return y_pred, y_true, event_sizes, ev_types


def get_per_event_types_ordered(data_path, split, val_subset_cut, batch_size):
    """Event types in the exact order the dataloader iterates (for both train and val_subset)."""
    with h5py.File(data_path, "r") as f:
        ev_ids = f[f"{split}/ev_ids/data"][:]
    types = np.array(
        [0 if b"muatm" in eid else (1 if b"nuatm" in eid else 2) for eid in ev_ids],
        dtype=np.int32,
    )
    n_events = len(ev_ids)
    n_batches = (n_events + batch_size - 1) // batch_size

    if split == "val":
        batch_idxs = list(range(0, n_batches, val_subset_cut))
    else:
        batch_idxs = list(range(n_batches))

    out = []
    for bi in batch_idxs:
        out.append(types[bi * batch_size : min((bi + 1) * batch_size, n_events)])
    return np.concatenate(out)


def get_per_event_types(data_path, split, val_subset_cut, batch_size, n_events_eval):
    """Return per-event particle type for the eval subset in the same order as the dataloader.

    The val subset uses Subset(val_ds, range(0, n_batches, val_subset_cut)) — i.e., every
    val_subset_cut-th batch — so we replicate that indexing here.
    """
    with h5py.File(data_path, "r") as f:
        ev_ids = f[f"{split}/ev_ids/data"][:]
    n_events = len(ev_ids)
    n_batches = (n_events + batch_size - 1) // batch_size
    batch_idxs = list(range(0, n_batches, val_subset_cut))

    selected = []
    for bi in batch_idxs:
        selected.append(ev_ids[bi * batch_size : min((bi + 1) * batch_size, n_events)])
    ev_ids_ordered = np.concatenate(selected)[:n_events_eval]
    types = np.array(
        [0 if b"muatm" in eid else (1 if b"nuatm" in eid else 2) for eid in ev_ids_ordered],
        dtype=np.int32,
    )
    return types


# --------------- Plotting functions ---------------


def _compute_per_event_pr(
    y_pred_prob: np.ndarray,
    y_true: np.ndarray,
    event_ids: np.ndarray,
    thresholds: np.ndarray,
    min_signal_hits: int = 1,
):
    """Per-event precision and recall at each threshold.

    Returns:
        P: (n_thresholds, n_events_kept) - NaN when an event has no predicted positives.
        R: (n_thresholds, n_events_kept) - NaN when an event has no ground-truth positives.
        ev_mask: boolean mask over original event order (which events were kept).
    """
    n_events = event_ids.max() + 1
    order = np.argsort(event_ids, kind="stable")
    probs_s = y_pred_prob[order]
    labels_s = y_true[order]
    ids_s = event_ids[order]
    boundaries = np.searchsorted(ids_s, np.arange(n_events + 1))

    n_sig = np.zeros(n_events, dtype=np.int64)
    for e in range(n_events):
        n_sig[e] = labels_s[boundaries[e] : boundaries[e + 1]].sum()
    keep = n_sig >= min_signal_hits
    kept_events = np.flatnonzero(keep)

    P = np.full((len(thresholds), kept_events.size), np.nan, dtype=np.float32)
    R = np.full((len(thresholds), kept_events.size), np.nan, dtype=np.float32)

    for i, t in enumerate(thresholds):
        preds_s = probs_s > t
        for out_i, e in enumerate(kept_events):
            s, f = boundaries[e], boundaries[e + 1]
            if f == s:
                continue
            pred = preds_s[s:f]
            label = labels_s[s:f]
            tp = int((pred & (label == 1)).sum())
            pp = int(pred.sum())
            ap = int(label.sum())
            if pp > 0:
                P[i, out_i] = tp / pp
            if ap > 0:
                R[i, out_i] = tp / ap
    return P, R, keep


def _micro_pr_vs_threshold(y_pred_prob, y_true, thresholds):
    precisions, recalls = [], []
    for t in thresholds:
        p = y_pred_prob > t
        tp = int((p & (y_true == 1)).sum())
        fp = int((p & (y_true == 0)).sum())
        fn = int((~p & (y_true == 1)).sum())
        pr = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rc = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precisions.append(pr)
        recalls.append(rc)
    return np.array(precisions), np.array(recalls)


def _remap_event_ids(hit_event_ids, hit_mask, ev_mask, ev_types):
    """Return (hit_event_ids filtered to hit_mask, remapped to contiguous 0..K-1), n_events."""
    if ev_mask.sum() == 0:
        return None, 0
    sel_ids = hit_event_ids[hit_mask]
    uniq = np.flatnonzero(ev_mask)
    id_map = -np.ones(ev_types.size, dtype=np.int64)
    id_map[uniq] = np.arange(uniq.size)
    remapped = id_map[sel_ids]
    if np.any(remapped < 0):
        remapped = remapped[remapped >= 0]
    return remapped.astype(np.int64), uniq.size


def _apply_pr_axes(ax, title=None, color=None, xlabel="Recall", ylabel="Precision"):
    ax.set_xlim(0.85, 1.002)
    ax.set_ylim(0.85, 1.002)
    ax.grid(True, which="major", alpha=0.3, linestyle=":")
    ax.grid(True, which="minor", alpha=0.12, linestyle=":")
    ax.minorticks_on()
    ax.set_xlabel(xlabel, fontweight="bold")
    ax.set_ylabel(ylabel, fontweight="bold")
    if title:
        ax.set_title(title, fontweight="bold", color=color)


def plot_pr_curves_micro(
    y_pred_prob,
    y_true,
    hit_ev_types,
    output_dir,
    title_suffix="",
    filename="pr_curves_micro.png",
):
    """Micro PR curves per event type (hits pooled)."""
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.5))

    for col, (t, name) in enumerate(TYPE_NAMES.items()):
        ax = axes[col]
        color = TYPE_COLORS[t]
        hit_mask = hit_ev_types == t
        if hit_mask.sum() == 0 or len(np.unique(y_true[hit_mask])) < 2:
            ax.text(0.5, 0.5, f"N/A for {name}", ha="center", va="center", transform=ax.transAxes)
            _apply_pr_axes(ax, title=name, color=color)
            continue
        probs = y_pred_prob[hit_mask]
        labels = y_true[hit_mask]
        p_micro, r_micro, _ = precision_recall_curve(labels, probs)
        auc_val = roc_auc_score(labels, probs)
        ap_val = average_precision_score(labels, probs)

        ax.plot(r_micro, p_micro, linewidth=2.2, color=color)
        ax.fill_between(r_micro, p_micro, 0.0, color=color, alpha=0.08)
        ax.text(
            0.03,
            0.06,
            f"AUC = {auc_val:.4f}\nAP  = {ap_val:.4f}\nhits = {hit_mask.sum():,}",
            transform=ax.transAxes,
            fontsize=9,
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#cccccc", alpha=0.9),
            family="monospace",
        )
        _apply_pr_axes(ax, title=name, color=color)

    suptitle = "Precision-Recall (micro, pooled hits) by event type"
    if title_suffix:
        suptitle += f"  —  {title_suffix}"
    fig.suptitle(suptitle, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = output_dir / filename
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_pr_curves_macro(
    y_pred_prob,
    y_true,
    hit_ev_types,
    hit_event_ids,
    ev_types,
    output_dir,
    min_signal_hits=1,
    title_suffix="",
    filename="pr_curves_macro.png",
):
    """Macro PR curves per event type (per-event P/R averaged)."""
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.5))
    thresholds_macro = np.linspace(0.01, 0.99, 60)

    for col, (t, name) in enumerate(TYPE_NAMES.items()):
        ax = axes[col]
        color = TYPE_COLORS[t]
        hit_mask = hit_ev_types == t
        ev_mask = ev_types == t
        event_ids_remapped, _ = _remap_event_ids(hit_event_ids, hit_mask, ev_mask, ev_types)
        if event_ids_remapped is None or event_ids_remapped.size == 0:
            ax.text(0.5, 0.5, f"N/A for {name}", ha="center", va="center", transform=ax.transAxes)
            _apply_pr_axes(ax, title=name, color=color)
            continue

        probs = y_pred_prob[hit_mask]
        labels = y_true[hit_mask]
        P, R, keep = _compute_per_event_pr(
            probs, labels, event_ids_remapped, thresholds_macro, min_signal_hits=min_signal_hits
        )
        mean_P = np.nanmean(P, axis=1)
        mean_R = np.nanmean(R, axis=1)
        order = np.argsort(mean_R)
        ax.plot(mean_R[order], mean_P[order], linewidth=2.2, color=color)
        ax.fill_between(mean_R[order], mean_P[order], 0.0, color=color, alpha=0.08)
        try:
            ap_macro = float(np.trapz(mean_P[order], mean_R[order]))
        except Exception:
            ap_macro = float("nan")
        ax.text(
            0.03,
            0.06,
            f"AP = {ap_macro:.4f}\nevents = {int(keep.sum()):,}\nmin sig hits = {min_signal_hits}",
            transform=ax.transAxes,
            fontsize=9,
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#cccccc", alpha=0.9),
            family="monospace",
        )
        _apply_pr_axes(ax, title=name, color=color)

    suptitle = "Precision-Recall (macro, per-event average) by event type"
    if title_suffix:
        suptitle += f"  —  {title_suffix}"
    fig.suptitle(suptitle, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = output_dir / filename
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def _apply_thr_axes(ax, title=None, color=None, is_left=False):
    ax.set_ylim(0.82, 1.005)
    ax.grid(True, alpha=0.25, linestyle=":")
    ax.set_xlabel("Threshold", fontweight="bold")
    if is_left:
        ax.set_ylabel("Score", fontweight="bold")
    if title:
        ax.set_title(title, fontweight="bold", color=color)


def plot_pr_vs_threshold_micro(
    y_pred_prob,
    y_true,
    hit_ev_types,
    output_dir,
    title_suffix="",
    filename="pr_vs_threshold_micro.png",
):
    thresholds = np.linspace(0.05, 0.98, 80)
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.5), sharex=True)

    for col, (t, name) in enumerate(TYPE_NAMES.items()):
        ax = axes[col]
        color = TYPE_COLORS[t]
        hit_mask = hit_ev_types == t
        if hit_mask.sum() == 0:
            ax.text(0.5, 0.5, f"No {name}", ha="center", va="center", transform=ax.transAxes)
            _apply_thr_axes(ax, title=name, color=color, is_left=col == 0)
            continue
        p_mic, r_mic = _micro_pr_vs_threshold(
            y_pred_prob[hit_mask], y_true[hit_mask].astype(bool), thresholds
        )
        ax.plot(thresholds, p_mic, linewidth=2, color=color, label="Precision")
        ax.plot(thresholds, r_mic, linewidth=2, color=color, linestyle="--", label="Recall")

        ax.axvline(0.5, color="#888", linewidth=0.8, linestyle=":")
        idx05 = int(np.argmin(np.abs(thresholds - 0.5)))
        ax.annotate(
            f"@ t=0.50\nP={p_mic[idx05]:.3f} R={r_mic[idx05]:.3f}",
            xy=(0.5, r_mic[idx05]),
            xytext=(0.05, 0.08),
            textcoords="axes fraction",
            fontsize=9,
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#cccccc", alpha=0.9),
        )
        _apply_thr_axes(ax, title=name, color=color, is_left=col == 0)
        if col == 0:
            ax.legend(loc="center left", framealpha=0.95, bbox_to_anchor=(0.0, 0.55))

    suptitle = "P/R vs threshold (micro) by event type"
    if title_suffix:
        suptitle += f"  —  {title_suffix}"
    fig.suptitle(suptitle, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = output_dir / filename
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_pr_vs_threshold_macro(
    y_pred_prob,
    y_true,
    hit_ev_types,
    hit_event_ids,
    ev_types,
    output_dir,
    min_signal_hits=1,
    title_suffix="",
    filename="pr_vs_threshold_macro.png",
):
    thr_macro = np.linspace(0.05, 0.98, 40)
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.5), sharex=True)

    for col, (t, name) in enumerate(TYPE_NAMES.items()):
        ax = axes[col]
        color = TYPE_COLORS[t]
        hit_mask = hit_ev_types == t
        ev_mask = ev_types == t
        event_ids_remapped, _ = _remap_event_ids(hit_event_ids, hit_mask, ev_mask, ev_types)
        if event_ids_remapped is None or event_ids_remapped.size == 0:
            ax.text(0.5, 0.5, f"No {name}", ha="center", va="center", transform=ax.transAxes)
            _apply_thr_axes(ax, title=name, color=color, is_left=col == 0)
            continue
        P, R, _ = _compute_per_event_pr(
            y_pred_prob[hit_mask],
            y_true[hit_mask],
            event_ids_remapped,
            thr_macro,
            min_signal_hits=min_signal_hits,
        )
        mean_P = np.nanmean(P, axis=1)
        mean_R = np.nanmean(R, axis=1)
        ax.plot(thr_macro, mean_P, linewidth=2, color=color, label="Precision")
        ax.plot(thr_macro, mean_R, linewidth=2, color=color, linestyle="--", label="Recall")

        ax.axvline(0.5, color="#888", linewidth=0.8, linestyle=":")
        idx05 = int(np.argmin(np.abs(thr_macro - 0.5)))
        ax.annotate(
            f"@ t=0.50\nP={mean_P[idx05]:.3f} R={mean_R[idx05]:.3f}",
            xy=(0.5, mean_R[idx05]),
            xytext=(0.05, 0.08),
            textcoords="axes fraction",
            fontsize=9,
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#cccccc", alpha=0.9),
        )
        _apply_thr_axes(ax, title=name, color=color, is_left=col == 0)
        if col == 0:
            ax.legend(loc="center left", framealpha=0.95, bbox_to_anchor=(0.0, 0.55))

    suptitle = "P/R vs threshold (macro, per-event average) by event type"
    if title_suffix:
        suptitle += f"  —  {title_suffix}"
    fig.suptitle(suptitle, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = output_dir / filename
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def _compute_pr_table(
    y_pred_prob,
    y_true,
    hit_ev_types,
    hit_event_ids,
    ev_types,
    threshold=0.5,
    min_signal_hits=1,
):
    """Return per-type micro/macro P/R metrics table rows."""
    rows = []
    for t, name in TYPE_NAMES.items():
        hit_mask = hit_ev_types == t
        if hit_mask.sum() == 0 or len(np.unique(y_true[hit_mask])) < 2:
            rows.append(
                {
                    "name": name,
                    "n_hits": int(hit_mask.sum()),
                    "n_events": 0,
                    "p_micro": np.nan,
                    "r_micro": np.nan,
                    "p_macro": np.nan,
                    "r_macro": np.nan,
                }
            )
            continue
        probs = y_pred_prob[hit_mask]
        labels = y_true[hit_mask]
        p_mi = precision_score(labels, probs > threshold, zero_division=0)
        r_mi = recall_score(labels, probs > threshold, zero_division=0)

        ev_mask = ev_types == t
        event_ids_remapped, _ = _remap_event_ids(hit_event_ids, hit_mask, ev_mask, ev_types)
        if event_ids_remapped is None:
            p_ma, r_ma, n_ev = np.nan, np.nan, 0
        else:
            P, R, keep = _compute_per_event_pr(
                probs,
                labels,
                event_ids_remapped,
                np.array([threshold]),
                min_signal_hits=min_signal_hits,
            )
            p_ma = float(np.nanmean(P))
            r_ma = float(np.nanmean(R))
            n_ev = int(keep.sum())
        rows.append(
            {
                "name": name,
                "n_hits": int(hit_mask.sum()),
                "n_events": n_ev,
                "p_micro": p_mi,
                "r_micro": r_mi,
                "p_macro": p_ma,
                "r_macro": r_ma,
            }
        )
    return rows


def _plot_bars_single(rows, output_dir, mode, threshold, title_suffix, filename):
    """Bar chart of P/R per event type for either 'micro' or 'macro'."""
    names = [r["name"] for r in rows]
    ns_hits = [r["n_hits"] for r in rows]
    ns_events = [r["n_events"] for r in rows]
    key_p, key_r = f"p_{mode}", f"r_{mode}"
    precs = [r[key_p] for r in rows]
    recs = [r[key_r] for r in rows]

    x = np.arange(len(names))
    w = 0.32
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = [TYPE_COLORS[t] for t, _ in TYPE_NAMES.items()][: len(names)]

    for i, c in enumerate(colors):
        if np.isfinite(precs[i]):
            ax.bar(x[i] - w / 2, precs[i], w, color=c, edgecolor="black", linewidth=0.4)
        if np.isfinite(recs[i]):
            ax.bar(
                x[i] + w / 2,
                recs[i],
                w,
                color=c,
                alpha=0.55,
                edgecolor="black",
                linewidth=0.4,
                hatch="//",
            )

    for i, (pv, rv) in enumerate(zip(precs, recs)):
        if np.isfinite(pv):
            ax.text(x[i] - w / 2, pv + 0.003, f"{pv:.3f}", ha="center", va="bottom", fontsize=9)
        if np.isfinite(rv):
            ax.text(x[i] + w / 2, rv + 0.003, f"{rv:.3f}", ha="center", va="bottom", fontsize=9)

    all_vals = [v for v in precs + recs if np.isfinite(v)]
    y_lo = max(0.0, (min(all_vals) if all_vals else 0.9) - 0.05)

    ax.set_xticks(x)
    if mode == "micro":
        xlabels = [f"{n}\nhits={hc:,}" for n, hc in zip(names, ns_hits)]
    else:
        xlabels = [f"{n}\nevents={ec:,}" for n, ec in zip(names, ns_events)]
    ax.set_xticklabels(xlabels)
    ax.set_ylabel("Score")
    ax.set_ylim(y_lo, 1.01)
    ax.axhline(1.0, color="gray", linewidth=0.5)

    from matplotlib.patches import Patch

    legend_elems = [
        Patch(facecolor="gray", edgecolor="black", label="Precision"),
        Patch(facecolor="gray", alpha=0.55, hatch="//", edgecolor="black", label="Recall"),
    ]
    ax.legend(handles=legend_elems, loc="lower right", framealpha=0.95)

    st = f"{mode.capitalize()} P/R at threshold = {threshold:.2f}"
    if title_suffix:
        st += f"  —  {title_suffix}"
    ax.set_title(st, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.25, linestyle=":")
    plt.tight_layout()
    out = output_dir / filename
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_pr_bars_micro(rows, output_dir, threshold=0.5, title_suffix=""):
    _plot_bars_single(rows, output_dir, "micro", threshold, title_suffix, "pr_bars_micro.png")


def plot_pr_bars_macro(rows, output_dir, threshold=0.5, title_suffix=""):
    _plot_bars_single(rows, output_dir, "macro", threshold, title_suffix, "pr_bars_macro.png")


def plot_pr_micro_vs_macro_overall(
    y_pred_prob,
    y_true,
    hit_event_ids,
    output_dir,
    min_signal_hits=1,
    title_suffix="",
):
    """Single panel: overall micro PR curve vs overall macro PR curve."""
    fig, ax = plt.subplots(figsize=(7.5, 6))

    if len(np.unique(y_true)) >= 2:
        p_mic, r_mic, _ = precision_recall_curve(y_true, y_pred_prob)
        ap_mic = average_precision_score(y_true, y_pred_prob)
        ax.plot(
            r_mic,
            p_mic,
            linewidth=2.2,
            color="#1f77b4",
            label=f"Micro (pool hits)   AP={ap_mic:.4f}",
        )
        ax.fill_between(r_mic, p_mic, 0, color="#1f77b4", alpha=0.08)

    thr = np.linspace(0.01, 0.99, 60)
    P, R, keep = _compute_per_event_pr(
        y_pred_prob, y_true, hit_event_ids, thr, min_signal_hits=min_signal_hits
    )
    mean_P = np.nanmean(P, axis=1)
    mean_R = np.nanmean(R, axis=1)
    order = np.argsort(mean_R)
    try:
        ap_mac = float(np.trapz(mean_P[order], mean_R[order]))
    except Exception:
        ap_mac = float("nan")
    ax.plot(
        mean_R[order],
        mean_P[order],
        linewidth=2.2,
        color="#2ca02c",
        label=f"Macro (avg over {int(keep.sum()):,} events)   AP={ap_mac:.4f}",
    )
    ax.fill_between(mean_R[order], mean_P[order], 0, color="#2ca02c", alpha=0.08)

    _apply_pr_axes(ax)
    st = "Overall PR: micro vs macro"
    if title_suffix:
        st += f"  —  {title_suffix}"
    ax.set_title(st, fontweight="bold")
    ax.legend(loc="lower left", framealpha=0.95)
    plt.tight_layout()
    out = output_dir / "pr_micro_vs_macro_overall.png"
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


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

    _readers.MAX_SEQ_LEN = args.max_seq_len
    print(f"Using MAX_SEQ_LEN = {_readers.MAX_SEQ_LEN} (training default was 256)")

    is_classification = args.mode == "classification"
    out_size = args.out_size if args.out_size is not None else (2 if is_classification else 1)
    enc_kwargs = build_encoder_params(args)

    print(f"Loading model ({args.mode}) from {args.checkpoint}")
    model = load_model(args.checkpoint, out_size, enc_kwargs)

    preprocessor = build_preprocessor(args)

    print(f"Loading data from {args.data}")
    dataloaders = create_dataloaders(
        path_to_data=args.data,
        DatasetType=BaikalDataset,
        batch_size=args.batch_size,
        is_graph=False,
        is_classification=is_classification,
        preprocessor=preprocessor,
        val_subset_cut=args.val_subset_cut,
    )
    if args.split == "train":
        from torch.utils.data import DataLoader

        train_ds = dataloaders["train_dataset"]
        loader = DataLoader(train_ds, batch_size=None, shuffle=False, num_workers=1)
    else:
        loader = dataloaders[args.split]

    print("Computing per-event types (dataloader order)...")
    ev_types_full_order = get_per_event_types_ordered(
        args.data, args.split, args.val_subset_cut, args.batch_size
    )

    print(f"Running inference (cap_per_type={args.max_events_per_type})...")
    y_pred, y_true, event_sizes, ev_types = run_inference_with_type_cap(
        model,
        loader,
        is_classification,
        ev_types_full_order,
        cap_per_type=args.max_events_per_type,
        max_batches=args.max_batches,
    )
    n_events_eval = len(event_sizes)
    print(f"  Total hits: {len(y_pred):,}   events: {n_events_eval:,}")
    if n_events_eval > 0:
        print(
            f"  Events > 256 hits: {(event_sizes > 256).sum()}   "
            f"max event size: {int(event_sizes.max())}"
        )

    hit_ev_types = np.repeat(ev_types, event_sizes)
    hit_event_ids = np.repeat(np.arange(n_events_eval, dtype=np.int64), event_sizes)
    assert len(hit_ev_types) == len(y_pred), (len(hit_ev_types), len(y_pred))

    print(f"\nGenerating plots in {output_dir}")

    def _print_pr_table(rows, threshold, label):
        print(f"\n--- {label} (threshold={threshold:.2f}) ---")
        print(
            f"{'type':<8}{'n_hits':>12}{'n_events':>12}"
            f"{'P_micro':>10}{'R_micro':>10}{'P_macro':>10}{'R_macro':>10}"
        )
        for r in rows:

            def fmt(v):
                return f"{v:.4f}" if np.isfinite(v) else "  n/a "

            print(
                f"{r['name']:<8}{r['n_hits']:>12,}{r['n_events']:>12,}"
                f"{fmt(r['p_micro']):>10}{fmt(r['r_micro']):>10}"
                f"{fmt(r['p_macro']):>10}{fmt(r['r_macro']):>10}"
            )

    if is_classification:
        if args.signal_definition == "labels_only":
            title_bits = ["signal = |label|>0"]
        else:
            title_bits = [f"tres_cut={args.tres_cut}ns"]
            if args.signal_definition == "or_labels":
                title_bits.append("signal = |t_res|<cut OR |label|>0")
            else:
                title_bits.append("signal = |t_res|<cut only")
        title_sfx = ", ".join(title_bits)

        plot_pr_curves_micro(y_pred, y_true, hit_ev_types, output_dir, title_suffix=title_sfx)
        plot_pr_curves_macro(
            y_pred,
            y_true,
            hit_ev_types,
            hit_event_ids,
            ev_types,
            output_dir,
            min_signal_hits=args.min_event_signal_hits,
            title_suffix=title_sfx,
        )
        plot_pr_vs_threshold_micro(y_pred, y_true, hit_ev_types, output_dir, title_suffix=title_sfx)
        plot_pr_vs_threshold_macro(
            y_pred,
            y_true,
            hit_ev_types,
            hit_event_ids,
            ev_types,
            output_dir,
            min_signal_hits=args.min_event_signal_hits,
            title_suffix=title_sfx,
        )
        rows = _compute_pr_table(
            y_pred,
            y_true,
            hit_ev_types,
            hit_event_ids,
            ev_types,
            threshold=args.pr_bar_threshold,
            min_signal_hits=args.min_event_signal_hits,
        )
        plot_pr_bars_micro(
            rows, output_dir, threshold=args.pr_bar_threshold, title_suffix=title_sfx
        )
        plot_pr_bars_macro(
            rows, output_dir, threshold=args.pr_bar_threshold, title_suffix=title_sfx
        )
        plot_pr_micro_vs_macro_overall(
            y_pred,
            y_true,
            hit_event_ids,
            output_dir,
            min_signal_hits=args.min_event_signal_hits,
            title_suffix=title_sfx,
        )

        auc = roc_auc_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0
        ap = average_precision_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0
        print(f"\n=== Classification ({args.signal_definition}, tres_cut={args.tres_cut}ns) ===")
        print(f"Overall micro AUC: {auc:.4f}   AP: {ap:.4f}")

        _print_pr_table(rows, args.pr_bar_threshold, "Per-type P/R")

    else:
        plot_tres_histograms(y_pred, y_true, output_dir, tres_cut=args.tres_cut)
        plot_tres_scatter_and_residuals(y_pred, y_true, output_dir, tres_cut=args.tres_cut)
        plot_tres_metrics_vs_true(y_pred, y_true, output_dir, tres_cut=args.tres_cut)

        signal_prob = 1.0 - np.minimum(np.clip(y_pred, 0, None) / args.tres_cut, 1.0)
        true_signal = (y_true < args.tres_cut).astype(np.int32)

        reg_sfx = f"regression, cut={args.tres_cut}ns"
        plot_pr_curves_micro(
            signal_prob, true_signal, hit_ev_types, output_dir, title_suffix=reg_sfx
        )
        plot_pr_curves_macro(
            signal_prob,
            true_signal,
            hit_ev_types,
            hit_event_ids,
            ev_types,
            output_dir,
            min_signal_hits=args.min_event_signal_hits,
            title_suffix=reg_sfx,
        )
        plot_pr_vs_threshold_micro(
            signal_prob, true_signal, hit_ev_types, output_dir, title_suffix=reg_sfx
        )
        plot_pr_vs_threshold_macro(
            signal_prob,
            true_signal,
            hit_ev_types,
            hit_event_ids,
            ev_types,
            output_dir,
            min_signal_hits=args.min_event_signal_hits,
            title_suffix=reg_sfx,
        )
        rows = _compute_pr_table(
            signal_prob,
            true_signal,
            hit_ev_types,
            hit_event_ids,
            ev_types,
            threshold=args.pr_bar_threshold,
            min_signal_hits=args.min_event_signal_hits,
        )
        plot_pr_bars_micro(rows, output_dir, threshold=args.pr_bar_threshold, title_suffix=reg_sfx)
        plot_pr_bars_macro(rows, output_dir, threshold=args.pr_bar_threshold, title_suffix=reg_sfx)

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
