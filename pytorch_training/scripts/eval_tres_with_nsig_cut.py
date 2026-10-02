"""Evaluate tres model on signal hits selected by noise/signal classifier (2-8 cut).

Generates 3 panels:
  1. MC: tres eval on hits predicted as signal by network (2-str, 8-hit cut)
  2. MC: tres eval on hits with ground truth label != 0
  3. EXP: tres eval on hits predicted as signal by network (2-str, 8-hit cut)
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
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.encoder import Encoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MC_DATA = "/home/plotnikovgp/baikal/data/baikal_mc_merged_m50-na15-nue15_all_smaller_norm.h5"
EXP_DATA = "/home/plotnikovgp/baikal/Baikal-ML/pytorch_training/data/baikal_exp_filtered_norm_w_model_enc_v2_fix_q.h5"

NSIG_CKPT = "/home/plotnikovgp/baikal/Baikal-ML/pytorch_training/checkpoints/noise_sig_experiments/k_nsol_labelneq0_hs128/best.ckpt"
TRES_CKPT = "/home/plotnikovgp/baikal/Baikal-ML/pytorch_training/checkpoints/noise_sig_experiments/tres_abs_merged_hs128/best.ckpt"

TRES_MEAN = 7.8
TRES_STD = 26.2

MIN_SIGNAL_HITS = 8
MIN_SIGNAL_STRINGS = 2
SIGNAL_THRESHOLD = 0.5

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "grid.alpha": 0.3,
        "grid.linestyle": ":",
    }
)


def load_norm_params(h5_path):
    with h5py.File(h5_path, "r") as f:
        return f["norm_param/mean"][:].astype(np.float32), f["norm_param/std"][:].astype(np.float32)


def renormalize(data, src_mean, src_std, dst_mean, dst_std):
    return (data * src_std + src_mean - dst_mean) / dst_std


def load_model(ckpt_path, out_size, hs=128, dff=512):
    model = Encoder(
        in_features=5,
        hidden_size=hs,
        num_layers=5,
        dim_feedforward_size=dff,
        n_heads=1,
        out_size=out_size,
        dropout_p=0.0,
        head_bias=(out_size == 1),
    )
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(sd)
    return model.to(DEVICE).eval()


def load_mc_events(h5_path, split, max_events):
    """Load MC events with labels and t_res."""
    with h5py.File(h5_path, "r") as f:
        ev_starts = f[f"{split}/ev_starts/data"][:]
        n = min(max_events, len(ev_starts) - 1)
        events = []
        for i in tqdm(range(n), desc=f"Loading MC {split}"):
            s, e = int(ev_starts[i]), int(ev_starts[i + 1])
            events.append(
                {
                    "data": f[f"{split}/data/data"][s:e].astype(np.float32),
                    "channels": f[f"{split}/channels/data"][s:e],
                    "labels": f[f"{split}/labels/data"][s:e],
                    "t_res": f[f"{split}/t_res/data"][s:e].astype(np.float32),
                }
            )
    return events


def load_exp_events(h5_path, split, max_events):
    """Load EXP events (no labels/t_res)."""
    with h5py.File(h5_path, "r") as f:
        ev_starts = f[f"{split}/ev_starts/data"][:]
        n = min(max_events, len(ev_starts) - 1)
        events = []
        for i in tqdm(range(n), desc=f"Loading EXP {split}"):
            s, e = int(ev_starts[i]), int(ev_starts[i + 1])
            events.append(
                {
                    "data": f[f"{split}/data/data"][s:e].astype(np.float32),
                    "channels": f[f"{split}/channels/data"][s:e],
                }
            )
    return events


def batch_inference(model, hits_list, batch_size=64):
    """Run model on variable-length hit sequences. Returns list of outputs."""
    results = []
    for bs in range(0, len(hits_list), batch_size):
        batch = hits_list[bs : bs + batch_size]
        max_len = max(len(h) for h in batch)
        x = np.zeros((len(batch), max_len, 5), dtype=np.float32)
        mask = np.zeros((len(batch), max_len), dtype=bool)
        for i, hits in enumerate(batch):
            L = len(hits)
            x[i, :L] = hits
            mask[i, :L] = True

        x_t = torch.tensor(x, device=DEVICE)
        mask_t = torch.tensor(mask, device=DEVICE)

        with torch.no_grad():
            out = model(x_t, mask_t)

        out_np = out.cpu().numpy()
        for i, hits in enumerate(batch):
            L = len(hits)
            results.append(out_np[i, :L])
    return results


def classify_hits(nsig_model, events, data_to_nsig_renorm_fn):
    """Run noise/signal classifier on events, return per-hit signal probabilities."""
    hits_for_model = [data_to_nsig_renorm_fn(ev["data"]) for ev in events]
    outputs = batch_inference(nsig_model, hits_for_model, batch_size=128)
    probs = []
    for out in outputs:
        p = torch.sigmoid(torch.tensor(out[:, 1])).numpy()
        probs.append(p)
    return probs


def predict_tres(tres_model, signal_hits_list, tres_mean=TRES_MEAN, tres_std=TRES_STD):
    """Run tres model on signal-only hits, return predicted |t_res| in ns."""
    if not signal_hits_list:
        return []
    outputs = batch_inference(tres_model, signal_hits_list, batch_size=128)
    predictions = []
    for out in outputs:
        pred_norm = out[:, 0]
        pred_ns = pred_norm * tres_std + tres_mean
        predictions.append(np.abs(pred_ns))
    return predictions


def apply_2_8_cut(
    probs, channels, threshold=0.5, min_hits=MIN_SIGNAL_HITS, min_strings=MIN_SIGNAL_STRINGS
):
    """Check if event passes 2-string 8-hit cut on predicted signal hits."""
    sig_mask = probs > threshold
    n_sig = sig_mask.sum()
    if n_sig < min_hits:
        return False, sig_mask
    n_strings = len(np.unique(channels[sig_mask] // 36))
    if n_strings < min_strings:
        return False, sig_mask
    return True, sig_mask


def compute_metrics(pred, true):
    """Compute tres evaluation metrics."""
    errors = pred - true
    abs_errors = np.abs(errors)
    return {
        "MAE": np.mean(abs_errors),
        "RMSE": np.sqrt(np.mean(errors**2)),
        "Median AE": np.median(abs_errors),
        "Q68": np.percentile(abs_errors, 68),
        "Q90": np.percentile(abs_errors, 90),
        "Bias": np.mean(errors),
        "N_hits": len(pred),
    }


def plot_tres_eval_panel(ax_row, pred, true, title, has_gt=True):
    """Plot a row of tres evaluation subplots.

    If has_gt=True: scatter, MAE-by-bin, error hist, metrics
    If has_gt=False: prediction distribution only
    """
    if has_gt:
        ax_scatter, ax_mae, ax_mae_zoom, ax_dist, ax_err, ax_metrics = ax_row

        true_abs = np.abs(true)
        pred_abs = pred

        # 1. Pred vs True 2D histogram
        mask_plot = true_abs < 50
        ax_scatter.hist2d(
            true_abs[mask_plot],
            pred_abs[mask_plot],
            bins=100,
            cmap="inferno",
            cmin=1,
        )
        lim = 50
        ax_scatter.plot([0, lim], [0, lim], "r--", lw=1, alpha=0.7)
        ax_scatter.set_xlabel("True |t_res| (ns)")
        ax_scatter.set_ylabel("Predicted |t_res| (ns)")
        ax_scatter.set_title("Pred vs True (|t_res| < 50 ns)")
        ax_scatter.set_xlim(0, lim)
        ax_scatter.set_ylim(0, lim)

        # 2. MAE by true |t_res| (full range)
        bin_edges = np.linspace(0, 200, 21)
        centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        maes = []
        for b0, b1 in zip(bin_edges[:-1], bin_edges[1:]):
            sel = (true_abs >= b0) & (true_abs < b1)
            if sel.sum() > 10:
                maes.append(np.mean(np.abs(pred_abs[sel] - true_abs[sel])))
            else:
                maes.append(np.nan)
        ax_mae.plot(centers, maes, "o-", color="#1f77b4", lw=2, ms=4)
        ax_mae.set_xlabel("True |t_res| (ns)")
        ax_mae.set_ylabel("MAE (ns)")
        ax_mae.set_title("MAE by true |t_res|")
        ax_mae.grid(True, alpha=0.3)

        # 3. MAE by true |t_res| < 30 ns
        bin_edges_z = np.linspace(0, 30, 16)
        centers_z = 0.5 * (bin_edges_z[:-1] + bin_edges_z[1:])
        maes_z = []
        for b0, b1 in zip(bin_edges_z[:-1], bin_edges_z[1:]):
            sel = (true_abs >= b0) & (true_abs < b1)
            if sel.sum() > 10:
                maes_z.append(np.mean(np.abs(pred_abs[sel] - true_abs[sel])))
            else:
                maes_z.append(np.nan)
        ax_mae_zoom.plot(centers_z, maes_z, "o-", color="#ff7f0e", lw=2, ms=4)
        ax_mae_zoom.set_xlabel("True |t_res| (ns)")
        ax_mae_zoom.set_ylabel("MAE (ns)")
        ax_mae_zoom.set_title("MAE by true |t_res| < 30 ns")
        ax_mae_zoom.grid(True, alpha=0.3)

        # 4. Prediction distribution
        bins_d = np.linspace(0, 80, 100)
        ax_dist.hist(
            pred_abs,
            bins=bins_d,
            density=True,
            histtype="step",
            lw=2,
            color="#1f77b4",
            label="Predicted",
        )
        ax_dist.hist(
            true_abs[true_abs < 80],
            bins=bins_d,
            density=True,
            histtype="step",
            lw=2,
            color="#2ca02c",
            ls="--",
            label="True",
        )
        ax_dist.set_xlabel("Predicted |t_res| (ns)")
        ax_dist.set_ylabel("Density")
        ax_dist.set_yscale("log")
        ax_dist.set_title("Prediction distribution")
        ax_dist.legend()
        ax_dist.grid(True, alpha=0.3)

        # 5. Error distribution
        errors = pred_abs - true_abs
        bins_e = np.linspace(-50, 50, 100)
        ax_err.hist(errors, bins=bins_e, color="#2ca02c", alpha=0.7)
        ax_err.axvline(0, color="red", ls="--", lw=1.5)
        ax_err.axvline(np.mean(errors), color="blue", ls="-", lw=1.5, alpha=0.7)
        ax_err.set_xlabel("Prediction error (ns)")
        ax_err.set_ylabel("Count")
        ax_err.set_title(
            f"Error distribution\nMean={np.mean(errors):.2f}, Std={np.std(errors):.2f}"
        )
        ax_err.grid(True, alpha=0.3)

        # 6. Metrics summary
        metrics = compute_metrics(pred_abs, true_abs)
        text = (
            f"Signal hits: {metrics['N_hits']:,}\n\n"
            f"MAE:       {metrics['MAE']:.2f} ns\n"
            f"RMSE:      {metrics['RMSE']:.2f} ns\n"
            f"Median AE: {metrics['Median AE']:.2f} ns\n"
            f"Q68:       {metrics['Q68']:.2f} ns\n"
            f"Q90:       {metrics['Q90']:.2f} ns\n"
            f"Bias:      {metrics['Bias']:.2f} ns"
        )
        ax_metrics.text(
            0.5,
            0.5,
            text,
            transform=ax_metrics.transAxes,
            ha="center",
            va="center",
            fontsize=11,
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", edgecolor="orange"),
        )
        ax_metrics.set_title("Metrics Summary")
        ax_metrics.axis("off")
    else:
        ax_dist, ax_dist_zoom, ax_metrics = ax_row

        # Prediction distribution (no GT)
        bins_d = np.linspace(0, 80, 100)
        ax_dist.hist(
            pred,
            bins=bins_d,
            density=True,
            histtype="step",
            lw=2,
            color="#1f77b4",
            label=f"Predicted ({len(pred):,} hits)",
        )
        ax_dist.set_xlabel("|t_res| predicted (ns)")
        ax_dist.set_ylabel("Density")
        ax_dist.set_yscale("log")
        ax_dist.set_title("Predicted |t_res| distribution")
        ax_dist.legend()
        ax_dist.grid(True, alpha=0.3)

        bins_z = np.linspace(0, 30, 80)
        ax_dist_zoom.hist(
            pred,
            bins=bins_z,
            density=True,
            histtype="step",
            lw=2,
            color="#1f77b4",
            label=f"EXP ({len(pred):,} hits)",
        )
        ax_dist_zoom.set_xlabel("|t_res| predicted (ns)")
        ax_dist_zoom.set_ylabel("Density")
        ax_dist_zoom.set_title("Predicted |t_res| < 30 ns")
        ax_dist_zoom.legend()
        ax_dist_zoom.grid(True, alpha=0.3)

        text = (
            f"Signal hits: {len(pred):,}\n\n"
            f"Mean pred:   {np.mean(pred):.2f} ns\n"
            f"Median pred: {np.median(pred):.2f} ns\n"
            f"Std pred:    {np.std(pred):.2f} ns\n"
            f"Q10:         {np.percentile(pred, 10):.2f} ns\n"
            f"Q90:         {np.percentile(pred, 90):.2f} ns"
        )
        ax_metrics.text(
            0.5,
            0.5,
            text,
            transform=ax_metrics.transAxes,
            ha="center",
            va="center",
            fontsize=11,
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", edgecolor="orange"),
        )
        ax_metrics.set_title("Prediction Stats")
        ax_metrics.axis("off")


def main():
    parser = argparse.ArgumentParser(description="Evaluate tres model with noise/signal 2-8 cut")
    parser.add_argument("--mc-data", default=MC_DATA)
    parser.add_argument("--exp-data", default=EXP_DATA)
    parser.add_argument("--nsig-ckpt", default=NSIG_CKPT)
    parser.add_argument("--tres-ckpt", default=TRES_CKPT)
    parser.add_argument("--mc-max-events", type=int, default=10000)
    parser.add_argument("--exp-max-events", type=int, default=10000)
    parser.add_argument("--output-dir", default="plots/tres_eval_with_nsig_cut")
    parser.add_argument("--threshold", type=float, default=SIGNAL_THRESHOLD)
    parser.add_argument("--tres-mean", type=float, default=TRES_MEAN)
    parser.add_argument("--tres-std", type=float, default=TRES_STD)
    args = parser.parse_args()

    tres_mean_val = args.tres_mean
    tres_std_val = args.tres_std
    signal_thr = args.threshold

    out_dir = Path(args.output_dir)
    print(f"Config: tres_mean={tres_mean_val}, tres_std={tres_std_val}, threshold={signal_thr}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load norm params
    mc_mean, mc_std = load_norm_params(args.mc_data)
    nsig_data_path = "/home/plotnikovgp/baikal/Baikal-ML/pytorch_training/data/baikal_2020_sig-noise_mid-eq_normed.h5"
    nsig_mean, nsig_std = load_norm_params(nsig_data_path)
    exp_mean, exp_std = load_norm_params(args.exp_data)

    print(f"MC norm: mean={mc_mean}, std={mc_std}")
    print(f"NSig norm: mean={nsig_mean}, std={nsig_std}")
    print(f"EXP norm: mean={exp_mean}, std={exp_std}")

    # Load models
    print(f"Loading noise/signal model from {args.nsig_ckpt}...")
    nsig_model = load_model(args.nsig_ckpt, out_size=2)
    print(f"Loading tres model from {args.tres_ckpt}...")
    tres_model = load_model(args.tres_ckpt, out_size=1)

    # ─── MC data ───────────────────────────────────────────────────────────
    print(f"\nLoading MC events (val, max {args.mc_max_events})...")
    mc_events = load_mc_events(args.mc_data, "val", args.mc_max_events)
    print(f"  Loaded {len(mc_events)} MC events")

    # Classify MC hits (renorm from MC→NSig)
    print("Classifying MC hits with noise/signal model...")

    def mc_to_nsig(data):
        return renormalize(data, mc_mean, mc_std, nsig_mean, nsig_std)

    mc_probs = classify_hits(nsig_model, mc_events, mc_to_nsig)

    # Panel 1: MC with network-predicted signal (2-8 cut)
    print("Evaluating tres on MC signal hits (network prediction, 2-8 cut)...")
    mc_net_signal_hits = []
    mc_net_true_tres = []
    mc_net_n_pass = 0

    for ev, probs in zip(mc_events, mc_probs):
        passes, sig_mask = apply_2_8_cut(probs, ev["channels"], threshold=signal_thr)
        if not passes:
            continue
        mc_net_n_pass += 1
        mc_net_signal_hits.append(ev["data"][sig_mask])
        mc_net_true_tres.append(ev["t_res"][sig_mask])

    print(f"  Events passing 2-8 cut (network): {mc_net_n_pass}/{len(mc_events)}")
    print(f"  Total signal hits: {sum(len(h) for h in mc_net_signal_hits):,}")

    mc_net_pred = predict_tres(tres_model, mc_net_signal_hits, tres_mean_val, tres_std_val)
    mc_net_pred_all = np.concatenate(mc_net_pred) if mc_net_pred else np.array([])
    mc_net_true_all = np.concatenate(mc_net_true_tres) if mc_net_true_tres else np.array([])

    # Panel 2: MC with GT labels (label != 0), same events (that pass 2-8 cut by network)
    print("Evaluating tres on MC signal hits (GT label != 0, same events)...")
    mc_gt_signal_hits = []
    mc_gt_true_tres = []
    mc_gt_n_events = 0

    for ev, probs in zip(mc_events, mc_probs):
        passes, _ = apply_2_8_cut(probs, ev["channels"], threshold=signal_thr)
        if not passes:
            continue
        gt_mask = ev["labels"] != 0
        if gt_mask.sum() < MIN_SIGNAL_HITS:
            continue
        mc_gt_n_events += 1
        mc_gt_signal_hits.append(ev["data"][gt_mask])
        mc_gt_true_tres.append(ev["t_res"][gt_mask])

    print(f"  Events with GT signal (from 2-8 cut passing): {mc_gt_n_events}")
    print(f"  Total GT signal hits: {sum(len(h) for h in mc_gt_signal_hits):,}")

    mc_gt_pred = predict_tres(tres_model, mc_gt_signal_hits, tres_mean_val, tres_std_val)
    mc_gt_pred_all = np.concatenate(mc_gt_pred) if mc_gt_pred else np.array([])
    mc_gt_true_all = np.concatenate(mc_gt_true_tres) if mc_gt_true_tres else np.array([])

    # ─── EXP data ──────────────────────────────────────────────────────────
    print(f"\nLoading EXP events (val, max {args.exp_max_events})...")
    exp_events = load_exp_events(args.exp_data, "val", args.exp_max_events)
    print(f"  Loaded {len(exp_events)} EXP events")

    # Classify EXP hits (renorm from EXP→NSig)
    print("Classifying EXP hits with noise/signal model...")

    def exp_to_nsig(data):
        return renormalize(data, exp_mean, exp_std, nsig_mean, nsig_std)

    def exp_to_mc(data):
        return renormalize(data, exp_mean, exp_std, mc_mean, mc_std)

    exp_probs = classify_hits(nsig_model, exp_events, exp_to_nsig)

    # Panel 3: EXP with network-predicted signal (2-8 cut)
    print("Evaluating tres on EXP signal hits (network prediction, 2-8 cut)...")
    exp_signal_hits = []
    exp_n_pass = 0

    for ev, probs in zip(exp_events, exp_probs):
        passes, sig_mask = apply_2_8_cut(probs, ev["channels"], threshold=signal_thr)
        if not passes:
            continue
        exp_n_pass += 1
        exp_signal_hits.append(exp_to_mc(ev["data"][sig_mask]))

    print(f"  Events passing 2-8 cut (network): {exp_n_pass}/{len(exp_events)}")
    print(f"  Total signal hits: {sum(len(h) for h in exp_signal_hits):,}")

    exp_pred = predict_tres(tres_model, exp_signal_hits, tres_mean_val, tres_std_val)
    exp_pred_all = np.concatenate(exp_pred) if exp_pred else np.array([])

    del nsig_model, tres_model
    torch.cuda.empty_cache()

    # ─── Plotting ──────────────────────────────────────────────────────────
    print("\nGenerating plots...")

    fig = plt.figure(figsize=(24, 18), constrained_layout=True)
    gs = fig.add_gridspec(3, 6)

    # Row 1: MC network-predicted signal
    row1_axes = [fig.add_subplot(gs[0, i]) for i in range(6)]
    if len(mc_net_pred_all) > 0:
        plot_tres_eval_panel(
            row1_axes,
            mc_net_pred_all,
            mc_net_true_all,
            "MC: signal by network (2-8 cut)",
            has_gt=True,
        )

    # Row 2: MC GT labels
    row2_axes = [fig.add_subplot(gs[1, i]) for i in range(6)]
    if len(mc_gt_pred_all) > 0:
        plot_tres_eval_panel(
            row2_axes, mc_gt_pred_all, mc_gt_true_all, "MC: signal by GT (label != 0)", has_gt=True
        )

    # Row 3: EXP network-predicted signal (3 panels, spanning 2 cols each)
    row3_axes = [
        fig.add_subplot(gs[2, 0:2]),
        fig.add_subplot(gs[2, 2:4]),
        fig.add_subplot(gs[2, 4:6]),
    ]
    if len(exp_pred_all) > 0:
        plot_tres_eval_panel(
            row3_axes, exp_pred_all, None, "EXP: signal by network (2-8 cut)", has_gt=False
        )

    # Row labels
    for i, label in enumerate(
        [
            f"MC: Network signal (2-8 cut)\n{mc_net_n_pass} events, {len(mc_net_pred_all):,} hits",
            f"MC: GT signal (label≠0)\n{mc_gt_n_events} events, {len(mc_gt_pred_all):,} hits",
            f"EXP: Network signal (2-8 cut)\n{exp_n_pass} events, {len(exp_pred_all):,} hits",
        ]
    ):
        y = 1.0 - (i + 0.5) / 3.0
        fig.text(
            0.005, y, label, rotation=90, va="center", ha="center", fontsize=11, fontweight="bold"
        )

    fig.suptitle(
        f"|t_res| Prediction — tres model with noise/sig 2-8 cut "
        f"(thr={signal_thr})\n"
        f"tres: {args.tres_ckpt.split('/')[-2]}, "
        f"nsig: {args.nsig_ckpt.split('/')[-2]}",
        fontsize=13,
        fontweight="bold",
    )

    out_path = out_dir / "tres_eval_nsig_cut.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_path}")

    # Also plot MC comparison: network vs GT overlay
    if len(mc_net_pred_all) > 0 and len(mc_gt_pred_all) > 0:
        fig2, axes2 = plt.subplots(1, 4, figsize=(24, 5), constrained_layout=True)

        # Predicted distribution comparison
        bins_d = np.linspace(0, 50, 80)
        axes2[0].hist(
            mc_net_pred_all,
            bins=bins_d,
            density=True,
            histtype="step",
            lw=2,
            color="#1f77b4",
            label=f"Network signal ({len(mc_net_pred_all):,})",
        )
        axes2[0].hist(
            mc_gt_pred_all,
            bins=bins_d,
            density=True,
            histtype="step",
            lw=2,
            color="#2ca02c",
            label=f"GT signal ({len(mc_gt_pred_all):,})",
        )
        if len(exp_pred_all) > 0:
            axes2[0].hist(
                exp_pred_all,
                bins=bins_d,
                density=True,
                histtype="step",
                lw=2,
                color="#d62728",
                ls="--",
                label=f"EXP network ({len(exp_pred_all):,})",
            )
        axes2[0].set_xlabel("Predicted |t_res| (ns)")
        axes2[0].set_ylabel("Density")
        axes2[0].set_yscale("log")
        axes2[0].set_title("Predicted |t_res| distribution comparison")
        axes2[0].legend()
        axes2[0].grid(True, alpha=0.3)

        # Error comparison
        mc_net_err = mc_net_pred_all - np.abs(mc_net_true_all)
        mc_gt_err = mc_gt_pred_all - np.abs(mc_gt_true_all)
        bins_e = np.linspace(-30, 30, 80)
        axes2[1].hist(
            mc_net_err,
            bins=bins_e,
            density=True,
            histtype="step",
            lw=2,
            color="#1f77b4",
            label=f"Network (MAE={np.mean(np.abs(mc_net_err)):.2f})",
        )
        axes2[1].hist(
            mc_gt_err,
            bins=bins_e,
            density=True,
            histtype="step",
            lw=2,
            color="#2ca02c",
            label=f"GT (MAE={np.mean(np.abs(mc_gt_err)):.2f})",
        )
        axes2[1].axvline(0, color="red", ls="--", lw=1)
        axes2[1].set_xlabel("Prediction error (ns)")
        axes2[1].set_ylabel("Density")
        axes2[1].set_title("Error distribution: Network vs GT signal")
        axes2[1].legend()
        axes2[1].grid(True, alpha=0.3)

        # MAE vs true |t_res|
        mc_net_true_abs = np.abs(mc_net_true_all)
        mc_gt_true_abs = np.abs(mc_gt_true_all)
        bin_edges_mae = np.linspace(0, 30, 16)
        centers_mae = 0.5 * (bin_edges_mae[:-1] + bin_edges_mae[1:])

        net_maes, gt_maes = [], []
        for b0, b1 in zip(bin_edges_mae[:-1], bin_edges_mae[1:]):
            sel_net = (mc_net_true_abs >= b0) & (mc_net_true_abs < b1)
            sel_gt = (mc_gt_true_abs >= b0) & (mc_gt_true_abs < b1)
            if sel_net.sum() > 10:
                net_maes.append(
                    np.mean(np.abs(mc_net_pred_all[sel_net] - mc_net_true_abs[sel_net]))
                )
            else:
                net_maes.append(np.nan)
            if sel_gt.sum() > 10:
                gt_maes.append(np.mean(np.abs(mc_gt_pred_all[sel_gt] - mc_gt_true_abs[sel_gt])))
            else:
                gt_maes.append(np.nan)

        axes2[2].plot(
            centers_mae, net_maes, "o-", color="#1f77b4", lw=2, ms=5, label="Network signal"
        )
        axes2[2].plot(centers_mae, gt_maes, "s-", color="#2ca02c", lw=2, ms=5, label="GT signal")
        axes2[2].set_xlabel("True |t_res| (ns)")
        axes2[2].set_ylabel("MAE (ns)")
        axes2[2].set_title("MAE vs true |t_res| (< 30 ns)")
        axes2[2].legend()
        axes2[2].grid(True, alpha=0.3)

        # Metrics table
        m_net = compute_metrics(mc_net_pred_all, np.abs(mc_net_true_all))
        m_gt = compute_metrics(mc_gt_pred_all, np.abs(mc_gt_true_all))
        text = (
            f"{'Metric':<12} {'Network':>10} {'GT':>10}\n"
            f"{'─' * 34}\n"
            f"{'MAE':<12} {m_net['MAE']:>9.2f}  {m_gt['MAE']:>9.2f}\n"
            f"{'RMSE':<12} {m_net['RMSE']:>9.2f}  {m_gt['RMSE']:>9.2f}\n"
            f"{'Median AE':<12} {m_net['Median AE']:>9.2f}  {m_gt['Median AE']:>9.2f}\n"
            f"{'Q68':<12} {m_net['Q68']:>9.2f}  {m_gt['Q68']:>9.2f}\n"
            f"{'Q90':<12} {m_net['Q90']:>9.2f}  {m_gt['Q90']:>9.2f}\n"
            f"{'Bias':<12} {m_net['Bias']:>9.2f}  {m_gt['Bias']:>9.2f}\n"
            f"{'N_hits':<12} {m_net['N_hits']:>9,}  {m_gt['N_hits']:>9,}"
        )
        axes2[3].text(
            0.5,
            0.5,
            text,
            transform=axes2[3].transAxes,
            ha="center",
            va="center",
            fontsize=10,
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", edgecolor="orange"),
        )
        axes2[3].set_title("Network vs GT metrics")
        axes2[3].axis("off")

        fig2.suptitle(
            "tres prediction comparison — Network signal vs GT signal (MC)",
            fontsize=13,
            fontweight="bold",
        )
        out_path2 = out_dir / "tres_eval_comparison.png"
        fig2.savefig(out_path2, dpi=150, bbox_inches="tight")
        plt.close(fig2)
        print(f"Saved: {out_path2}")

    print("\nDone!")


if __name__ == "__main__":
    main()
