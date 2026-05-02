"""Generate comprehensive PDF report: MC val vs EXP distributions + PR curves.

Model: noise_sig_rerun_v2/encoder_nl5_nh1_dff128_hs128_bs128/best_2020.ckpt
Features: [charge, time, x, y, z] (indices 0-4).
Signal definition for PR: |t_res| < 10 ns.
"""

import argparse
import datetime
import hashlib
import json
import os
import platform
import re
import shlex
import subprocess
import sys
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import torch
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.encoder import Encoder, EncoderDomainAdaptation

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MC_DEFAULT = "/home/plotnikovgp/baikal/Baikal-ML/pytorch_training/data/baikal_2020_sig-noise_mid-eq_normed.h5"
EXP_DATA = "/home/plotnikovgp/baikal/Baikal-ML/pytorch_training/data/baikal_exp_filtered_norm_w_model_enc_v2_fix_q.h5"
CKPT = "checkpoints/noise_sig_rerun_v2/encoder_nl5_nh1_dff128_hs128_bs128/best_2020.ckpt"
NO_ZM_CKPT = "checkpoints/noise_sig_experiments/k_nsol_labelneq0_hs128/best.ckpt"
ZM_CKPT = "checkpoints/noise_sig_experiments/k_nsol_labelneq0_hs128_zm/best.ckpt"
DA_CKPT = "checkpoints/noise_sig_experiments/k_nsol_labelneq0_da_hs128/best_mc_2020.ckpt"
TRES_CKPT = "checkpoints/noise_sig_experiments/k_nsol_labelneq0_tres_hs128/best.ckpt"

FEAT_CHARGE = 0
FEAT_Z = 4

MC_COL = "#2ca02c"
EXP_COL = "#1f77b4"
EVTYPE_COLS = {"muatm": "#d62728", "nuatm": "#ff7f0e", "nue2": "#9467bd"}
EVTYPE_LABELS = {
    "muatm": "EAS",
    "nuatm": "atmospheric &#957;<sub>&#956;</sub>",
    "nue2": "cosmogenic &#957;<sub>&#956;</sub>",
}
EVTYPE_LABELS_MPL = {
    "muatm": "EAS",
    "nuatm": r"atmospheric $\nu_\mu$",
    "nue2": r"cosmogenic $\nu_\mu$",
}
PRECISION_COL = "#D55E00"
RECALL_COL = "#009E73"
PRECISION_BAND = "rgba(213, 94, 0, 0.18)"
RECALL_BAND = "rgba(0, 158, 115, 0.18)"
PLOTLY_TEMPLATE = "simple_white"
DEFAULT_THRESHOLD_POINTS = 80
METRIC_Y_MIN = 0.7
METRIC_Y_TICK = 0.05
METRIC_Y_MINOR = 0.01
THRESHOLD_DTICK = 0.2
THRESHOLD_MINOR = 0.05
XI_HTML = "<i>&#958;</i>"
XI_MPL = r"$\xi$"

PLOTLY_FONT = dict(family="Arial, Helvetica, sans-serif", size=16, color="#1f1f1f")
PLOTLY_AXIS_TITLE_FONT = dict(size=20)
PLOTLY_TICK_FONT = dict(size=18)
PLOTLY_LEGEND_FONT = dict(size=14)
PLOTLY_SUBPLOT_TITLE_FONT = dict(size=18)

P_LS, R_LS = "-", "--"

plt.rcParams.update(
    {
        "font.size": 16,
        "axes.titlesize": 18,
        "axes.labelsize": 20,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 14,
        "figure.titlesize": 18,
        "grid.alpha": 0.3,
        "grid.linestyle": ":",
    }
)


def load_norm_params(h5_path):
    with h5py.File(h5_path, "r") as f:
        return (
            f["norm_param/mean"][:].astype(np.float32),
            f["norm_param/std"][:].astype(np.float32),
        )


def renormalize(data, src_mean, src_std, dst_mean, dst_std):
    return (data * src_std + src_mean - dst_mean) / dst_std


def denormalize(data, mean, std):
    return data * std + mean


def load_mc_events(
    h5_path, split, max_per_type, batch_size=128, load_labels=False, event_types=None
):
    """Load MC events with t_res (and optionally labels), sampling each event type."""
    src_mean, src_std = load_norm_params(h5_path)
    with h5py.File(h5_path, "r") as f:
        ev_starts = f[f"{split}/ev_starts/data"][:]
        data_all = f[f"{split}/data/data"]
        channels_all = f[f"{split}/channels/data"]
        t_res_all = f[f"{split}/t_res/data"]
        ev_ids = f[f"{split}/ev_ids/data"][:]
        prime_all = f[f"{split}/prime_prty/data"] if f"{split}/prime_prty/data" in f else None
        labels_all = f[f"{split}/labels/data"] if load_labels and f"{split}/labels" in f else None
        n_total = len(ev_starts) - 1

        type_indices = {}
        for i in range(n_total):
            t = ev_ids[i].decode().split("_")[0]
            type_indices.setdefault(t, []).append(i)

        indices = []
        for t, idxs in type_indices.items():
            if event_types is not None and t not in event_types:
                continue
            sel = idxs[:max_per_type]
            indices.extend(sel)
            print(f"    {t}: {len(sel)} events (of {len(idxs)} available)")
        indices.sort()

        for bs_start in tqdm(
            range(0, len(indices), batch_size), desc=f"Loading MC {split} batches"
        ):
            batch_idx = indices[bs_start : bs_start + batch_size]
            evs_norm, evs_raw, evs_ch, evs_tres, evs_type, evs_lab, evs_energy = (
                [],
                [],
                [],
                [],
                [],
                [],
                [],
            )
            for idx in batch_idx:
                s, e = int(ev_starts[idx]), int(ev_starts[idx + 1])
                hits = data_all[s:e].astype(np.float32)
                raw = denormalize(hits, src_mean, src_std)
                evs_norm.append(hits)
                evs_raw.append(raw)
                evs_ch.append(channels_all[s:e])
                evs_tres.append(t_res_all[s:e])
                evs_type.append(ev_ids[idx].decode().split("_")[0])
                evs_energy.append(float(prime_all[idx, 2]) if prime_all is not None else np.nan)
                if labels_all is not None:
                    evs_lab.append(labels_all[s:e])

            max_len = max(len(ev) for ev in evs_norm)
            bs = len(evs_norm)
            x_n = np.zeros((bs, max_len, 5), dtype=np.float32)
            x_r = np.zeros((bs, max_len, 5), dtype=np.float32)
            mask = np.zeros((bs, max_len), dtype=np.float32)
            ch = np.zeros((bs, max_len), dtype=np.int32)
            tr = np.zeros((bs, max_len), dtype=np.float32)
            lab = np.zeros((bs, max_len), dtype=np.float32) if labels_all is not None else None
            for i in range(bs):
                L = len(evs_norm[i])
                x_n[i, :L] = evs_norm[i]
                x_r[i, :L] = evs_raw[i]
                mask[i, :L] = 1.0
                ch[i, :L] = evs_ch[i]
                tr[i, :L] = evs_tres[i]
                if lab is not None:
                    lab[i, :L] = evs_lab[i]

            yield (
                torch.tensor(x_n),
                torch.tensor(x_r),
                torch.tensor(mask),
                ch,
                tr,
                evs_type,
                lab,
                evs_energy,
            )


def load_exp_events(h5_path, split, max_events, renorm_fn, batch_size=128):
    """Load EXP events (no labels)."""
    src_mean, src_std = load_norm_params(h5_path)
    with h5py.File(h5_path, "r") as f:
        ev_starts = f[f"{split}/ev_starts/data"][:]
        data_all = f[f"{split}/data/data"]
        channels_all = f[f"{split}/channels/data"]
        n = min(max_events, len(ev_starts) - 1) if max_events else len(ev_starts) - 1

        for bs_start in tqdm(range(0, n, batch_size), desc=f"Loading EXP {split} batches"):
            bs_end = min(bs_start + batch_size, n)
            evs_norm, evs_raw, evs_ch = [], [], []
            for idx in range(bs_start, bs_end):
                s, e = int(ev_starts[idx]), int(ev_starts[idx + 1])
                hits = data_all[s:e].astype(np.float32)
                raw = denormalize(hits, src_mean, src_std)
                model_input = renorm_fn(hits)
                evs_norm.append(model_input)
                evs_raw.append(raw)
                evs_ch.append(channels_all[s:e])

            max_len = max(len(ev) for ev in evs_norm)
            bs = len(evs_norm)
            x_n = np.zeros((bs, max_len, 5), dtype=np.float32)
            x_r = np.zeros((bs, max_len, 5), dtype=np.float32)
            mask = np.zeros((bs, max_len), dtype=np.float32)
            ch = np.zeros((bs, max_len), dtype=np.int32)
            for i in range(bs):
                L = len(evs_norm[i])
                x_n[i, :L] = evs_norm[i]
                x_r[i, :L] = evs_raw[i]
                mask[i, :L] = 1.0
                ch[i, :L] = evs_ch[i]

            yield torch.tensor(x_n), torch.tensor(x_r), torch.tensor(mask), ch


def load_model(ckpt_path, hs=128, dff=512, model_type="encoder", da_kwargs=None, out_size=2):
    if model_type == "encoder_da":
        da_kw = da_kwargs or {}
        model = EncoderDomainAdaptation(
            in_features=5,
            hidden_size=hs,
            num_layers=5,
            dim_feedforward_size=dff,
            n_heads=1,
            out_size=out_size,
            dropout_p=0.0,
            num_domains=da_kw.get("num_domains", 2),
            domain_classifier_hidden_size=da_kw.get("domain_classifier_hidden_size", 64),
            domain_classifier_layers=da_kw.get("domain_classifier_layers", 2),
            gradient_reversal_alpha=da_kw.get("gradient_reversal_alpha", 1.0),
            aggregate_output=False,
        )
    else:
        model = Encoder(
            in_features=5,
            hidden_size=hs,
            num_layers=5,
            dim_feedforward_size=dff,
            n_heads=1,
            out_size=out_size,
            dropout_p=0.0,
        )
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(sd)
    return model.to(DEVICE).eval()


def _get_output(model, x, mask):
    out = model(x, mask)
    if isinstance(out, tuple):
        out = out[0]
    return out


def _classification_probs(output):
    return torch.sigmoid(output[:, :, 1])


def _get_output_and_hidden(model, x, mask):
    if isinstance(model, EncoderDomainAdaptation):
        out, _, hidden = model(x, mask, return_hidden_states=True)
        return out, hidden
    old_return_hidden = getattr(model, "return_hidden", False)
    model.return_hidden = True
    out, hidden = model(x, mask)
    model.return_hidden = old_return_hidden
    return out, hidden


def infer_mc(model, data_iter, signal_definition="tres"):
    events = []
    with torch.no_grad():
        for x_n, x_r, mask, ch, tr, ev_types, lab, energies in tqdm(data_iter, desc="MC inference"):
            out = _get_output(model, x_n.to(DEVICE), mask.to(DEVICE).bool())
            probs = _classification_probs(out).cpu().numpy()
            m_np = mask.numpy().astype(bool)
            r_np = x_r.numpy()
            for i in range(x_n.shape[0]):
                m = m_np[i]
                tres_i = tr[i][m]
                true_sig = np.abs(tres_i) < 10
                if signal_definition == "tres_or_labels" and lab is not None:
                    true_sig = true_sig | (lab[i][m] != 0)
                elif signal_definition == "label_nonzero" and lab is not None:
                    true_sig = lab[i][m] != 0
                events.append(
                    {
                        "probs": probs[i][m],
                        "charge": r_np[i, m, FEAT_CHARGE],
                        "z": r_np[i, m, FEAT_Z],
                        "channels": ch[i][m],
                        "t_res": tres_i,
                        "ev_type": ev_types[i],
                        "energy": energies[i],
                        "true_sig": true_sig,
                    }
                )
    return events


def infer_exp(model, data_iter):
    events = []
    with torch.no_grad():
        for x_n, x_r, mask, ch in tqdm(data_iter, desc="EXP inference"):
            out = _get_output(model, x_n.to(DEVICE), mask.to(DEVICE).bool())
            probs = _classification_probs(out).cpu().numpy()
            m_np = mask.numpy().astype(bool)
            r_np = x_r.numpy()
            for i in range(x_n.shape[0]):
                m = m_np[i]
                events.append(
                    {
                        "probs": probs[i][m],
                        "charge": r_np[i, m, FEAT_CHARGE],
                        "z": r_np[i, m, FEAT_Z],
                        "channels": ch[i][m],
                    }
                )
    return events


def get_sig_data(events, threshold, min_hits=0, min_strings=0):
    """Apply threshold + cuts, return per-event signal info."""
    passed = []
    for ev in events:
        sig = ev["probs"] > threshold
        n_sig = sig.sum()
        n_str = len(np.unique(ev["channels"][sig] // 36)) if n_sig > 0 else 0
        if n_sig >= min_hits and n_str >= min_strings:
            passed.append(
                {
                    "all_probs": ev["probs"],
                    "sig_charge": ev["charge"][sig],
                    "sig_z": ev["z"][sig],
                    "n_sig": int(n_sig),
                    "q_sum": float(ev["charge"][sig].sum()) if n_sig > 0 else 0.0,
                }
            )
    return passed


def event_passes_cut(ev, threshold, min_hits=0, min_strings=0):
    sig = ev["probs"] > threshold
    n_sig = int(sig.sum())
    n_str = len(np.unique(ev["channels"][sig] // 36)) if n_sig > 0 else 0
    return n_sig >= min_hits and n_str >= min_strings


def event_truly_passes(ev, min_hits=0, min_strings=0):
    sig = ev["true_sig"].astype(bool)
    n_sig = int(sig.sum())
    n_str = len(np.unique(ev["channels"][sig] // 36)) if n_sig > 0 else 0
    return n_sig >= min_hits and n_str >= min_strings


def _binom_err(p, n):
    if n <= 0 or not np.isfinite(p):
        return np.nan
    return float(np.sqrt(max(p * (1.0 - p), 0.0) / n))


def _event_types(events):
    return sorted({e["ev_type"] for e in events})


def _metric_from_counts(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    return precision, recall


def _compute_micro_pr(evs, thresholds):
    if not evs:
        return np.full_like(thresholds, np.nan), np.full_like(thresholds, np.nan)
    all_probs = np.concatenate([e["probs"] for e in evs])
    all_true = np.concatenate([e["true_sig"].astype(bool) for e in evs])
    prec_vals, rec_vals = [], []
    for t in thresholds:
        pred = all_probs > t
        tp = int((pred & all_true).sum())
        fp = int((pred & ~all_true).sum())
        fn = int((~pred & all_true).sum())
        p, r = _metric_from_counts(tp, fp, fn)
        prec_vals.append(p)
        rec_vals.append(r)
    return np.asarray(prec_vals), np.asarray(rec_vals)


def _compute_macro_event_pr(evs, thresholds):
    if not evs:
        return np.full_like(thresholds, np.nan), np.full_like(thresholds, np.nan)

    precision_sum = np.zeros_like(thresholds, dtype=np.float64)
    recall_sum = np.zeros_like(thresholds, dtype=np.float64)
    precision_count = np.zeros_like(thresholds, dtype=np.int64)
    recall_count = np.zeros_like(thresholds, dtype=np.int64)

    for ev in evs:
        probs = ev["probs"]
        true = ev["true_sig"].astype(bool)
        pred = probs[:, None] > thresholds[None, :]
        true_2d = true[:, None]
        tp = (pred & true_2d).sum(axis=0)
        fp = (pred & ~true_2d).sum(axis=0)
        fn = (~pred & true_2d).sum(axis=0)

        precision_den = tp + fp
        recall_den = tp + fn
        precision_mask = precision_den > 0
        recall_mask = recall_den > 0

        precision_sum[precision_mask] += tp[precision_mask] / precision_den[precision_mask]
        recall_sum[recall_mask] += tp[recall_mask] / recall_den[recall_mask]
        precision_count[precision_mask] += 1
        recall_count[recall_mask] += 1

    precision = np.full_like(thresholds, np.nan, dtype=np.float64)
    recall = np.full_like(thresholds, np.nan, dtype=np.float64)
    np.divide(precision_sum, precision_count, out=precision, where=precision_count > 0)
    np.divide(recall_sum, recall_count, out=recall, where=recall_count > 0)
    return precision, recall


def add_section_page(pdf, title, subtitle=""):
    fig = plt.figure(figsize=(16, 2))
    fig.text(0.5, 0.65, title, ha="center", va="center", fontsize=22, fontweight="bold")
    if subtitle:
        fig.text(0.5, 0.25, subtitle, ha="center", va="center", fontsize=14, color="gray")
    plt.axis("off")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ── PLOT FUNCTIONS ──


def _balance(mc_list, ex_list, seed=0):
    """Truncate both lists to the minimum length for a fair MC/EXP comparison."""
    n = min(len(mc_list), len(ex_list))
    rng = np.random.default_rng(seed)
    mc_idx = rng.choice(len(mc_list), n, replace=False) if len(mc_list) > n else np.arange(n)
    ex_idx = rng.choice(len(ex_list), n, replace=False) if len(ex_list) > n else np.arange(n)
    return [mc_list[i] for i in mc_idx], [ex_list[i] for i in ex_idx]


def plot_score_distributions(pdf, mc_muon, exp_evs, cuts_list, threshold):
    """P(signal) score distributions: MC muon vs EXP (log y)."""
    n_cuts = len(cuts_list)
    ncols = min(n_cuts, 2)
    nrows = (n_cuts + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 6 * nrows), squeeze=False)
    axes = axes.flatten()
    bins = np.linspace(0, 1, 80)

    for i, (mh, ms) in enumerate(cuts_list):
        ax = axes[i]
        cl = f"≥{mh}h ≥{ms}s" if (mh or ms) else "no cuts"
        mc_p = get_sig_data(mc_muon, threshold, mh, ms)
        ex_p = get_sig_data(exp_evs, threshold, mh, ms)
        if not mc_p or not ex_p:
            ax.set_title(f"{cl}: insufficient events")
            continue
        mc_p, ex_p = _balance(mc_p, ex_p)
        mc_all = np.concatenate([e["all_probs"] for e in mc_p])
        ex_all = np.concatenate([e["all_probs"] for e in ex_p])
        ax.hist(
            mc_all,
            bins=bins,
            density=True,
            histtype="step",
            lw=2,
            color=MC_COL,
            label="MC muon",
        )
        ax.hist(
            ex_all,
            bins=bins,
            density=True,
            histtype="step",
            lw=2,
            color=EXP_COL,
            label="EXP",
        )
        ax.set_yscale("log")
        ax.set_xlabel("P(signal)")
        ax.set_ylabel("Density")
        ax.set_title(f"{cl}")
        ax.legend()
        ax.grid(True)

    for j in range(n_cuts, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Score distribution: MC muon vs EXP", fontweight="bold")
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _plot_pr_panels(pdf, mc_events, ylim, suptitle, macro_event=False, cuts=(0, 0)):
    thresholds = np.linspace(0.01, 0.99, 200)
    ev_types = ["muatm", "nuatm", "nue2"]
    min_hits, min_strings = cuts

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    for ax, evtype in zip(axes, ev_types):
        evs = [e for e in mc_events if e["ev_type"] == evtype]
        if min_hits or min_strings:
            evs = [e for e in evs if event_passes_cut(e, 0.5, min_hits, min_strings)]
        if not evs:
            ax.set_title(f"{evtype}: no events")
            continue
        if macro_event:
            prec_vals, rec_vals = _compute_macro_event_pr(evs, thresholds)
        else:
            prec_vals, rec_vals = _compute_micro_pr(evs, thresholds)
        color = EVTYPE_COLS[evtype]

        ax.plot(thresholds, prec_vals, lw=2.2, ls=P_LS, color=color, label="Precision")
        ax.plot(thresholds, rec_vals, lw=2.2, ls=R_LS, color=color, label="Recall")

        idx = int(np.argmin(np.abs(thresholds - 0.5)))
        ax.axvline(0.5, color="k", ls=":", lw=1.0, alpha=0.5)
        ax.annotate(
            f"@0.5: P={prec_vals[idx]:.3f}  R={rec_vals[idx]:.3f}",
            xy=(0.5, ylim[0] + 0.02 * (ylim[1] - ylim[0])),
            xytext=(0.02, 0.05),
            textcoords="axes fraction",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, alpha=0.85),
        )

        ax.set_xlabel("Threshold")
        ax.set_ylabel("Precision / Recall")
        cut_desc = f", cut ≥{min_hits}h ≥{min_strings}s" if (min_hits or min_strings) else ""
        ax.set_title(f"{evtype} ({len(evs)} events{cut_desc})")
        ax.set_xlim(0, 1)
        ax.set_ylim(*ylim)
        ax.legend(loc="lower center")
        ax.grid(True)

    fig.suptitle(suptitle, fontweight="bold")
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def plot_pr_curves(pdf, mc_events, signal_desc):
    _plot_pr_panels(
        pdf,
        mc_events,
        ylim=(0.0, 1.05),
        suptitle=f"Precision & Recall vs threshold by event type (micro, signal = {signal_desc})",
    )


def plot_pr_zoomed(pdf, mc_events, signal_desc):
    _plot_pr_panels(
        pdf,
        mc_events,
        ylim=(0.8, 1.005),
        suptitle=f"Precision & Recall (zoomed ≥ 0.8) by event type (micro, signal = {signal_desc})",
    )


def plot_macro_event_pr_curves(pdf, mc_events, signal_desc):
    for cuts in [(0, 0), (8, 2)]:
        cut_desc = "all events" if cuts == (0, 0) else "events passing NN cut ≥8 hits / ≥2 strings"
        _plot_pr_panels(
            pdf,
            mc_events,
            ylim=(0.0, 1.05),
            suptitle=(
                "Precision & Recall vs threshold by event type "
                f"(macro over events, {cut_desc}, signal = {signal_desc})"
            ),
            macro_event=True,
            cuts=cuts,
        )


def plot_micro_pr_after_event_cut(pdf, mc_events, signal_desc):
    _plot_pr_panels(
        pdf,
        mc_events,
        ylim=(0.0, 1.05),
        suptitle=(
            "Precision & Recall after event cut ≥8 predicted hits / ≥2 strings "
            f"(micro over hits, signal = {signal_desc})"
        ),
        macro_event=False,
        cuts=(8, 2),
    )


def plot_event_selection_fraction(pdf, mc_events, thresholds=(0.3, 0.5, 0.7), cuts=(8, 2)):
    """Fraction of events passing the NN hit/string cut."""
    min_hits, min_strings = cuts
    ev_types = _event_types(mc_events)
    x = np.arange(len(ev_types))
    width = 0.8 / len(thresholds)
    fig, ax = plt.subplots(figsize=(11, 6))
    for i, threshold in enumerate(thresholds):
        vals = []
        for evtype in ev_types:
            evs = [e for e in mc_events if e["ev_type"] == evtype]
            passed = sum(event_passes_cut(e, threshold, min_hits, min_strings) for e in evs)
            vals.append(passed / len(evs) if evs else np.nan)
        ax.bar(x + (i - (len(thresholds) - 1) / 2) * width, vals, width, label=f"thr={threshold}")
    ax.set_xticks(x)
    ax.set_xticklabels(ev_types)
    ax.set_ylabel("Selected-event fraction")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Selected-event fraction: ≥{min_hits} predicted hits / ≥{min_strings} strings")
    ax.grid(True, axis="y")
    ax.legend()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def plot_metrics_vs_energy(pdf, mc_events, threshold):
    """Precision/recall binned by primary particle energy from prime_prty[:, 2]."""
    evs = [e for e in mc_events if np.isfinite(e.get("energy", np.nan)) and e["energy"] > 0]
    if not evs:
        add_section_page(
            pdf, "Energy-Binned Metrics", "No prime_prty[:, 2] energy field in this MC file"
        )
        return
    log_energies = np.log10(np.asarray([e["energy"] for e in evs], dtype=np.float64))
    bins = np.linspace(np.nanmin(log_energies), np.nanmax(log_energies), 12)
    ev_types = _event_types(evs)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)
    for ax, evtype in zip(axes, ev_types):
        type_events = [e for e in evs if e["ev_type"] == evtype]
        purities, efficiencies, centers = [], [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            selected = [e for e in type_events if lo <= np.log10(e["energy"]) < hi]
            if not selected:
                purities.append(np.nan)
                efficiencies.append(np.nan)
                centers.append(0.5 * (lo + hi))
                continue
            tp = fp = fn = 0
            for ev in selected:
                pred = ev["probs"] > threshold
                true = ev["true_sig"].astype(bool)
                tp += int((pred & true).sum())
                fp += int((pred & ~true).sum())
                fn += int((~pred & true).sum())
            p, r = _metric_from_counts(tp, fp, fn)
            purities.append(p)
            efficiencies.append(r)
            centers.append(0.5 * (lo + hi))
        ax.plot(centers, purities, marker="o", label="Precision")
        ax.plot(centers, efficiencies, marker="s", label="Recall")
        ax.set_xlabel("log10(E / TeV)")
        ax.set_title(f"{evtype} ({len(type_events)} events)")
        ax.grid(True, which="both")
        ax.legend()
    axes[0].set_ylabel("Metric")
    fig.suptitle(
        f"Precision / recall vs log10 primary energy (@threshold={threshold})", fontweight="bold"
    )
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def plot_tres_binned_metrics(pdf, mc_events, threshold):
    """Precision/recall for nue2 hits in bins of |t_res|."""
    evs = [e for e in mc_events if e["ev_type"] == "nue2"]
    if not evs:
        add_section_page(pdf, "t_res-Binned Metrics", "No nue2 events loaded")
        return
    probs = np.concatenate([e["probs"] for e in evs])
    true = np.concatenate([e["true_sig"].astype(bool) for e in evs])
    tres = np.abs(np.concatenate([e["t_res"] for e in evs]))
    bins = np.array([0, 2, 5, 10, 20, 40, 80, 160, 320], dtype=float)
    labels = [f"{lo:g}-{hi:g}" for lo, hi in zip(bins[:-1], bins[1:])]
    purities, efficiencies = [], []
    pred = probs > threshold
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (tres >= lo) & (tres < hi)
        tp = int((m & pred & true).sum())
        fp = int((m & pred & ~true).sum())
        fn = int((m & ~pred & true).sum())
        p, r = _metric_from_counts(tp, fp, fn)
        purities.append(p)
        efficiencies.append(r)

    x = np.arange(len(labels))
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(x, purities, marker="o", label="Precision")
    ax1.plot(x, efficiencies, marker="s", label="Recall")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=30, ha="right")
    ax1.set_xlabel(r"$|t_{\mathrm{res}}|$ bin, ns")
    ax1.set_ylabel("Metric")
    ax1.set_ylim(0, 1.05)
    ax1.grid(True)
    ax1.legend(loc="upper left")
    fig.suptitle(
        rf"{EVTYPE_LABELS_MPL['nue2']} precision / recall vs $|t_{{\mathrm{{res}}}}|$ (@{XI_MPL} = {threshold})",
        fontweight="bold",
    )
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def infer_scores_for_batches(model, mc_batches, exp_batches):
    mc_events = infer_mc(model, iter(mc_batches), signal_definition="label_nonzero")
    exp_events = infer_exp(model, iter(exp_batches))
    return mc_events, exp_events


def infer_tres_model(model, data_iter, signal_definition="label_nonzero"):
    events = []
    true_tres = []
    pred_tres = []
    with torch.no_grad():
        for x_n, x_r, mask, ch, tr, ev_types, lab, energies in tqdm(
            data_iter, desc="MC inference (cls+tres)"
        ):
            out = _get_output(model, x_n.to(DEVICE), mask.to(DEVICE).bool())
            probs = _classification_probs(out).cpu().numpy()
            tres_out = out[:, :, 2].detach().cpu().numpy()
            m_np = mask.numpy().astype(bool)
            r_np = x_r.numpy()
            for i in range(x_n.shape[0]):
                m = m_np[i]
                tres_i = tr[i][m]
                true_sig = np.abs(tres_i) < 10
                if signal_definition == "tres_or_labels" and lab is not None:
                    true_sig = true_sig | (lab[i][m] != 0)
                elif signal_definition == "label_nonzero" and lab is not None:
                    true_sig = lab[i][m] != 0
                events.append(
                    {
                        "probs": probs[i][m],
                        "charge": r_np[i, m, FEAT_CHARGE],
                        "z": r_np[i, m, FEAT_Z],
                        "channels": ch[i][m],
                        "t_res": tres_i,
                        "ev_type": ev_types[i],
                        "energy": energies[i],
                        "true_sig": true_sig,
                    }
                )
                true_tres.append(np.abs(tres_i[true_sig]))
                pred_tres.append(np.abs(tres_out[i][m][true_sig]))
    true_tres = np.concatenate(true_tres) if true_tres else np.asarray([], dtype=np.float32)
    pred_tres = np.concatenate(pred_tres) if pred_tres else np.asarray([], dtype=np.float32)
    return events, true_tres, pred_tres


def plot_model_score_comparison(pdf, baseline, zmirror, threshold):
    """Compare no-z-mirror and z-mirror model score/selection distributions."""
    for source_name, ev_key in [("MC muon", "mc_muon"), ("EXP", "exp")]:
        base_events = baseline[ev_key]
        zm_events = zmirror[ev_key]
        if not base_events or not zm_events:
            continue
        base_all = np.concatenate([e["probs"] for e in base_events])
        zm_all = np.concatenate([e["probs"] for e in zm_events])
        base_sig = np.asarray([int((e["probs"] > threshold).sum()) for e in base_events])
        zm_sig = np.asarray([int((e["probs"] > threshold).sum()) for e in zm_events])

        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
        axes[0].hist(
            base_all,
            bins=np.linspace(0, 1, 80),
            density=True,
            histtype="step",
            lw=2,
            label="no z-mirror",
        )
        axes[0].hist(
            zm_all,
            bins=np.linspace(0, 1, 80),
            density=True,
            histtype="step",
            lw=2,
            label="z-mirror",
        )
        axes[0].set_yscale("log")
        axes[0].set_xlabel("P(signal)")
        axes[0].set_ylabel("Density")
        axes[0].set_title("Hit score distribution")
        axes[0].grid(True, which="both")
        axes[0].legend()

        max_n = max(1, int(np.percentile(np.concatenate([base_sig, zm_sig]), 99)))
        bins = np.arange(0, max_n + 2) - 0.5
        axes[1].hist(base_sig, bins=bins, density=True, histtype="step", lw=2, label="no z-mirror")
        axes[1].hist(zm_sig, bins=bins, density=True, histtype="step", lw=2, label="z-mirror")
        axes[1].set_xlabel(f"N predicted signal hits/event @ threshold={threshold}")
        axes[1].set_ylabel("Density")
        axes[1].set_title("Event multiplicity")
        axes[1].grid(True)
        axes[1].legend()

        fig.suptitle(
            f"Model comparison: z-mirror vs no z-mirror ({source_name})", fontweight="bold"
        )
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def collect_event_embeddings(model, batches, source_label, max_events, selected_flags=None):
    embeddings = []
    labels = []
    source_event_offset = 0
    with torch.no_grad():
        for batch in tqdm(batches, desc=f"Embeddings: {source_label}"):
            if len(batch) >= 8:
                x_n, _, mask, _, _, ev_types, _, _ = batch
                keep = np.asarray([t == "muatm" for t in ev_types])
                if not np.any(keep):
                    continue
                source_count = int(keep.sum())
                batch_labels = ["MC muatm"] * source_count
            else:
                x_n, _, mask, _ = batch
                keep = np.ones(x_n.shape[0], dtype=bool)
                source_count = int(keep.sum())
                batch_labels = [source_label] * source_count

            if selected_flags is not None:
                selected = np.asarray(
                    selected_flags[source_event_offset : source_event_offset + source_count],
                    dtype=bool,
                )
                source_event_offset += source_count
                if len(selected) < source_count:
                    selected = np.pad(
                        selected, (0, source_count - len(selected)), constant_values=False
                    )
                keep_indices = np.flatnonzero(keep)
                keep[keep_indices[~selected]] = False
                batch_labels = [
                    label for label, keep_label in zip(batch_labels, selected) if keep_label
                ]
                if not np.any(keep):
                    continue
            else:
                source_event_offset += source_count

            x = x_n.to(DEVICE)
            mask_t = mask.to(DEVICE).bool()
            _, hidden = _get_output_and_hidden(model, x, mask_t)
            mask_f = mask_t.float().unsqueeze(-1)
            pooled = (hidden * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0)
            embeddings.append(pooled.cpu().numpy()[keep])
            labels.extend(batch_labels)
            if sum(len(x) for x in embeddings) >= max_events:
                break
    if not embeddings:
        return np.empty((0, 0), dtype=np.float32), []
    emb = np.concatenate(embeddings, axis=0)[:max_events]
    return emb, labels[: len(emb)]


def plot_da_embedding_umap(pdf, no_da_model, da_model, mc_batches, exp_batches, max_events):
    try:
        import umap
    except ImportError:
        add_section_page(pdf, "DA Embedding UMAP", "Skipped: umap-learn is not installed")
        return

    model_items = [("No DA", no_da_model), ("DA", da_model)]
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    colors = {"MC muatm": "#d62728", "EXP": "#1f77b4"}

    for ax, (model_name, model) in zip(axes, model_items):
        mc_emb, mc_lbl = collect_event_embeddings(model, mc_batches, "MC muatm", max_events)
        exp_emb, exp_lbl = collect_event_embeddings(model, exp_batches, "EXP", max_events)
        emb = np.concatenate([mc_emb, exp_emb], axis=0)
        lbl = np.asarray(mc_lbl + exp_lbl)
        emb = StandardScaler().fit_transform(emb)
        reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, metric="euclidean", random_state=42)
        xy = reducer.fit_transform(emb)
        for label in sorted(set(lbl)):
            m = lbl == label
            ax.scatter(xy[m, 0], xy[m, 1], s=5, alpha=0.45, label=label, color=colors.get(label))
        ax.set_title(model_name)
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        ax.legend(markerscale=3)
        ax.grid(True)

    fig.suptitle(
        "Last hidden layer event embeddings: DA vs no-DA (MC muon vs EXP)", fontweight="bold"
    )
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def plot_hit_distributions(pdf, mc_muon, exp_evs, cuts_list, threshold):
    """Charge and Z distributions on signal hits."""
    for mh, ms in cuts_list:
        cl = f"≥{mh}h ≥{ms}s" if (mh or ms) else "no cuts"
        mc_p = get_sig_data(mc_muon, threshold, mh, ms)
        ex_p = get_sig_data(exp_evs, threshold, mh, ms)
        if not mc_p or not ex_p:
            continue
        mc_p, ex_p = _balance(mc_p, ex_p)
        mc_q = np.concatenate([e["sig_charge"] for e in mc_p])
        ex_q = np.concatenate([e["sig_charge"] for e in ex_p])
        mc_z = np.concatenate([e["sig_z"] for e in mc_p])
        ex_z = np.concatenate([e["sig_z"] for e in ex_p])
        mc_lbl = "MC muon"
        ex_lbl = "EXP"

        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

        # Charge (log y)
        ax = axes[0]
        q_max = np.percentile(np.concatenate([mc_q, ex_q]), 99)
        bins_q = np.linspace(0, q_max, 60)
        ax.hist(mc_q, bins=bins_q, density=True, histtype="step", lw=2, color=MC_COL, label=mc_lbl)
        ax.hist(ex_q, bins=bins_q, density=True, histtype="step", lw=2, color=EXP_COL, label=ex_lbl)
        ax.set_yscale("log")
        ax.set_xlabel("Charge")
        ax.set_ylabel("Density (log)")
        ax.set_title("Charge on signal hits")
        ax.legend()
        ax.grid(True, which="both")

        # Z unweighted
        ax = axes[1]
        bins_z = np.linspace(-300, 300, 22)
        ax.hist(
            mc_z, bins=bins_z, density=True, histtype="step", lw=2, color=MC_COL, label="MC muon"
        )
        ax.hist(ex_z, bins=bins_z, density=True, histtype="step", lw=2, color=EXP_COL, label="EXP")
        ax.set_xlabel("Z [m]")
        ax.set_ylabel("Density")
        ax.set_title("Z distribution")
        ax.legend()
        ax.grid(True)

        # Z charge-weighted
        ax = axes[2]
        ax.hist(
            mc_z,
            bins=bins_z,
            weights=mc_q,
            density=True,
            histtype="step",
            lw=2,
            color=MC_COL,
            label="MC muon",
        )
        ax.hist(
            ex_z,
            bins=bins_z,
            weights=ex_q,
            density=True,
            histtype="step",
            lw=2,
            color=EXP_COL,
            label="EXP",
        )
        ax.set_xlabel("Z [m]")
        ax.set_ylabel("Density (charge-weighted)")
        ax.set_title("Z distribution (charge-weighted)")
        ax.legend()
        ax.grid(True)

        fig.suptitle(f"Hit-level distributions — {cl}, threshold={threshold}", fontweight="bold")
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def plot_event_distributions(pdf, mc_muon, exp_evs, cuts_list, threshold):
    """Signal hit multiplicity and charge sum per event."""
    for mh, ms in cuts_list:
        cl = f"≥{mh}h ≥{ms}s" if (mh or ms) else "no cuts"
        mc_p = get_sig_data(mc_muon, threshold, mh, ms)
        ex_p = get_sig_data(exp_evs, threshold, mh, ms)
        if not mc_p or not ex_p:
            continue
        mc_p, ex_p = _balance(mc_p, ex_p)
        mc_nh = np.array([e["n_sig"] for e in mc_p])
        ex_nh = np.array([e["n_sig"] for e in ex_p])
        mc_qs = np.array([e["q_sum"] for e in mc_p])
        ex_qs = np.array([e["q_sum"] for e in ex_p])
        mc_lbl = "MC muon"
        ex_lbl = "EXP"

        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

        # Multiplicity
        ax = axes[0]
        nh_max = int(np.percentile(np.concatenate([mc_nh, ex_nh]), 99))
        bins_nh = np.arange(0, nh_max + 2) - 0.5
        ax.hist(
            mc_nh, bins=bins_nh, density=True, histtype="step", lw=2, color=MC_COL, label=mc_lbl
        )
        ax.hist(
            ex_nh, bins=bins_nh, density=True, histtype="step", lw=2, color=EXP_COL, label=ex_lbl
        )
        ax.set_xlabel("N signal hits per event")
        ax.set_ylabel("Density")
        ax.set_title("Signal hit multiplicity")
        ax.legend()
        ax.grid(True)

        # Charge sum
        ax = axes[1]
        qs_max = np.percentile(np.concatenate([mc_qs, ex_qs]), 99)
        bins_qs = np.linspace(0, qs_max, 60)
        ax.hist(
            mc_qs, bins=bins_qs, density=True, histtype="step", lw=2, color=MC_COL, label=mc_lbl
        )
        ax.hist(
            ex_qs, bins=bins_qs, density=True, histtype="step", lw=2, color=EXP_COL, label=ex_lbl
        )
        ax.set_xlabel("Total charge of signal hits per event")
        ax.set_ylabel("Density")
        ax.set_title("Charge sum per event")
        ax.legend()
        ax.grid(True)

        fig.suptitle(f"Event-level distributions — {cl}, threshold={threshold}", fontweight="bold")
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def _style_plotly(fig, title=None, height=520):
    title_kwargs = dict(text=title, x=0.02, xanchor="left", font=dict(size=20)) if title else None
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=title_kwargs,
        font=PLOTLY_FONT,
        height=height,
        margin=dict(l=85, r=40, t=70 if title else 40, b=75),
        legend=dict(
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#d0d0d0",
            borderwidth=1,
            font=PLOTLY_LEGEND_FONT,
        ),
    )
    fig.update_xaxes(
        showline=True,
        linewidth=1,
        linecolor="#222",
        ticks="outside",
        mirror=True,
        gridcolor="#e8e8e8",
        zeroline=False,
        title_font=PLOTLY_AXIS_TITLE_FONT,
        tickfont=PLOTLY_TICK_FONT,
    )
    fig.update_yaxes(
        showline=True,
        linewidth=1,
        linecolor="#222",
        ticks="outside",
        mirror=True,
        gridcolor="#e8e8e8",
        zeroline=False,
        title_font=PLOTLY_AXIS_TITLE_FONT,
        tickfont=PLOTLY_TICK_FONT,
    )
    if fig.layout.annotations:
        for ann in fig.layout.annotations:
            if ann.text and not ann.xref or ann.xref == "paper":
                ann.update(font=PLOTLY_SUBPLOT_TITLE_FONT)
    return fig


def _set_metric_yaxis(fig, row=None, col=None, title_text=None, y_min=METRIC_Y_MIN):
    fig.update_yaxes(
        title_text=title_text,
        range=[y_min, 1.01],
        tick0=y_min,
        dtick=METRIC_Y_TICK,
        showgrid=True,
        minor=dict(
            ticks="outside",
            ticklen=3,
            tickcolor="#222",
            dtick=METRIC_Y_MINOR,
            showgrid=False,
        ),
        row=row,
        col=col,
    )


def _metric_axis_range(values, pad=0.025, floor=0.0):
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return [METRIC_Y_MIN, 1.01]
    lo = max(floor, float(np.nanmin(vals)) - pad)
    hi = min(1.02, float(np.nanmax(vals)) + pad)
    if hi - lo < 0.05:
        mid = 0.5 * (lo + hi)
        lo = max(floor, mid - 0.025)
        hi = min(1.02, mid + 0.025)
    return [lo, hi]


def _hist_step(values, bins, name, color, row=None, col=None):
    counts, edges = np.histogram(values, bins=bins, density=True)
    x = np.repeat(edges, 2)[1:-1]
    y = np.repeat(counts, 2)
    kwargs = dict(x=x, y=y, mode="lines", name=name, line=dict(color=color, width=2.4))
    if row is not None:
        kwargs.update(row=row, col=col)
    return go.Scatter(**{k: v for k, v in kwargs.items() if k not in {"row", "col"}})


def make_score_distribution_fig(mc_muon, exp_events, threshold, cuts=(8, 2)):
    min_hits, min_strings = cuts
    mc_after = [e for e in mc_muon if event_passes_cut(e, threshold, min_hits, min_strings)]
    exp_after = [e for e in exp_events if event_passes_cut(e, threshold, min_hits, min_strings)]

    fig = make_subplots(
        rows=2,
        cols=2,
        horizontal_spacing=0.14,
        vertical_spacing=0.18,
    )
    bins_score = np.linspace(0, 1, 100)

    rows = [
        (1, mc_muon, exp_events, "no cut", 0),
        (2, mc_after, exp_after, "after 8-2", min_hits),
    ]

    for row, mc_row, exp_row, _, _ in rows:
        for events, name, color in [(mc_row, "MC muon", MC_COL), (exp_row, "EXP", EXP_COL)]:
            if not events:
                continue
            scores = np.concatenate([e["probs"] for e in events])
            counts, edges = np.histogram(scores, bins=bins_score, density=True)
            fig.add_trace(
                go.Scatter(
                    x=np.repeat(edges, 2)[1:-1],
                    y=np.repeat(counts, 2),
                    mode="lines",
                    name=name,
                    legendgroup=name,
                    showlegend=row == 1,
                    line=dict(color=color, width=2.6),
                ),
                row=row,
                col=1,
            )

        multiplicities = [
            (
                np.asarray([int((e["probs"] > threshold).sum()) for e in mc_row]),
                "MC muon",
                MC_COL,
            ),
            (
                np.asarray([int((e["probs"] > threshold).sum()) for e in exp_row]),
                "EXP",
                EXP_COL,
            ),
        ]
        non_empty = [vals for vals, _, _ in multiplicities if len(vals) > 0]
        bins_n = None
        if non_empty:
            all_vals = np.concatenate(non_empty)
            n_lo = rows[row - 1][4]
            n_hi = max(n_lo + 1, int(np.percentile(all_vals, 99)))
            bins_n = np.arange(n_lo, n_hi + 2) - 0.5
            for vals, name, color in multiplicities:
                if len(vals) == 0:
                    continue
                counts, edges = np.histogram(vals, bins=bins_n, density=True)
                fig.add_trace(
                    go.Scatter(
                        x=np.repeat(edges, 2)[1:-1],
                        y=np.repeat(counts, 2),
                        mode="lines",
                        name=name,
                        legendgroup=name,
                        line=dict(color=color, width=2.6),
                        showlegend=False,
                    ),
                    row=row,
                    col=2,
                )

        fig.update_yaxes(
            type="log",
            title_text="Density",
            exponentformat="power",
            showexponent="all",
            row=row,
            col=1,
        )
        fig.update_yaxes(
            title_text="Density",
            exponentformat="power",
            showexponent="all",
            row=row,
            col=2,
        )
        fig.update_xaxes(
            title_text=XI_HTML,
            range=[0, 1],
            dtick=THRESHOLD_DTICK,
            tick0=0,
            minor=dict(
                ticks="outside",
                ticklen=3,
                tickcolor="#222",
                dtick=THRESHOLD_MINOR,
                showgrid=False,
            ),
            row=row,
            col=1,
        )
        if bins_n is not None:
            x_max = float(bins_n[-1])
            x_lo = float(bins_n[0])
            x_dtick = max(1, int(np.ceil((x_max - x_lo) / 10)))
            fig.update_xaxes(
                title_text="N",
                range=[x_lo, x_max],
                dtick=x_dtick,
                row=row,
                col=2,
            )
        else:
            fig.update_xaxes(title_text="N", row=row, col=2)

    # Row labels (left of each row, centered vertically)
    for ann_text, ann_y in [("no cut", 0.80), ("after 8-2", 0.205)]:
        fig.add_annotation(
            text=ann_text,
            xref="paper",
            yref="paper",
            x=-0.11,
            y=ann_y,
            xanchor="right",
            yanchor="middle",
            showarrow=False,
            textangle=-90,
            font=dict(size=15),
        )

    return _style_plotly(fig, title=None, height=720)


def make_pr_fig(
    mc_events,
    signal_desc,
    macro_event=False,
    cuts=(0, 0),
    threshold_points=DEFAULT_THRESHOLD_POINTS,
):
    thresholds = np.linspace(0.01, 0.99, threshold_points)
    ev_types = ["muatm", "nuatm", "nue2"]
    min_hits, min_strings = cuts
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=[EVTYPE_LABELS[t] for t in ev_types],
        horizontal_spacing=0.08,
    )
    for col, evtype in enumerate(ev_types, start=1):
        evs = [e for e in mc_events if e["ev_type"] == evtype]
        if min_hits or min_strings:
            evs = [e for e in evs if event_passes_cut(e, 0.5, min_hits, min_strings)]
        if macro_event:
            precision, recall = _compute_macro_event_pr(evs, thresholds)
        else:
            precision, recall = _compute_micro_pr(evs, thresholds)
        fig.add_trace(
            go.Scatter(
                x=thresholds,
                y=precision,
                mode="lines",
                name="precision",
                line=dict(color=PRECISION_COL, width=2.6),
                legendgroup="precision",
                showlegend=col == 1,
            ),
            row=1,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=thresholds,
                y=recall,
                mode="lines",
                name="recall",
                line=dict(color=RECALL_COL, width=2.6, dash="dash"),
                legendgroup="recall",
                showlegend=col == 1,
            ),
            row=1,
            col=col,
        )
        fig.update_xaxes(
            title_text=XI_HTML,
            range=[0, 1],
            dtick=THRESHOLD_DTICK,
            tick0=0,
            minor=dict(
                ticks="outside",
                ticklen=3,
                tickcolor="#222",
                dtick=THRESHOLD_MINOR,
                showgrid=False,
            ),
            row=1,
            col=col,
        )
        _set_metric_yaxis(fig, title_text="Metric" if col == 1 else None, row=1, col=col)
    return _style_plotly(fig, title=None, height=420)


def make_event_pr_fig(
    mc_events,
    threshold_points=DEFAULT_THRESHOLD_POINTS,
    cuts=(8, 2),
):
    """Per-type event-level precision and recall vs threshold ξ.

    True positive event = event whose TRUE signal hits already satisfy the same
    8-2 cut applied to the predicted ones. Bands show ±1σ binomial errors.
    """
    min_hits, min_strings = cuts
    ev_types = ["muatm", "nuatm", "nue2"]
    thresholds = np.linspace(0.01, 0.99, threshold_points)
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=[EVTYPE_LABELS[t] for t in ev_types],
        horizontal_spacing=0.08,
    )
    for col, evtype in enumerate(ev_types, start=1):
        evs = [e for e in mc_events if e["ev_type"] == evtype]
        if not evs:
            continue
        truly_pos = np.asarray(
            [event_truly_passes(e, min_hits, min_strings) for e in evs], dtype=bool
        )
        precisions = np.full_like(thresholds, np.nan, dtype=np.float64)
        recalls = np.full_like(thresholds, np.nan, dtype=np.float64)
        prec_err = np.zeros_like(thresholds, dtype=np.float64)
        rec_err = np.zeros_like(thresholds, dtype=np.float64)
        for i, thr in enumerate(thresholds):
            pred_pos = np.asarray(
                [event_passes_cut(e, thr, min_hits, min_strings) for e in evs], dtype=bool
            )
            tp = int((pred_pos & truly_pos).sum())
            fp = int((pred_pos & ~truly_pos).sum())
            fn = int((~pred_pos & truly_pos).sum())
            p, r = _metric_from_counts(tp, fp, fn)
            precisions[i] = p
            recalls[i] = r
            prec_err[i] = _binom_err(p, tp + fp) or 0.0
            rec_err[i] = _binom_err(r, tp + fn) or 0.0

        # ±1σ bands (drawn first so the lines sit on top)
        for vals, errs, fillcolor, group in [
            (precisions, prec_err, PRECISION_BAND, "precision"),
            (recalls, rec_err, RECALL_BAND, "recall"),
        ]:
            mask = np.isfinite(vals)
            if not np.any(mask):
                continue
            xb = thresholds[mask]
            up = np.clip(vals[mask] + errs[mask], 0.0, 1.0)
            lo = np.clip(vals[mask] - errs[mask], 0.0, 1.0)
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([xb, xb[::-1]]),
                    y=np.concatenate([up, lo[::-1]]),
                    fill="toself",
                    fillcolor=fillcolor,
                    line=dict(color="rgba(0,0,0,0)"),
                    showlegend=False,
                    legendgroup=group,
                    hoverinfo="skip",
                ),
                row=1,
                col=col,
            )

        fig.add_trace(
            go.Scatter(
                x=thresholds,
                y=precisions,
                mode="lines",
                name="precision",
                line=dict(color=PRECISION_COL, width=2.6),
                legendgroup="precision",
                showlegend=col == 1,
            ),
            row=1,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=thresholds,
                y=recalls,
                mode="lines",
                name="recall (event efficiency)",
                line=dict(color=RECALL_COL, width=2.6, dash="dash"),
                legendgroup="recall",
                showlegend=col == 1,
            ),
            row=1,
            col=col,
        )
        fig.update_xaxes(
            title_text=XI_HTML,
            range=[0, 1],
            dtick=THRESHOLD_DTICK,
            tick0=0,
            showgrid=True,
            showline=True,
            mirror=True,
            minor=dict(
                ticks="outside",
                ticklen=3,
                tickcolor="#222",
                dtick=THRESHOLD_MINOR,
                showgrid=False,
            ),
            row=1,
            col=col,
        )
        fig.update_yaxes(
            title_text="Metric" if col == 1 else None,
            range=[0, 1.02],
            dtick=0.2,
            showgrid=True,
            showline=True,
            mirror=True,
            minor=dict(
                ticks="outside",
                ticklen=3,
                tickcolor="#222",
                dtick=0.05,
                showgrid=False,
            ),
            row=1,
            col=col,
        )
    return _style_plotly(fig, title=None, height=420)


def make_energy_metrics_fig(mc_events, threshold, min_events_per_bin=300):
    evs = [e for e in mc_events if np.isfinite(e.get("energy", np.nan)) and e["energy"] > 0]
    if not evs:
        return None
    ev_types = [t for t in ["muatm", "nuatm", "nue2"] if any(e["ev_type"] == t for e in evs)]
    specs = [[{"secondary_y": True} for _ in ev_types]]
    fig = make_subplots(
        rows=1,
        cols=len(ev_types),
        specs=specs,
        subplot_titles=[EVTYPE_LABELS.get(t, t) for t in ev_types],
        horizontal_spacing=0.10,
    )
    metric_values = []
    for col, evtype in enumerate(ev_types, start=1):
        type_events = [e for e in evs if e["ev_type"] == evtype]
        type_energies = np.asarray([e["energy"] for e in type_events], dtype=np.float64)
        type_energies = type_energies[np.isfinite(type_energies) & (type_energies > 0)]
        if len(type_energies) < 2:
            continue
        type_log_energies = np.log10(type_energies)
        bins = np.linspace(np.nanmin(type_log_energies), np.nanmax(type_log_energies), 12)
        centers, precisions, recalls, n_per_bin = [], [], [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            selected = [e for e in type_events if lo <= np.log10(e["energy"]) < hi]
            if not selected:
                continue
            tp = fp = fn = 0
            for ev in selected:
                pred = ev["probs"] > threshold
                true = ev["true_sig"].astype(bool)
                tp += int((pred & true).sum())
                fp += int((pred & ~true).sum())
                fn += int((~pred & true).sum())
            p, r = _metric_from_counts(tp, fp, fn)
            centers.append(0.5 * (lo + hi))
            precisions.append(p)
            recalls.append(r)
            n_per_bin.append(len(selected))
        if not centers:
            continue

        # Trim leading/trailing bins with too few events; keep interior gaps as-is.
        keep_mask = [n > min_events_per_bin for n in n_per_bin]
        if any(keep_mask):
            first = keep_mask.index(True)
            last = len(keep_mask) - 1 - keep_mask[::-1].index(True)
            centers = centers[first : last + 1]
            precisions = precisions[first : last + 1]
            recalls = recalls[first : last + 1]
            n_per_bin = n_per_bin[first : last + 1]
        else:
            print(
                f"  [HTML] Energy plot: {evtype} has no bins with >{min_events_per_bin} events; skipping.",
                flush=True,
            )
            continue

        metric_values.extend(precisions)
        metric_values.extend(recalls)
        fig.add_trace(
            go.Scatter(
                x=centers,
                y=precisions,
                mode="lines+markers",
                name="precision",
                legendgroup="precision",
                showlegend=col == 1,
                line=dict(color=PRECISION_COL, width=2.6),
                marker=dict(size=8),
            ),
            row=1,
            col=col,
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=centers,
                y=recalls,
                mode="lines+markers",
                name="recall",
                legendgroup="recall",
                showlegend=col == 1,
                line=dict(color=RECALL_COL, width=2.6, dash="dash"),
                marker=dict(size=8),
            ),
            row=1,
            col=col,
            secondary_y=False,
        )
        fig.update_xaxes(
            title_text="log<sub>10</sub>(E / TeV)",
            nticks=6,
            row=1,
            col=col,
        )
    y_range = _metric_axis_range(metric_values, pad=0.015, floor=METRIC_Y_MIN)
    for col in range(1, len(ev_types) + 1):
        fig.update_yaxes(
            title_text="Metric" if col == 1 else None,
            range=y_range,
            dtick=METRIC_Y_TICK,
            showgrid=True,
            minor=dict(
                ticks="outside",
                ticklen=3,
                tickcolor="#222",
                dtick=METRIC_Y_MINOR,
                showgrid=False,
            ),
            row=1,
            col=col,
            secondary_y=False,
        )
    return _style_plotly(
        fig,
        title=f"Precision and recall vs log<sub>10</sub> primary energy (@{XI_HTML} = {threshold})",
        height=460,
    )


def make_tres_metrics_fig(mc_events, threshold):
    evs = [e for e in mc_events if e["ev_type"] == "nue2"]
    if not evs:
        return None
    probs = np.concatenate([e["probs"] for e in evs])
    true = np.concatenate([e["true_sig"].astype(bool) for e in evs])
    tres = np.abs(np.concatenate([e["t_res"] for e in evs]))
    bins = np.array([0, 2, 5, 10, 20, 40, 80, 160, 320], dtype=float)
    labels = [f"{lo:g}-{hi:g}" for lo, hi in zip(bins[:-1], bins[1:])]
    pred = probs > threshold
    precisions, recalls = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (tres >= lo) & (tres < hi)
        tp = int((m & pred & true).sum())
        fp = int((m & pred & ~true).sum())
        fn = int((m & ~pred & true).sum())
        p, r = _metric_from_counts(tp, fp, fn)
        precisions.append(p)
        recalls.append(r)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=labels,
            y=precisions,
            mode="lines+markers",
            name="Precision",
            line=dict(color=PRECISION_COL, width=2.6),
            marker=dict(size=9),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=labels,
            y=recalls,
            mode="lines+markers",
            name="Recall",
            line=dict(color=RECALL_COL, width=2.6, dash="dash"),
            marker=dict(size=9),
        )
    )
    fig.update_yaxes(
        title_text="Metric",
        range=[METRIC_Y_MIN, 1.01],
        tick0=METRIC_Y_MIN,
        dtick=METRIC_Y_TICK,
        showgrid=True,
        minor=dict(
            ticks="outside",
            ticklen=3,
            tickcolor="#222",
            dtick=METRIC_Y_MINOR,
            showgrid=False,
        ),
    )
    fig.update_xaxes(title_text="|t<sub>res</sub>| bin, ns")
    return _style_plotly(
        fig,
        title=f"{EVTYPE_LABELS['nue2']} precision and recall vs |t<sub>res</sub>| (@{XI_HTML} = {threshold})",
        height=440,
    )


def make_model_comparison_fig(baseline, zmirror, threshold):
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "MC muon hit score",
            "MC muon signal multiplicity",
            "EXP hit score",
            "EXP signal multiplicity",
        ),
        horizontal_spacing=0.1,
        vertical_spacing=0.14,
    )
    for row, ev_key in enumerate(["mc_muon", "exp"], start=1):
        for events, name, color in [
            (baseline[ev_key], "no z-mirror", "#333333"),
            (zmirror[ev_key], "z-mirror", "#0072B2"),
        ]:
            scores = np.concatenate([e["probs"] for e in events])
            counts, edges = np.histogram(scores, bins=np.linspace(0, 1, 100), density=True)
            fig.add_trace(
                go.Scatter(
                    x=np.repeat(edges, 2)[1:-1],
                    y=np.repeat(counts, 2),
                    mode="lines",
                    name=name,
                    legendgroup=name,
                    showlegend=row == 1,
                    line=dict(color=color, width=2.4),
                ),
                row=row,
                col=1,
            )
            n_sig = np.asarray([(e["probs"] > threshold).sum() for e in events])
            max_n = max(1, int(np.percentile(n_sig, 99)))
            c2, e2 = np.histogram(n_sig, bins=np.arange(0, max_n + 2) - 0.5, density=True)
            fig.add_trace(
                go.Scatter(
                    x=np.repeat(e2, 2)[1:-1],
                    y=np.repeat(c2, 2),
                    mode="lines",
                    name=name,
                    legendgroup=name,
                    showlegend=False,
                    line=dict(color=color, width=2.4),
                ),
                row=row,
                col=2,
            )
    fig.update_yaxes(type="log", row=1, col=1)
    fig.update_yaxes(type="log", row=2, col=1)
    fig.update_xaxes(title_text="P(signal)", col=1)
    fig.update_xaxes(title_text=f"N predicted signal hits/event @ threshold={threshold}", col=2)
    return _style_plotly(fig, "z-mirror vs no-z-mirror model comparison", height=780)


def make_binary_model_comparison_fig(
    cls_events,
    tres_events,
    signal_desc,
    threshold_points=DEFAULT_THRESHOLD_POINTS,
):
    thresholds = np.linspace(0.01, 0.99, threshold_points)
    fig = go.Figure()
    for events, name, color in [
        (cls_events, "classification only", "#333333"),
        (tres_events, "classification + t_res", "#D55E00"),
    ]:
        precision, recall = _compute_macro_event_pr(events, thresholds)
        fig.add_trace(
            go.Scatter(
                x=thresholds,
                y=precision,
                mode="lines",
                name=f"{name} precision",
                legendgroup=f"{name}-precision",
                line=dict(color=color, width=2.4),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=thresholds,
                y=recall,
                mode="lines",
                name=f"{name} recall",
                legendgroup=f"{name}-recall",
                line=dict(color=color, width=2.4, dash="dash"),
            )
        )
    fig.update_xaxes(title_text="Threshold", range=[0, 1])
    _set_metric_yaxis(fig, title_text="Metric")
    return _style_plotly(
        fig,
        "Binary metrics: classification-only vs classification+t_res (macro over events)",
        height=560,
    )


def make_tres_prediction_fig(true_tres, pred_tres):
    finite = np.isfinite(true_tres) & np.isfinite(pred_tres)
    if finite.any():
        true_q = np.nanquantile(true_tres[finite], [0.5, 0.95, 0.995])
        pred_q = np.nanquantile(pred_tres[finite], [0.5, 0.95, 0.995])
        print(
            "[HTML] t_res prediction stats: "
            f"n={int(finite.sum())}, "
            f"true q50/q95/q99.5={true_q[0]:.3g}/{true_q[1]:.3g}/{true_q[2]:.3g}, "
            f"pred q50/q95/q99.5={pred_q[0]:.3g}/{pred_q[1]:.3g}/{pred_q[2]:.3g}",
            flush=True,
        )
    else:
        print("[HTML] t_res prediction stats: no finite points.", flush=True)

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Distribution up to 30 ns", "Distribution up to 100 ns"),
        horizontal_spacing=0.1,
    )
    for col, limit in enumerate([30, 100], start=1):
        bins = np.linspace(0, limit, 100)
        for vals, name, color in [
            (true_tres[finite], "true |t_res|", PRECISION_COL),
            (pred_tres[finite], "predicted |t_res|", RECALL_COL),
        ]:
            vals = vals[(vals >= 0) & (vals <= limit)]
            if len(vals) == 0:
                continue
            counts, edges = np.histogram(vals, bins=bins, density=True)
            fig.add_trace(
                go.Scatter(
                    x=np.repeat(edges, 2)[1:-1],
                    y=np.repeat(counts, 2),
                    mode="lines",
                    name=name,
                    legendgroup=name,
                    showlegend=col == 1,
                    line=dict(color=color, width=2.6),
                ),
                row=1,
                col=col,
            )
        fig.update_xaxes(title_text="|t<sub>res</sub>|, ns", range=[0, limit], row=1, col=col)
        fig.update_yaxes(title_text="Density" if col == 1 else None, type="log", row=1, col=col)
    return _style_plotly(fig, "t_res distributions: true vs predicted", height=560)


def make_da_embedding_umap_fig(
    embedding_models,
    mc_batches,
    exp_batches,
    max_events,
    mc_muon_events=None,
    exp_events=None,
    threshold=0.5,
):
    try:
        import umap
    except ImportError:
        return None
    model_subplot_titles = [label for label, _ in embedding_models]
    subplot_titles = model_subplot_titles + model_subplot_titles
    fig = make_subplots(
        rows=2,
        cols=len(embedding_models),
        subplot_titles=subplot_titles,
        horizontal_spacing=0.10 if len(embedding_models) >= 3 else 0.12,
        vertical_spacing=0.26,
    )
    if mc_muon_events is None or exp_events is None:
        return None
    mc_cut_mask_full = np.asarray(
        [event_passes_cut(e, threshold, 8, 2) for e in mc_muon_events], dtype=bool
    )
    exp_cut_mask_full = np.asarray(
        [event_passes_cut(e, threshold, 8, 2) for e in exp_events], dtype=bool
    )

    # Collect once per model with no cap, then balance both rows to a common count.
    embed_cap = max(max_events * 4, 1)
    rng = np.random.default_rng(42)

    for col, (model_name, model) in enumerate(embedding_models, start=1):
        print(f"  [HTML] Collecting embeddings for {model_name}...", flush=True)
        mc_emb_all, _ = collect_event_embeddings(model, mc_batches, "MC muatm", embed_cap)
        exp_emb_all, _ = collect_event_embeddings(model, exp_batches, "EXP", embed_cap)
        if len(mc_emb_all) == 0 or len(exp_emb_all) == 0:
            fig.add_annotation(
                text="No embeddings collected",
                x=0.5,
                y=0.5,
                showarrow=False,
                row=1,
                col=col,
            )
            continue

        mc_cut_mask = mc_cut_mask_full[: len(mc_emb_all)]
        exp_cut_mask = exp_cut_mask_full[: len(exp_emb_all)]
        mc_cut_idx = np.flatnonzero(mc_cut_mask)
        exp_cut_idx = np.flatnonzero(exp_cut_mask)

        # Common per-source size = min(no-cut available, after-cut available, soft cap).
        mc_target = int(min(len(mc_emb_all), len(mc_cut_idx), max_events * 2))
        exp_target = int(min(len(exp_emb_all), len(exp_cut_idx), max_events * 2))
        if mc_target == 0 or exp_target == 0:
            fig.add_annotation(
                text="Not enough events after 8-2 cut",
                x=0.5,
                y=0.5,
                showarrow=False,
                row=2,
                col=col,
            )
            continue

        mc_idx_no = rng.choice(len(mc_emb_all), mc_target, replace=False)
        exp_idx_no = rng.choice(len(exp_emb_all), exp_target, replace=False)
        mc_idx_cut = rng.choice(len(mc_cut_idx), mc_target, replace=False)
        exp_idx_cut = rng.choice(len(exp_cut_idx), exp_target, replace=False)
        mc_idx_cut = mc_cut_idx[mc_idx_cut]
        exp_idx_cut = exp_cut_idx[exp_idx_cut]

        for row, mc_idx, exp_idx, row_label in [
            (1, mc_idx_no, exp_idx_no, "no cut"),
            (2, mc_idx_cut, exp_idx_cut, "after 8-2"),
        ]:
            mc_emb = mc_emb_all[mc_idx]
            exp_emb = exp_emb_all[exp_idx]
            emb = np.concatenate([mc_emb, exp_emb], axis=0)
            lbl = np.asarray(["MC muatm"] * len(mc_emb) + ["EXP"] * len(exp_emb))
            print(
                f"  [HTML] Running UMAP for {model_name} ({row_label}): "
                f"{emb.shape[0]} events x {emb.shape[1]} dims...",
                flush=True,
            )
            emb_n = StandardScaler().fit_transform(emb)
            xy = umap.UMAP(
                n_neighbors=30, min_dist=0.1, metric="euclidean", random_state=42
            ).fit_transform(emb_n)
            print(f"  [HTML] UMAP done for {model_name} ({row_label}).", flush=True)
            for label, color in [("MC muatm", MC_COL), ("EXP", EXP_COL)]:
                m = lbl == label
                fig.add_trace(
                    go.Scattergl(
                        x=xy[m, 0],
                        y=xy[m, 1],
                        mode="markers",
                        name=label,
                        legendgroup=label,
                        showlegend=(row == 1 and col == 1),
                        marker=dict(size=4, opacity=0.5, color=color),
                    ),
                    row=row,
                    col=col,
                )
            fig.update_xaxes(title_text="UMAP-1", row=row, col=col)
            fig.update_yaxes(title_text="UMAP-2" if col == 1 else None, row=row, col=col)

    # Headers above each row (above the per-model subplot titles).
    fig.add_annotation(
        text="<b>no cut</b>",
        xref="paper",
        yref="paper",
        x=0.5,
        y=1.07,
        xanchor="center",
        yanchor="bottom",
        showarrow=False,
        font=dict(size=20, color="#1f1f1f"),
    )
    fig.add_annotation(
        text="<b>&ge; 8 predicted signal hits, &ge; 2 strings</b>",
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.50,
        xanchor="center",
        yanchor="middle",
        showarrow=False,
        font=dict(size=20, color="#1f1f1f"),
    )
    styled = _style_plotly(fig, title=None, height=1080)
    styled.update_layout(margin=dict(l=85, r=40, t=90, b=75))
    return styled


# ── MATPLOTLIB PNG PLOTS ──

EVTYPE_LABELS_MPL_PLAIN = {
    "muatm": "EAS",
    "nuatm": r"atmospheric $\nu_\mu$",
    "nue2": r"cosmogenic $\nu_\mu$",
}


def _mpl_apply_metric_yaxis(ax, y_min=METRIC_Y_MIN, y_max=1.01):
    ax.set_ylim(y_min, y_max)
    span = y_max - y_min
    if span <= 0.10:
        major, minor = 0.01, 0.005
    elif span <= 0.25:
        major, minor = 0.02, 0.01
    else:
        major, minor = METRIC_Y_TICK, METRIC_Y_MINOR
    ax.yaxis.set_major_locator(MultipleLocator(major))
    ax.yaxis.set_minor_locator(MultipleLocator(minor))
    ax.tick_params(which="major", length=6)
    ax.tick_params(which="minor", length=3)
    ax.grid(True, which="major", alpha=0.35)
    ax.grid(True, which="minor", alpha=0.18, linestyle=":")


def _mpl_apply_threshold_xaxis(ax):
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_locator(MultipleLocator(THRESHOLD_DTICK))
    ax.xaxis.set_minor_locator(MultipleLocator(THRESHOLD_MINOR))
    ax.tick_params(which="major", length=6)
    ax.tick_params(which="minor", length=3)
    ax.grid(True, which="major", axis="x", alpha=0.35)
    ax.grid(True, which="minor", axis="x", alpha=0.18, linestyle=":")


def _mpl_savefig(fig, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[PNG] Wrote {out_path}", flush=True)


def plot_score_distributions_png(mc_muon, exp_events, threshold, out_path, cuts=(8, 2)):
    min_hits, min_strings = cuts
    mc_after = [e for e in mc_muon if event_passes_cut(e, threshold, min_hits, min_strings)]
    exp_after = [e for e in exp_events if event_passes_cut(e, threshold, min_hits, min_strings)]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.4), constrained_layout=True)
    bins_score = np.linspace(0, 1, 100)

    rows = [
        (0, mc_muon, exp_events, "no cut", 0),
        (
            1,
            mc_after,
            exp_after,
            r"$\geq 8$ predicted hits, $\geq 2$ strings",
            min_hits,
        ),
    ]

    for row_idx, mc_row, exp_row, row_label, n_lo in rows:
        ax_score = axes[row_idx, 0]
        ax_mult = axes[row_idx, 1]

        for events, name, color in [(mc_row, "MC muon", MC_COL), (exp_row, "EXP", EXP_COL)]:
            if not events:
                continue
            scores = np.concatenate([e["probs"] for e in events])
            ax_score.hist(
                scores,
                bins=bins_score,
                density=True,
                histtype="step",
                lw=2.4,
                color=color,
                label=name,
            )
        ax_score.set_yscale("log")
        ax_score.set_xlabel(r"$\xi$")
        ax_score.set_ylabel("Density")
        _mpl_apply_threshold_xaxis(ax_score)
        if row_idx == 0:
            ax_score.legend(loc="upper center")
        ax_score.grid(True, which="major", alpha=0.3)

        multiplicities = [
            (
                np.asarray([int((e["probs"] > threshold).sum()) for e in mc_row]),
                "MC muon",
                MC_COL,
            ),
            (
                np.asarray([int((e["probs"] > threshold).sum()) for e in exp_row]),
                "EXP",
                EXP_COL,
            ),
        ]
        non_empty = [vals for vals, _, _ in multiplicities if len(vals) > 0]
        if non_empty:
            all_vals = np.concatenate(non_empty)
            n_hi = max(n_lo + 1, int(np.percentile(all_vals, 99)))
            bins_n = np.arange(n_lo, n_hi + 2) - 0.5
            for vals, name, color in multiplicities:
                if len(vals) == 0:
                    continue
                ax_mult.hist(
                    vals,
                    bins=bins_n,
                    density=True,
                    histtype="step",
                    lw=2.4,
                    color=color,
                    label=name,
                )
            ax_mult.set_xlim(bins_n[0], bins_n[-1])
        ax_mult.set_yscale("log")
        ax_mult.set_xlabel(r"$N$")
        ax_mult.set_ylabel("Density")
        ax_mult.minorticks_on()
        ax_mult.grid(True, which="major", alpha=0.3)

        ax_score.set_title(row_label, fontsize=15, fontweight="bold", pad=8, loc="left")

    _mpl_savefig(fig, out_path)


def _make_panel_grid(types, panel_size=(5.6, 4.6)):
    n = len(types)
    fig, axes = plt.subplots(
        1, n, figsize=(panel_size[0] * n, panel_size[1]), constrained_layout=True
    )
    if n == 1:
        axes = [axes]
    return fig, list(axes)


def plot_pr_curves_png(
    mc_events,
    out_path,
    cuts=(8, 2),
    macro_event=True,
    threshold_points=DEFAULT_THRESHOLD_POINTS,
    y_min=0.85,
):
    """One PNG with three panels (per event type), independent axes + own legend."""
    thresholds = np.linspace(0.01, 0.99, threshold_points)
    min_hits, min_strings = cuts
    types = ["muatm", "nuatm", "nue2"]
    fig, axes = _make_panel_grid(types)
    plotted = False
    for ax, evtype in zip(axes, types):
        evs = [e for e in mc_events if e["ev_type"] == evtype]
        if min_hits or min_strings:
            evs = [e for e in evs if event_passes_cut(e, 0.5, min_hits, min_strings)]
        if not evs:
            ax.set_title(f"{EVTYPE_LABELS_MPL_PLAIN[evtype]}: no events")
            ax.set_xlabel(r"$\xi$")
            continue
        if macro_event:
            precision, recall = _compute_macro_event_pr(evs, thresholds)
        else:
            precision, recall = _compute_micro_pr(evs, thresholds)
        ax.plot(thresholds, precision, lw=2.4, color=PRECISION_COL, label="precision")
        ax.plot(thresholds, recall, lw=2.4, ls="--", color=RECALL_COL, label="recall")
        ax.set_xlabel(r"$\xi$")
        ax.set_title(EVTYPE_LABELS_MPL_PLAIN[evtype])
        _mpl_apply_threshold_xaxis(ax)
        _mpl_apply_metric_yaxis(ax, y_min=y_min)
        ax.legend(loc="lower center")
        plotted = True
    if plotted:
        _mpl_savefig(fig, out_path)
    else:
        plt.close(fig)


def plot_event_pr_png(
    mc_events,
    out_path,
    cuts=(8, 2),
    threshold_points=DEFAULT_THRESHOLD_POINTS,
    y_min=0.75,
):
    """Event-level precision/recall vs threshold; one figure with 3 type-panels."""
    min_hits, min_strings = cuts
    thresholds = np.linspace(0.01, 0.99, threshold_points)
    types = ["muatm", "nuatm", "nue2"]
    fig, axes = _make_panel_grid(types)
    plotted = False
    for ax, evtype in zip(axes, types):
        evs = [e for e in mc_events if e["ev_type"] == evtype]
        if not evs:
            ax.set_title(f"{EVTYPE_LABELS_MPL_PLAIN[evtype]}: no events")
            ax.set_xlabel(r"$\xi$")
            continue
        truly_pos = np.asarray(
            [event_truly_passes(e, min_hits, min_strings) for e in evs], dtype=bool
        )
        precisions = np.full_like(thresholds, np.nan, dtype=np.float64)
        recalls = np.full_like(thresholds, np.nan, dtype=np.float64)
        prec_err = np.zeros_like(thresholds, dtype=np.float64)
        rec_err = np.zeros_like(thresholds, dtype=np.float64)
        for i, thr in enumerate(thresholds):
            pred_pos = np.asarray(
                [event_passes_cut(e, thr, min_hits, min_strings) for e in evs], dtype=bool
            )
            tp = int((pred_pos & truly_pos).sum())
            fp = int((pred_pos & ~truly_pos).sum())
            fn = int((~pred_pos & truly_pos).sum())
            p, r = _metric_from_counts(tp, fp, fn)
            precisions[i] = p
            recalls[i] = r
            prec_err[i] = _binom_err(p, tp + fp) or 0.0
            rec_err[i] = _binom_err(r, tp + fn) or 0.0

        for vals, errs, color, alpha in [
            (precisions, prec_err, PRECISION_COL, 0.18),
            (recalls, rec_err, RECALL_COL, 0.18),
        ]:
            mask = np.isfinite(vals)
            if not mask.any():
                continue
            up = np.clip(vals[mask] + errs[mask], 0, 1)
            lo = np.clip(vals[mask] - errs[mask], 0, 1)
            ax.fill_between(thresholds[mask], lo, up, color=color, alpha=alpha, linewidth=0)
        ax.plot(thresholds, precisions, lw=2.4, color=PRECISION_COL, label="precision")
        ax.plot(
            thresholds,
            recalls,
            lw=2.4,
            ls="--",
            color=RECALL_COL,
            label="recall (event efficiency)",
        )
        ax.set_xlabel(r"$\xi$")
        ax.set_title(EVTYPE_LABELS_MPL_PLAIN[evtype])
        _mpl_apply_threshold_xaxis(ax)
        _mpl_apply_metric_yaxis(ax, y_min=y_min)
        ax.legend(loc="lower center")
        plotted = True
    if plotted:
        _mpl_savefig(fig, out_path)
    else:
        plt.close(fig)


def _energy_bin_metrics(events, n_bins, metric_fn):
    """Compute log10-energy-binned (centers, precisions, recalls, n_per_bin)."""
    log_es = np.log10(np.asarray([e["energy"] for e in events], dtype=np.float64))
    bins = np.linspace(np.nanmin(log_es), np.nanmax(log_es), n_bins)
    centers, precisions, recalls, n_per_bin = [], [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        selected = [e for e in events if lo <= np.log10(e["energy"]) < hi]
        if not selected:
            continue
        p, r = metric_fn(selected)
        centers.append(0.5 * (lo + hi))
        precisions.append(p)
        recalls.append(r)
        n_per_bin.append(len(selected))
    return centers, precisions, recalls, n_per_bin


def _trim_to_min_bin(values_list, n_per_bin, min_events_per_bin):
    keep = [n > min_events_per_bin for n in n_per_bin]
    if not any(keep):
        return None
    first = keep.index(True)
    last = len(keep) - 1 - keep[::-1].index(True)
    return [v[first : last + 1] for v in values_list]


def plot_energy_dependence_png(
    mc_events,
    threshold,
    out_path,
    min_events_per_bin=300,
    energy_unit="GeV",
):
    """Hit-level (micro) precision/recall vs log10 energy; 1 figure × 3 type-panels."""
    evs_all = [e for e in mc_events if np.isfinite(e.get("energy", np.nan)) and e["energy"] > 0]
    if not evs_all:
        return
    types = ["muatm", "nuatm", "nue2"]
    fig, axes = _make_panel_grid(types)
    plotted = False

    def _hit_micro(selected):
        tp = fp = fn = 0
        for ev in selected:
            pred = ev["probs"] > threshold
            true = ev["true_sig"].astype(bool)
            tp += int((pred & true).sum())
            fp += int((pred & ~true).sum())
            fn += int((~pred & true).sum())
        return _metric_from_counts(tp, fp, fn)

    for ax, evtype in zip(axes, types):
        type_events = [e for e in evs_all if e["ev_type"] == evtype]
        if len(type_events) < 2:
            ax.set_title(f"{EVTYPE_LABELS_MPL_PLAIN[evtype]}: no events")
            ax.set_xlabel(rf"$\log_{{10}}(E\,/\,\mathrm{{{energy_unit}}})$")
            continue
        centers, precs, recs, n_per_bin = _energy_bin_metrics(type_events, 12, _hit_micro)
        if not centers:
            continue
        trimmed = _trim_to_min_bin([centers, precs, recs], n_per_bin, min_events_per_bin)
        if trimmed is None:
            continue
        centers, precs, recs = trimmed
        finite = [v for v in precs + recs if np.isfinite(v)]
        y_min_local = max(METRIC_Y_MIN, (min(finite) if finite else METRIC_Y_MIN) - 0.02)
        ax.plot(centers, precs, "o-", lw=2.4, color=PRECISION_COL, label="precision", markersize=7)
        ax.plot(centers, recs, "s--", lw=2.4, color=RECALL_COL, label="recall", markersize=7)
        ax.set_xlabel(rf"$\log_{{10}}(E\,/\,\mathrm{{{energy_unit}}})$")
        ax.set_title(EVTYPE_LABELS_MPL_PLAIN[evtype])
        ax.minorticks_on()
        _mpl_apply_metric_yaxis(ax, y_min=y_min_local)
        ax.legend(loc="lower right")
        plotted = True
    if plotted:
        _mpl_savefig(fig, out_path)
    else:
        plt.close(fig)


def plot_event_energy_dependence_png(
    mc_events,
    threshold,
    out_path,
    cuts=(8, 2),
    min_events_per_bin=200,
    energy_unit="GeV",
):
    """Event-level precision/recall vs log10 energy; 1 figure × 3 type-panels."""
    min_hits, min_strings = cuts
    evs_all = [e for e in mc_events if np.isfinite(e.get("energy", np.nan)) and e["energy"] > 0]
    if not evs_all:
        return
    types = ["muatm", "nuatm", "nue2"]
    fig, axes = _make_panel_grid(types)
    plotted = False

    def _event_micro(selected):
        pred_pos = np.asarray(
            [event_passes_cut(e, threshold, min_hits, min_strings) for e in selected], dtype=bool
        )
        true_pos = np.asarray(
            [event_truly_passes(e, min_hits, min_strings) for e in selected], dtype=bool
        )
        tp = int((pred_pos & true_pos).sum())
        fp = int((pred_pos & ~true_pos).sum())
        fn = int((~pred_pos & true_pos).sum())
        return _metric_from_counts(tp, fp, fn)

    for ax, evtype in zip(axes, types):
        type_events = [e for e in evs_all if e["ev_type"] == evtype]
        if len(type_events) < 2:
            ax.set_title(f"{EVTYPE_LABELS_MPL_PLAIN[evtype]}: no events")
            ax.set_xlabel(rf"$\log_{{10}}(E\,/\,\mathrm{{{energy_unit}}})$")
            continue
        centers, precs, recs, n_per_bin = _energy_bin_metrics(type_events, 12, _event_micro)
        if not centers:
            continue
        trimmed = _trim_to_min_bin([centers, precs, recs], n_per_bin, min_events_per_bin)
        if trimmed is None:
            continue
        centers, precs, recs = trimmed
        finite = [v for v in precs + recs if np.isfinite(v)]
        y_min_local = max(0.0, (min(finite) if finite else 0.0) - 0.05)
        ax.plot(centers, precs, "o-", lw=2.4, color=PRECISION_COL, label="precision", markersize=7)
        ax.plot(
            centers,
            recs,
            "s--",
            lw=2.4,
            color=RECALL_COL,
            label="recall (event efficiency)",
            markersize=7,
        )
        ax.set_xlabel(rf"$\log_{{10}}(E\,/\,\mathrm{{{energy_unit}}})$")
        ax.set_title(EVTYPE_LABELS_MPL_PLAIN[evtype])
        ax.minorticks_on()
        _mpl_apply_metric_yaxis(ax, y_min=y_min_local)
        ax.legend(loc="lower right")
        plotted = True
    if plotted:
        _mpl_savefig(fig, out_path)
    else:
        plt.close(fig)


def plot_tres_dependence_png(mc_events, threshold, out_path):
    evs = [e for e in mc_events if e["ev_type"] == "nue2"]
    if not evs:
        return
    probs = np.concatenate([e["probs"] for e in evs])
    true = np.concatenate([e["true_sig"].astype(bool) for e in evs])
    tres = np.abs(np.concatenate([e["t_res"] for e in evs]))
    bins = np.array([0, 2, 5, 10, 20, 40, 80, 160, 320], dtype=float)
    labels = [f"{lo:g}-{hi:g}" for lo, hi in zip(bins[:-1], bins[1:])]
    pred = probs > threshold
    precisions, recalls = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (tres >= lo) & (tres < hi)
        tp = int((m & pred & true).sum())
        fp = int((m & pred & ~true).sum())
        fn = int((m & ~pred & true).sum())
        p, r = _metric_from_counts(tp, fp, fn)
        precisions.append(p)
        recalls.append(r)

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)
    ax.plot(x, precisions, "o-", lw=2.4, color=PRECISION_COL, label="precision", markersize=8)
    ax.plot(x, recalls, "s--", lw=2.4, color=RECALL_COL, label="recall", markersize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_xlabel(r"$|t_\mathrm{res}|$ bin, ns")
    finite = [v for v in precisions + recalls if np.isfinite(v)]
    y_min_local = max(0.7, min(finite) - 0.03) if finite else 0.7
    _mpl_apply_metric_yaxis(ax, y_min=y_min_local)
    ax.set_title(EVTYPE_LABELS_MPL_PLAIN["nue2"])
    ax.legend(loc="lower left")

    _mpl_savefig(fig, out_path)


def plot_da_embedding_umap_png(
    no_da_model,
    da_model,
    mc_batches,
    exp_batches,
    max_events,
    out_path,
    mc_muon_events=None,
    exp_events=None,
    threshold=0.5,
):
    """2×2 panels: rows = (no cut, ≥8 hits ≥2 strings); cols = (No DA, DA).
    Single shared legend; MC muon labelled "EAS" to match PR plots."""
    try:
        import umap
    except ImportError:
        print("[PNG] Skipping UMAP — umap-learn not installed.")
        return

    have_cut = mc_muon_events is not None and exp_events is not None
    if have_cut:
        mc_cut_full = np.asarray(
            [event_passes_cut(e, threshold, 8, 2) for e in mc_muon_events], dtype=bool
        )
        exp_cut_full = np.asarray(
            [event_passes_cut(e, threshold, 8, 2) for e in exp_events], dtype=bool
        )
    else:
        mc_cut_full = exp_cut_full = None

    UMAP_FS = 12

    n_rows = 2 if have_cut else 1
    fig, axes = plt.subplots(
        n_rows,
        2,
        figsize=(13.5, 6.0 * n_rows),
        constrained_layout=True,
        squeeze=False,
    )
    color_map = {"MC EAS": MC_COL, "EXP": EXP_COL}
    embed_cap = max(max_events * 4, 1)
    rng = np.random.default_rng(42)
    plotted = False

    model_items = [
        (name, m) for name, m in [("No DA", no_da_model), ("DA", da_model)] if m is not None
    ]
    if not model_items:
        plt.close(fig)
        return

    embeddings_by_model = {}
    for model_name, model in model_items:
        print(f"  [PNG-UMAP] Collecting embeddings: {model_name}", flush=True)
        mc_emb_all, _ = collect_event_embeddings(model, mc_batches, "MC EAS", embed_cap)
        exp_emb_all, _ = collect_event_embeddings(model, exp_batches, "EXP", embed_cap)
        embeddings_by_model[model_name] = (mc_emb_all, exp_emb_all)

    rows_specs = [(0, False, "no cut")]
    if have_cut:
        rows_specs.append((1, True, r"$\geq$ 8 hits, $\geq$ 2 strings"))

    all_pools = {}
    for row, use_cut, row_label in rows_specs:
        per_panel = {}
        for model_name, _ in model_items:
            mc_emb_all, exp_emb_all = embeddings_by_model[model_name]
            if mc_emb_all.size == 0 or exp_emb_all.size == 0:
                per_panel[model_name] = (None, None)
                continue
            if not use_cut:
                per_panel[model_name] = (mc_emb_all, exp_emb_all)
            else:
                mc_cut = mc_cut_full[: len(mc_emb_all)]
                exp_cut = exp_cut_full[: len(exp_emb_all)]
                per_panel[model_name] = (
                    mc_emb_all[np.flatnonzero(mc_cut)],
                    exp_emb_all[np.flatnonzero(exp_cut)],
                )
        all_pools[(row, row_label)] = per_panel

    # Single global target across ALL panels (rows and columns).
    all_min_sizes = []
    for per_panel in all_pools.values():
        for mc_p, exp_p in per_panel.values():
            if mc_p is not None:
                all_min_sizes.append(min(len(mc_p), len(exp_p)))
    global_target = min(all_min_sizes + [max_events]) if all_min_sizes else 0
    print(f"  [PNG-UMAP] global_target = {global_target} events per panel", flush=True)

    for (row, row_label), per_panel in all_pools.items():
        for col, (model_name, _) in enumerate(model_items):
            ax = axes[row, col]
            mc_pool, exp_pool = per_panel[model_name]
            if mc_pool is None or global_target < 50:
                ax.set_axis_off()
                ax.set_title(
                    f"{model_name}, {row_label}\n(insufficient events)",
                    fontsize=UMAP_FS,
                )
                continue
            mc_idx = rng.choice(len(mc_pool), global_target, replace=False)
            exp_idx = rng.choice(len(exp_pool), global_target, replace=False)
            mc_emb = mc_pool[mc_idx]
            exp_emb = exp_pool[exp_idx]
            emb = np.concatenate([mc_emb, exp_emb], axis=0)
            lbl = np.asarray(["MC EAS"] * len(mc_emb) + ["EXP"] * len(exp_emb))
            print(
                f"  [PNG-UMAP] {model_name} | {row_label}: "
                f"{emb.shape[0]} events x {emb.shape[1]} dims",
                flush=True,
            )
            emb_n = StandardScaler().fit_transform(emb)
            xy = umap.UMAP(
                n_neighbors=30, min_dist=0.1, metric="euclidean", random_state=42
            ).fit_transform(emb_n)
            for label in ["MC EAS", "EXP"]:
                m = lbl == label
                ax.scatter(
                    xy[m, 0],
                    xy[m, 1],
                    s=5,
                    alpha=0.45,
                    color=color_map[label],
                    label=label,
                    rasterized=True,
                )
            ax.set_title(f"{model_name}, {row_label}", fontsize=UMAP_FS)
            ax.set_ylabel("UMAP-2", fontsize=UMAP_FS)
            ax.set_xlabel("UMAP-1", fontsize=UMAP_FS)
            ax.tick_params(labelsize=UMAP_FS - 1)
            ax.grid(True, alpha=0.3)
            plotted = True

    if not plotted:
        plt.close(fig)
        return

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=MC_COL,
            markersize=10,
            label="MC EAS",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=EXP_COL,
            markersize=10,
            label="EXP",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 1.0),
        frameon=True,
        fontsize=UMAP_FS,
    )
    _mpl_savefig(fig, out_path)


def plot_wasserstein_mc_exp_png(
    no_da_model,
    da_model,
    mc_batches,
    exp_batches,
    threshold,
    out_path,
    cuts=((0, 0), (8, 2)),
    signal_definition="label_nonzero",
):
    """Wasserstein distance between MC muon and EXP P(signal) distributions:
    no-DA vs DA, evaluated at no-cut and (8, 2) event-quality cut."""
    try:
        from scipy.stats import wasserstein_distance
    except ImportError:
        print("[PNG] Skipping wasserstein plot — scipy not installed.")
        return

    print("[PNG] Computing wasserstein MC-muon vs EXP for No DA / DA...", flush=True)
    no_da_mc_all = infer_mc(no_da_model, iter(mc_batches), signal_definition=signal_definition)
    no_da_mc = [e for e in no_da_mc_all if e["ev_type"] == "muatm"]
    no_da_ex = infer_exp(no_da_model, iter(exp_batches))
    da_mc_all = infer_mc(da_model, iter(mc_batches), signal_definition=signal_definition)
    da_mc = [e for e in da_mc_all if e["ev_type"] == "muatm"]
    da_ex = infer_exp(da_model, iter(exp_batches))

    cut_labels, no_da_w, da_w, n_kept = [], [], [], []
    for mh, ms in cuts:
        nda_mc_k = [e for e in no_da_mc if event_passes_cut(e, threshold, mh, ms)]
        nda_ex_k = [e for e in no_da_ex if event_passes_cut(e, threshold, mh, ms)]
        da_mc_k = [e for e in da_mc if event_passes_cut(e, threshold, mh, ms)]
        da_ex_k = [e for e in da_ex if event_passes_cut(e, threshold, mh, ms)]
        if not (nda_mc_k and nda_ex_k and da_mc_k and da_ex_k):
            print(f"  skipping cut ({mh}, {ms}) — empty after-cut sample")
            continue
        nda_mc_p = np.concatenate([e["probs"] for e in nda_mc_k])
        nda_ex_p = np.concatenate([e["probs"] for e in nda_ex_k])
        da_mc_p = np.concatenate([e["probs"] for e in da_mc_k])
        da_ex_p = np.concatenate([e["probs"] for e in da_ex_k])
        no_da_w.append(float(wasserstein_distance(nda_mc_p, nda_ex_p)))
        da_w.append(float(wasserstein_distance(da_mc_p, da_ex_p)))
        cut_labels.append("no cut" if (mh, ms) == (0, 0) else f"({mh}, {ms})")
        n_kept.append((len(nda_mc_k) + len(nda_ex_k), len(da_mc_k) + len(da_ex_k)))
        print(
            f"  cut ({mh}, {ms}): No DA W={no_da_w[-1]:.4f} (n={n_kept[-1][0]}) | "
            f"DA W={da_w[-1]:.4f} (n={n_kept[-1][1]})"
        )

    if not cut_labels:
        return

    no_da_arr = np.asarray(no_da_w)
    da_arr = np.asarray(da_w)

    fig, ax = plt.subplots(figsize=(7.5, 5.2), constrained_layout=True)
    x = np.arange(len(cut_labels))
    bar_w = 0.36
    no_da_color = "#1f77b4"
    da_color = "#ff7f0e"
    bars_nd = ax.bar(
        x - bar_w / 2,
        no_da_arr,
        bar_w,
        color=no_da_color,
        edgecolor="black",
        linewidth=0.8,
        label="No DA",
    )
    bars_da = ax.bar(
        x + bar_w / 2,
        da_arr,
        bar_w,
        color=da_color,
        edgecolor="black",
        linewidth=0.8,
        label="With DA",
    )
    for bar, val in zip(bars_nd, no_da_arr):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            color=no_da_color,
            fontweight="bold",
        )
    for bar, val in zip(bars_da, da_arr):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            color=da_color,
            fontweight="bold",
        )

    finite = np.isfinite(no_da_arr) & np.isfinite(da_arr) & (no_da_arr > 0)
    if finite.any():
        avg_imp = float(np.mean((no_da_arr[finite] - da_arr[finite]) / no_da_arr[finite]) * 100)
        ax.text(
            0.98,
            0.97,
            f"Avg. improvement: {avg_imp:+.1f}%",
            transform=ax.transAxes,
            va="top",
            ha="right",
            fontsize=11,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.85, edgecolor="0.6"),
        )

    ax.set_xticks(x)
    ax.set_xticklabels(cut_labels)
    ax.set_xlabel("Event quality cut (min predicted hits, min strings)")
    ax.set_ylabel("Wasserstein distance")
    ax.set_title("MC muon vs EXP — P(signal) alignment", fontsize=13)
    y_top = float(max(no_da_arr.max(), da_arr.max()))
    ax.set_ylim(0, y_top * 1.22 + 1e-6)
    ax.yaxis.grid(True, which="major", alpha=0.35)
    ax.yaxis.grid(True, which="minor", alpha=0.18, linestyle=":")
    ax.minorticks_on()
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", framealpha=0.95)

    _mpl_savefig(fig, out_path)


def plot_tres_prediction_error_png(true_tres, pred_tres, out_path, bins=None, log_x=None):
    finite = np.isfinite(true_tres) & np.isfinite(pred_tres)
    if not finite.any():
        return
    true_tres = np.abs(true_tres[finite])
    pred_tres = np.abs(pred_tres[finite])
    err = np.abs(true_tres - pred_tres)

    if bins is None:
        bins = np.array([0, 2, 5, 10, 20, 40, 80, 160, 320], dtype=float)
    bins = np.asarray(bins, dtype=float)
    if log_x is None:
        log_x = bool(bins.max() / max(bins[bins > 0].min(), 1e-9) > 30)
    centers = 0.5 * (bins[:-1] + bins[1:])
    medians, means, q25, q75 = [], [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (true_tres >= lo) & (true_tres < hi)
        if not m.any():
            for arr in (medians, means, q25, q75):
                arr.append(np.nan)
            continue
        medians.append(float(np.median(err[m])))
        means.append(float(err[m].mean()))
        q25.append(float(np.quantile(err[m], 0.25)))
        q75.append(float(np.quantile(err[m], 0.75)))

    medians = np.asarray(medians)
    means = np.asarray(means)
    q25 = np.asarray(q25)
    q75 = np.asarray(q75)

    fig, ax = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)
    valid = np.isfinite(medians)
    if log_x:
        plot_centers = np.array([np.sqrt(max(lo, 0.5) * hi) for lo, hi in zip(bins[:-1], bins[1:])])
    else:
        plot_centers = centers
    ax.fill_between(
        plot_centers[valid],
        q25[valid],
        q75[valid],
        color=PRECISION_COL,
        alpha=0.18,
        label="IQR",
        linewidth=0,
    )
    ax.plot(
        plot_centers[valid],
        medians[valid],
        "o-",
        lw=2.4,
        color=PRECISION_COL,
        label="median",
        markersize=7,
    )
    ax.plot(
        plot_centers[valid],
        means[valid],
        "s--",
        lw=2.2,
        color=RECALL_COL,
        label="mean",
        markersize=7,
    )
    ax.set_xlabel(r"true $|t_\mathrm{res}|$, ns")
    ax.set_ylabel(r"$|\,\hat{t}_\mathrm{res} - t_\mathrm{res}\,|$, ns")
    if log_x:
        ax.set_xscale("log")
        ax.set_xlim(left=0.8)
    ax.minorticks_on()
    ax.grid(True, which="major", alpha=0.35)
    ax.grid(True, which="minor", alpha=0.18, linestyle=":")
    ax.legend(loc="lower right" if log_x else "upper left")

    _mpl_savefig(fig, out_path)


def write_matplotlib_pngs(
    out_dir,
    mc_events,
    mc_muon,
    exp_events,
    threshold,
    tres_prediction=None,
    threshold_points=DEFAULT_THRESHOLD_POINTS,
    embedding_models=None,
    mc_batches=None,
    exp_batches=None,
    embedding_max_events=5000,
    mc_muon_for_cut=None,
    exp_for_cut=None,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[PNG] Writing matplotlib PNGs to {out_dir}", flush=True)

    plot_score_distributions_png(
        mc_muon, exp_events, threshold, out_dir / "score_distributions.png"
    )
    plot_pr_curves_png(
        mc_events,
        out_dir / "precision_recall_after_8_2.png",
        cuts=(8, 2),
        macro_event=True,
        threshold_points=threshold_points,
        y_min=0.85,
    )
    plot_event_pr_png(
        mc_events,
        out_dir / "event_precision_recall.png",
        cuts=(8, 2),
        threshold_points=threshold_points,
        y_min=0.75,
    )
    plot_energy_dependence_png(mc_events, threshold, out_dir / "energy_dependence.png")
    plot_event_energy_dependence_png(
        mc_events, threshold, out_dir / "event_energy_dependence.png", cuts=(8, 2)
    )
    plot_tres_dependence_png(mc_events, threshold, out_dir / "nue2_tres_dependence.png")
    if tres_prediction is not None:
        plot_tres_prediction_error_png(
            tres_prediction[0],
            tres_prediction[1],
            out_dir / "tres_prediction_error.png",
        )
        plot_tres_prediction_error_png(
            tres_prediction[0],
            tres_prediction[1],
            out_dir / "tres_prediction_error_lt30.png",
            bins=np.array([0, 2, 5, 10, 15, 20, 25, 30], dtype=float),
        )

    if embedding_models and mc_batches is not None and exp_batches is not None:
        no_da_model = embedding_models[0][1] if len(embedding_models) >= 1 else None
        da_model = embedding_models[1][1] if len(embedding_models) >= 2 else None
        if no_da_model is not None and da_model is not None:
            plot_da_embedding_umap_png(
                no_da_model,
                da_model,
                mc_batches,
                exp_batches,
                embedding_max_events,
                out_dir / "da_embedding_umap.png",
                mc_muon_events=mc_muon_for_cut if mc_muon_for_cut is not None else mc_muon,
                exp_events=exp_for_cut if exp_for_cut is not None else exp_events,
                threshold=threshold,
            )
            plot_wasserstein_mc_exp_png(
                no_da_model,
                da_model,
                mc_batches,
                exp_batches,
                threshold,
                out_dir / "wasserstein_mc_muon_vs_exp.png",
            )


def _html_section(title, body):
    return f'<section class="section"><h2>{title}</h2>{body}</section>'


def _fig_to_html(fig, include_plotlyjs=False):
    return pio.to_html(
        fig,
        include_plotlyjs="cdn" if include_plotlyjs else False,
        full_html=False,
        config={
            "displaylogo": False,
            "toImageButtonOptions": {"format": "svg", "scale": 2},
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        },
    )


def _slugify_title(title):
    slug = re.sub(r"[^a-z0-9]+", "_", title.lower())
    return slug.strip("_") or "figure"


def _write_individual_plot(fig, title, figures_dir, include_plotlyjs="cdn"):
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    slug = _slugify_title(title)
    html_path = figures_dir / f"{slug}.html"
    png_path = figures_dir / f"{slug}.png"
    print(f"[HTML] Saving standalone figure: {html_path}...", flush=True)
    pio.write_html(
        fig,
        html_path,
        include_plotlyjs=include_plotlyjs,
        full_html=True,
        config={
            "displaylogo": False,
            "toImageButtonOptions": {"format": "svg", "scale": 2},
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        },
    )
    try:
        print(f"[HTML] Saving PNG figure: {png_path}...", flush=True)
        fig.write_image(str(png_path), scale=2)
    except Exception as exc:
        print(
            f"[WARN] Could not save PNG for '{title}': {type(exc).__name__}: {exc}. "
            "Install kaleido to enable Plotly PNG export.",
            flush=True,
        )


def write_html_report(
    html_path,
    model_name,
    mc_events,
    mc_muon,
    exp_events,
    signal_desc,
    threshold,
    model_comparison=None,
    tres_comparison=None,
    tres_prediction=None,
    embedding_models=None,
    mc_batches=None,
    exp_batches=None,
    embedding_max_events=5000,
    threshold_points=DEFAULT_THRESHOLD_POINTS,
    save_individual_figures=True,
):
    figures = []

    def add_figure(title, maker):
        print(f"[HTML] Building: {title}...", flush=True)
        fig = maker()
        figures.append((title, fig))
        print(f"[HTML] Done: {title}.", flush=True)

    add_figure(
        "Precision / Recall: Macro Over Events",
        lambda: make_pr_fig(
            mc_events, signal_desc, macro_event=True, threshold_points=threshold_points
        ),
    )
    add_figure(
        "Score Distributions", lambda: make_score_distribution_fig(mc_muon, exp_events, threshold)
    )
    add_figure(
        "Metrics After NN Event Cut 8-2",
        lambda: make_pr_fig(
            mc_events,
            signal_desc,
            macro_event=True,
            cuts=(8, 2),
            threshold_points=threshold_points,
        ),
    )
    add_figure(
        "Event-Level Precision and Recall (cut 8-2)",
        lambda: make_event_pr_fig(mc_events, threshold_points=threshold_points),
    )

    print("[HTML] Building: Primary Energy Dependence...", flush=True)
    energy_fig = make_energy_metrics_fig(mc_events, threshold)
    if energy_fig is not None:
        figures.append(("Primary Energy Dependence", energy_fig))
        print("[HTML] Done: Primary Energy Dependence.", flush=True)
    else:
        print("[HTML] Skipped: Primary Energy Dependence (no energy field).", flush=True)

    print("[HTML] Building: nue2 |t_res| Dependence...", flush=True)
    tres_fig = make_tres_metrics_fig(mc_events, threshold)
    if tres_fig is not None:
        figures.append(("nue2 |t_res| Dependence", tres_fig))
        print("[HTML] Done: nue2 |t_res| Dependence.", flush=True)
    else:
        print("[HTML] Skipped: nue2 |t_res| Dependence (no nue2 events).", flush=True)

    if tres_comparison is not None:
        add_figure(
            "Classification-only vs classification+t_res",
            lambda: make_binary_model_comparison_fig(
                tres_comparison[0],
                tres_comparison[1],
                signal_desc,
                threshold_points=threshold_points,
            ),
        )
    if tres_prediction is not None:
        add_figure(
            "t_res Prediction",
            lambda: make_tres_prediction_fig(tres_prediction[0], tres_prediction[1]),
        )
    if embedding_models is not None and mc_batches is not None and exp_batches is not None:
        print("[HTML] Building: Embedding UMAP comparison...", flush=True)
        umap_fig = make_da_embedding_umap_fig(
            embedding_models,
            mc_batches,
            exp_batches,
            embedding_max_events,
            mc_muon_events=mc_muon,
            exp_events=exp_events,
            threshold=threshold,
        )
        if umap_fig is not None:
            figures.append(("Embedding UMAP Comparison", umap_fig))
            print("[HTML] Done: Embedding UMAP comparison.", flush=True)
        else:
            print("[HTML] Skipped: Embedding UMAP comparison (umap unavailable).", flush=True)

    body = []
    body.append(
        f"""
        <header>
          <h1>Noise/Signal Classification Report</h1>
          <p class="subtitle">Model: <b>{model_name}</b> · MC events: {len(mc_events):,} ·
          EXP events: {len(exp_events):,} · threshold={threshold:g} · signal={signal_desc}</p>
        </header>
        """
    )
    body.append(
        """
        <section class="note">
          <b>Notes.</b> Precision is TP/(TP+FP); recall is TP/(TP+FN).
          Precision/recall curves are macro over events: metrics are computed per event and
          then averaged over finite event values. Primary energy is read from
          <code>BMCEvent.fPrimaryParticleEnergy</code> and shown as <code>log10(E / TeV)</code>.
        </section>
        """
    )
    html_path = Path(html_path)
    figures_dir = html_path.with_suffix("").parent / f"{html_path.stem}_figures"
    for i, (title, fig) in enumerate(tqdm(figures, desc="Serializing Plotly figures")):
        print(f"[HTML] Serializing: {title}...", flush=True)
        if save_individual_figures:
            _write_individual_plot(fig, title, figures_dir)
        body.append(_html_section(title, _fig_to_html(fig, include_plotlyjs=i == 0)))

    html = f"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Noise/Signal Classification Report</title>
  <style>
    body {{
      margin: 0;
      background: #f7f7f5;
      color: #1f1f1f;
      font-family: Arial, Helvetica, sans-serif;
      line-height: 1.45;
    }}
    header, .section, .note {{
      max-width: 1320px;
      margin: 24px auto;
      background: #fff;
      border: 1px solid #deded8;
      box-shadow: 0 1px 2px rgba(0,0,0,0.04);
      padding: 22px 28px;
    }}
    h1 {{
      margin: 0 0 8px 0;
      font-size: 30px;
      font-weight: 700;
      letter-spacing: -0.02em;
    }}
    h2 {{
      margin: 0 0 14px 0;
      font-size: 22px;
      font-weight: 700;
      border-bottom: 1px solid #e2e2dc;
      padding-bottom: 8px;
    }}
    .subtitle {{
      margin: 0;
      color: #555;
      font-size: 15px;
    }}
    .note {{
      font-size: 14px;
      color: #333;
      background: #fffffb;
    }}
    code {{
      background: #f0f0ec;
      padding: 1px 4px;
      border-radius: 3px;
    }}
  </style>
</head>
<body>
{"".join(body)}
</body>
</html>
"""
    html_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[HTML] Writing file: {html_path}...", flush=True)
    html_path.write_text(html)
    print(f"[HTML] Wrote file: {html_path}.", flush=True)


def _git_info(repo_root):
    def _run(cmd):
        try:
            return (
                subprocess.check_output(cmd, cwd=repo_root, stderr=subprocess.DEVNULL)
                .decode()
                .strip()
            )
        except Exception:
            return None

    sha = _run(["git", "rev-parse", "HEAD"])
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    status = _run(["git", "status", "--porcelain"])
    dirty = bool(status) if status is not None else None
    return {
        "commit": sha,
        "branch": branch,
        "dirty": dirty,
        "status": status,
        "status_filtered": _filter_pipeline_status(status),
    }


# Files that actually affect training / evaluation behavior. Anything outside this
# allow-list (notebooks, ad-hoc plots, legacy scripts, generated artefacts, etc.) is
# stripped from the git status block in report_info.md to keep it readable.
_PIPELINE_PATH_PREFIXES = (
    "pytorch_training/conf/",
    "pytorch_training/data_utils/",
    "pytorch_training/models/",
    "pytorch_training/training/",
    "pytorch_training/train_types/",
    "pytorch_training/train_configs/",
)
_PIPELINE_EXACT_FILES = {
    "pytorch_training/train.py",
    "pytorch_training/scripts/generate_report.py",
    "pytorch_training/scripts/eval_noise_sig.py",
    "pytorch_training/scripts/eval_da.py",
    "pytorch_training/scripts/plot_training_metrics.py",
}


def _filter_pipeline_status(status):
    """Return git status --porcelain output filtered to train/eval-relevant paths."""
    if not status:
        return status
    kept = []
    for raw in status.splitlines():
        if len(raw) < 4:
            continue
        path = raw[3:].split(" -> ")[-1].strip().strip('"')
        if path in _PIPELINE_EXACT_FILES or any(
            path.startswith(prefix) for prefix in _PIPELINE_PATH_PREFIXES
        ):
            kept.append(raw)
    return "\n".join(kept)


def _file_meta(path):
    p = Path(path)
    if not p.exists():
        return {"path": str(p), "exists": False}
    h = hashlib.sha1()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return {
        "path": str(p.resolve()),
        "exists": True,
        "size_bytes": p.stat().st_size,
        "mtime": datetime.datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds"),
        "sha1": h.hexdigest(),
    }


def save_predictions(out_path, mc_events, exp_events, tres_prediction=None):
    """Save per-hit predictions in a single .npz file for plot reproduction."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _stack(events, fields):
        if not events:
            return {f: np.zeros(0, dtype=np.float32) for f in fields}, np.zeros(0, dtype=np.int64)
        arrays = {f: [] for f in fields}
        starts = [0]
        for ev in events:
            for f in fields:
                arr = ev.get(f)
                if arr is None:
                    arrays[f].append(np.zeros(0, dtype=np.float32))
                else:
                    arrays[f].append(np.asarray(arr))
            n = len(ev["probs"])
            starts.append(starts[-1] + n)
        return (
            {f: np.concatenate(arrays[f]) if arrays[f] else np.zeros(0) for f in fields},
            np.asarray(starts, dtype=np.int64),
        )

    mc_fields = ("probs", "true_sig", "t_res", "channels")
    mc_arr, mc_starts = _stack(mc_events, mc_fields)
    mc_types = np.asarray([e["ev_type"] for e in mc_events])
    mc_energy = np.asarray([e.get("energy", np.nan) for e in mc_events], dtype=np.float64)

    exp_fields = ("probs", "channels")
    exp_arr, exp_starts = _stack(exp_events, exp_fields)

    payload = {
        "mc_ev_starts": mc_starts,
        "mc_ev_type": mc_types,
        "mc_energy": mc_energy,
        **{f"mc_{k}": v for k, v in mc_arr.items()},
        "exp_ev_starts": exp_starts,
        **{f"exp_{k}": v for k, v in exp_arr.items()},
    }
    if tres_prediction is not None:
        true_t, pred_t = tres_prediction
        payload["tres_true"] = np.asarray(true_t)
        payload["tres_pred"] = np.asarray(pred_t)

    np.savez_compressed(out_path, **payload)
    print(f"[REPRO] Saved predictions: {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")


def write_repro_md(
    md_path,
    args,
    *,
    signal_desc,
    n_mc,
    n_exp,
    mc_per_type,
    exp_max,
    cuts_summary,
    extra_paths=None,
    predictions_path=None,
    png_dir=None,
):
    """Write a markdown file with everything needed to reproduce the report."""
    md_path = Path(md_path)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parent.parent
    git = _git_info(str(repo_root))
    now = datetime.datetime.now().isoformat(timespec="seconds")

    cmd = "python " + " ".join(shlex.quote(a) for a in sys.argv)
    extra_paths = extra_paths or {}

    files = {"checkpoint": _file_meta(args.checkpoint)}
    if not args.skip_model_comparisons:
        for label, path in [
            ("no_zmirror_checkpoint", args.no_zmirror_checkpoint),
            ("zmirror_checkpoint", args.zmirror_checkpoint),
            ("da_checkpoint", args.da_checkpoint),
            ("tres_checkpoint", args.tres_checkpoint),
        ]:
            if path:
                files[label] = _file_meta(path)
    files["mc_data"] = _file_meta(args.mc_data_path)
    files["exp_data"] = _file_meta(EXP_DATA)
    for k, v in extra_paths.items():
        files[k] = _file_meta(v)

    env = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }

    lines = []
    lines.append(f"# Report reproducibility — {Path(args.output).stem}\n")
    lines.append(f"_Generated {now}_\n")

    lines.append("## Git\n")
    if git["commit"]:
        lines.append(f"- commit: `{git['commit']}`")
        lines.append(f"- branch: `{git['branch']}`")
        lines.append(f"- dirty: **{git['dirty']}**")
        pipeline_status = git.get("status_filtered")
        if pipeline_status:
            lines.append("\n<details><summary>git status (pipeline files only)</summary>\n\n```")
            lines.append(pipeline_status)
            lines.append("```\n</details>\n")
    else:
        lines.append("- (git info unavailable)")
    lines.append("")

    lines.append("## Command\n")
    lines.append("```bash")
    lines.append(f"cd {repo_root}")
    cuda_env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_env is not None:
        lines.append(f"CUDA_VISIBLE_DEVICES={cuda_env} \\")
    lines.append(cmd)
    lines.append("```\n")

    lines.append("## Settings\n")
    lines.append(f"- signal definition: `{args.signal_definition}` ({signal_desc})")
    lines.append(f"- threshold: `{args.threshold}`")
    lines.append(f"- mc-split: `{args.mc_split}`, mc-events-per-type: `{mc_per_type}`")
    lines.append(f"- exp-split: `{args.exp_split}`, exp-max-events: `{exp_max}`")
    lines.append(f"- model: hs={args.hs}, dff={args.dff}, type={args.model_type}")
    lines.append(f"- threshold-points: {args.threshold_points}")
    lines.append("")

    lines.append("## Counts\n")
    lines.append(f"- MC events loaded: **{n_mc}**")
    lines.append(f"- EXP events loaded: **{n_exp}**")
    if cuts_summary:
        lines.append("")
        lines.append("| cut | MC muon kept | EXP kept |")
        lines.append("|---|---|---|")
        for (mh, ms), (mc_keep, ex_keep) in cuts_summary.items():
            lines.append(f"| ≥{mh}h ≥{ms}s @ thr={args.threshold} | {mc_keep} | {ex_keep} |")
    lines.append("")

    lines.append("## Files\n")
    for label, meta in files.items():
        if not meta.get("exists"):
            lines.append(f"- **{label}** — `{meta['path']}` (missing)")
            continue
        lines.append(
            f"- **{label}** — `{meta['path']}`  "
            f"({meta['size_bytes']:,} B, mtime {meta['mtime']}, sha1 `{meta['sha1'][:12]}…`)"
        )
    lines.append("")

    if predictions_path:
        lines.append("## Predictions\n")
        lines.append(f"- saved to `{predictions_path}` (load with `np.load(path)`)")
        lines.append(
            "- arrays: `mc_ev_starts`, `mc_ev_type`, `mc_energy`, `mc_probs`, "
            "`mc_true_sig`, `mc_t_res`, `mc_channels`, "
            "`exp_ev_starts`, `exp_probs`, `exp_channels` (and optionally `tres_true`, `tres_pred`)"
        )
        lines.append("")
    if png_dir:
        lines.append("## Outputs\n")
        lines.append(f"- PNG directory: `{png_dir}`")
        lines.append(f"- PDF (if requested): `{args.output}`")
        lines.append("")

    lines.append("## Environment\n")
    lines.append("```json")
    lines.append(json.dumps(env, indent=2))
    lines.append("```")

    md_path.write_text("\n".join(lines))
    print(f"[REPRO] Wrote reproducibility info: {md_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-events",
        type=int,
        default=10_000,
        help="Deprecated: fallback default for mc-events-per-type and exp-max-events",
    )
    parser.add_argument(
        "--mc-events-per-type",
        type=int,
        default=None,
        help="Max MC events per event-type to load (should be large enough so >=10k survive cuts)",
    )
    parser.add_argument("--exp-max-events", type=int, default=None, help="Max EXP events to load")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--mc-split", default="val")
    parser.add_argument("--exp-split", default="train")
    parser.add_argument("--output", default="plots/report.pdf")
    parser.add_argument(
        "--png-output-dir",
        default=None,
        help="Directory where standalone matplotlib PNGs are written. "
        "Defaults to <output-stem>_pngs alongside the report.",
    )
    parser.add_argument("--html-output", default=None)
    parser.add_argument("--report-format", choices=["html", "pdf", "both"], default="html")
    parser.add_argument(
        "--threshold-points",
        type=int,
        default=DEFAULT_THRESHOLD_POINTS,
        help="Number of threshold values for precision/recall curves; lower is faster for macro event metrics",
    )
    parser.add_argument(
        "--no-save-individual-figures",
        action="store_true",
        help="Do not save standalone HTML/PNG files for each Plotly figure",
    )
    parser.add_argument("--checkpoint", default=CKPT)
    parser.add_argument("--no-zmirror-checkpoint", default=NO_ZM_CKPT)
    parser.add_argument("--zmirror-checkpoint", default=ZM_CKPT)
    parser.add_argument("--da-checkpoint", default=DA_CKPT)
    parser.add_argument(
        "--embedding-left-checkpoint",
        default=None,
        help="Optional left model for embedding UMAP comparison; defaults to no-zmirror checkpoint",
    )
    parser.add_argument(
        "--embedding-right-checkpoint",
        default=None,
        help="Optional right model for embedding UMAP comparison; defaults to DA checkpoint",
    )
    parser.add_argument("--embedding-left-label", default="No DA")
    parser.add_argument("--embedding-right-label", default="DA")
    parser.add_argument(
        "--embedding-third-checkpoint",
        default=None,
        help="Optional third model for embedding UMAP comparison; defaults to z-mirror checkpoint",
    )
    parser.add_argument("--embedding-third-label", default="No DA + z-mirror")
    parser.add_argument(
        "--embedding-third-model-type", default="encoder", choices=["encoder", "encoder_da"]
    )
    parser.add_argument(
        "--embedding-left-model-type", default="encoder", choices=["encoder", "encoder_da"]
    )
    parser.add_argument(
        "--embedding-right-model-type", default="encoder_da", choices=["encoder", "encoder_da"]
    )
    parser.add_argument("--tres-checkpoint", default=TRES_CKPT)
    parser.add_argument(
        "--skip-model-comparisons",
        action="store_true",
        help="Skip z-mirror/no-z-mirror score comparison and DA/no-DA UMAP sections",
    )
    parser.add_argument("--embedding-max-events", type=int, default=5000)
    parser.add_argument("--model-type", default="encoder", choices=["encoder", "encoder_da"])
    parser.add_argument("--model-name", default=None, help="Display name for title")
    parser.add_argument("--mc-data-path", default=MC_DEFAULT, help="MC h5 file for inference")
    parser.add_argument("--hs", type=int, default=128)
    parser.add_argument("--dff", type=int, default=512)
    parser.add_argument(
        "--signal-definition",
        default="label_nonzero",
        choices=["tres", "tres_or_labels", "label_nonzero"],
        help=(
            "GT signal definition: |t_res|<10 (tres), |t_res|<10 OR |label|!=0 "
            "(tres_or_labels), or |label|!=0 (label_nonzero)"
        ),
    )
    args = parser.parse_args()

    mc_per_type = args.mc_events_per_type or args.max_events
    exp_max = args.exp_max_events or args.max_events

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    html_path = Path(args.html_output) if args.html_output else out_path.with_suffix(".html")
    if args.report_format in {"html", "both"}:
        html_path.parent.mkdir(parents=True, exist_ok=True)

    exp_mean, exp_std = load_norm_params(EXP_DATA)
    mc_mean, mc_std = load_norm_params(args.mc_data_path)

    def renorm_exp(data):
        return renormalize(data, exp_mean, exp_std, mc_mean, mc_std)

    signal_desc = "|t_res| < 10 ns"
    if args.signal_definition == "tres_or_labels":
        signal_desc = "|t_res| < 10 ns OR |label| > 0"
    elif args.signal_definition == "label_nonzero":
        signal_desc = "|label| != 0"
    load_labels = args.signal_definition in {"tres_or_labels", "label_nonzero"}

    print(
        f"Loading model from {args.checkpoint} (type={args.model_type}, hs={args.hs}, dff={args.dff})..."
    )
    print(f"Signal definition: {signal_desc}")
    model = load_model(args.checkpoint, hs=args.hs, dff=args.dff, model_type=args.model_type)

    print(f"Loading MC events ({args.mc_split}, max {mc_per_type} per type)...")
    mc_batches = list(
        load_mc_events(args.mc_data_path, args.mc_split, mc_per_type, load_labels=load_labels)
    )
    n_mc = sum(x.shape[0] for x, *_ in mc_batches)
    print(f"  {n_mc} MC events total")

    print(f"Loading EXP events ({args.exp_split}, max {exp_max})...")
    exp_batches = list(load_exp_events(EXP_DATA, args.exp_split, exp_max, renorm_fn=renorm_exp))
    n_exp = sum(x.shape[0] for x, _, _, _ in exp_batches)
    print(f"  {n_exp} EXP events")

    print("Inference on MC...")
    mc_events = infer_mc(model, iter(mc_batches), signal_definition=args.signal_definition)
    print("Inference on EXP...")
    exp_events = infer_exp(model, iter(exp_batches))
    del model
    torch.cuda.empty_cache()

    mc_muon = [e for e in mc_events if e["ev_type"] == "muatm"]
    print(f"\nMC total: {len(mc_events)}, muon: {len(mc_muon)}")
    print(f"EXP total: {len(exp_events)}")

    cuts = [(0, 0), (8, 2)]
    thr = args.threshold

    model_comparison = None
    embedding_models = None
    tres_comparison = None
    tres_prediction = None
    if not args.skip_model_comparisons:
        embedding_left_path = Path(args.embedding_left_checkpoint or args.no_zmirror_checkpoint)
        embedding_right_path = Path(args.embedding_right_checkpoint or args.da_checkpoint)
        if embedding_left_path.exists() and embedding_right_path.exists():
            print("\nPreparing embedding comparison models...")
            embedding_model_items = []
            for label, path, model_type in [
                (args.embedding_left_label, embedding_left_path, args.embedding_left_model_type),
                (args.embedding_right_label, embedding_right_path, args.embedding_right_model_type),
            ]:
                embedding_model_items.append(
                    (
                        label,
                        load_model(
                            str(path),
                            hs=args.hs,
                            dff=args.dff,
                            model_type=model_type,
                            da_kwargs={
                                "domain_classifier_hidden_size": 128,
                                "domain_classifier_layers": 2,
                            }
                            if model_type == "encoder_da"
                            else None,
                        ),
                    )
                )
            embedding_models = tuple(embedding_model_items)
        else:
            print(
                "Skipping embedding comparison; missing: "
                f"{embedding_left_path} or {embedding_right_path}"
            )

        tres_path = Path(args.tres_checkpoint)
        if tres_path.exists():
            print("\nRunning classification-only vs classification+t_res comparison...")
            tres_model = load_model(
                str(tres_path), hs=args.hs, dff=args.dff, model_type="encoder", out_size=3
            )
            tres_events, true_tres, pred_tres = infer_tres_model(
                tres_model, iter(mc_batches), signal_definition=args.signal_definition
            )
            tres_comparison = (mc_events, tres_events)
            tres_prediction = (true_tres, pred_tres)
            del tres_model
            torch.cuda.empty_cache()
        else:
            print(f"Skipping cls+t_res comparison; missing: {tres_path}")

    cuts_summary = {}
    for mh, ms in cuts:
        mc_keep = sum(
            1
            for e in mc_muon
            if ((e["probs"] > thr).sum() >= mh)
            and (len(np.unique(e["channels"][e["probs"] > thr] // 36)) >= ms)
        )
        ex_keep = sum(
            1
            for e in exp_events
            if ((e["probs"] > thr).sum() >= mh)
            and (len(np.unique(e["channels"][e["probs"] > thr] // 36)) >= ms)
        )
        cuts_summary[(mh, ms)] = (mc_keep, ex_keep)
        print(f"  After cut ≥{mh}h ≥{ms}s (@thr={thr}): MC muon={mc_keep}, EXP={ex_keep}")
        if (mh, ms) == (8, 2) and (mc_keep < 10_000 or ex_keep < 10_000):
            print(
                "  [WARN] Low statistics after 8-2 cut. For publication plots consider "
                "using --mc-split train with larger --mc-events-per-type and "
                "--exp-max-events, or a lower diagnostic threshold.",
                flush=True,
            )

    print("  8-2 selected-event counts by threshold:", flush=True)
    for diag_thr in (0.2, 0.3, 0.4, 0.5):
        mc_keep = sum(
            1
            for e in mc_muon
            if ((e["probs"] > diag_thr).sum() >= 8)
            and (len(np.unique(e["channels"][e["probs"] > diag_thr] // 36)) >= 2)
        )
        ex_keep = sum(
            1
            for e in exp_events
            if ((e["probs"] > diag_thr).sum() >= 8)
            and (len(np.unique(e["channels"][e["probs"] > diag_thr] // 36)) >= 2)
        )
        print(f"    threshold={diag_thr:.1f}: MC muon={mc_keep}, EXP={ex_keep}", flush=True)

    model_name = args.model_name or Path(args.checkpoint).parent.name
    if args.png_output_dir:
        png_dir = Path(args.png_output_dir)
    else:
        png_dir = out_path.with_name(out_path.stem + "_pngs")
    print(f"Writing matplotlib PNGs to {png_dir}")
    write_matplotlib_pngs(
        png_dir,
        mc_events,
        mc_muon,
        exp_events,
        thr,
        tres_prediction=tres_prediction,
        embedding_models=embedding_models,
        mc_batches=mc_batches,
        exp_batches=exp_batches,
        embedding_max_events=args.embedding_max_events,
    )
    print(f"Matplotlib PNGs written to {png_dir}")

    predictions_path = Path(png_dir) / "predictions.npz"
    save_predictions(predictions_path, mc_events, exp_events, tres_prediction)
    write_repro_md(
        Path(png_dir) / "report_info.md",
        args,
        signal_desc=signal_desc,
        n_mc=len(mc_events),
        n_exp=len(exp_events),
        mc_per_type=mc_per_type,
        exp_max=exp_max,
        cuts_summary=cuts_summary,
        predictions_path=predictions_path,
        png_dir=png_dir,
    )

    if args.report_format in {"html", "both"}:
        print(f"\nGenerating HTML report: {html_path}")
        write_html_report(
            html_path=html_path,
            model_name=model_name,
            mc_events=mc_events,
            mc_muon=mc_muon,
            exp_events=exp_events,
            signal_desc=signal_desc,
            threshold=thr,
            model_comparison=model_comparison,
            tres_comparison=tres_comparison,
            tres_prediction=tres_prediction,
            embedding_models=embedding_models,
            mc_batches=mc_batches,
            exp_batches=exp_batches,
            embedding_max_events=args.embedding_max_events,
            threshold_points=args.threshold_points,
            save_individual_figures=not args.no_save_individual_figures,
        )
        print(f"HTML report saved to: {html_path}")

    if args.report_format in {"pdf", "both"}:
        print(f"\nGenerating PDF report: {out_path}")
        with PdfPages(str(out_path)) as pdf:
            # Title
            add_section_page(
                pdf,
                "Noise/Signal Classification Report",
                f"Model: {model_name}  |  MC val vs EXP  |  "
                f"{len(mc_events)} MC / {len(exp_events)} EXP events  |  threshold={thr}\n"
                f"Signal definition: {signal_desc}",
            )

            # Section 1: Score distributions
            add_section_page(
                pdf,
                "1. P(signal) Score Distributions",
                "MC muon vs EXP — step histograms, log density scale",
            )
            plot_score_distributions(pdf, mc_muon, exp_events, cuts, thr)

            # Section 2: PR curves
            add_section_page(
                pdf,
                "2. Precision & Recall Curves",
                f"By event type (muatm / nuatm / nue2), signal = {signal_desc}",
            )
            plot_pr_curves(pdf, mc_events, signal_desc)
            plot_pr_zoomed(pdf, mc_events, signal_desc)
            plot_macro_event_pr_curves(pdf, mc_events, signal_desc)
            plot_micro_pr_after_event_cut(pdf, mc_events, signal_desc)
            plot_event_selection_fraction(pdf, mc_events, thresholds=(0.3, 0.5, 0.7), cuts=(8, 2))

            add_section_page(
                pdf,
                "2b. Energy and t_res Dependence",
                "Precision/recall vs primary energy and vs |t_res| for nue2",
            )
            plot_metrics_vs_energy(pdf, mc_events, thr)
            plot_tres_binned_metrics(pdf, mc_events, thr)

            # Section 3: Hit-level
            add_section_page(
                pdf,
                "3. Hit-Level Distributions",
                "Charge and Z-coordinate of signal hits — MC muon vs EXP",
            )
            plot_hit_distributions(pdf, mc_muon, exp_events, cuts, thr)

            # Section 4: Event-level
            add_section_page(
                pdf,
                "4. Event-Level Distributions",
                "Signal hit multiplicity and charge sum per event — MC muon vs EXP",
            )
            plot_event_distributions(pdf, mc_muon, exp_events, cuts, thr)

            if embedding_models is not None:
                add_section_page(
                    pdf,
                    "6. DA vs no-DA Embedding UMAP",
                    "Mean-pooled last hidden layer, MC muatm vs EXP with EXP renormalized to MC stats",
                )
                plot_da_embedding_umap(
                    pdf,
                    embedding_models[0],
                    embedding_models[1],
                    mc_batches,
                    exp_batches,
                    args.embedding_max_events,
                )

        print(f"PDF report saved to: {out_path}")

    if embedding_models is not None:
        del embedding_models
        torch.cuda.empty_cache()

    print("\nDone.")
    del mc_batches, exp_batches


if __name__ == "__main__":
    main()
