"""Round diagnostics for iterative pseudo-labeling on EXP data.

For a given checkpoint, compute:
  1. Wasserstein(P_mc_muon, P_exp) at no-cut and (8h, 2s) NN-cut.
  2. Predicted-signal-hit count histogram on MC train muon vs EXP after the
     (min_hits=8, min_strings=2) NN cut at threshold 0.5.
  3. (optional) precision / recall on MC val at thresholds {0.5, 0.73, 0.9}.

Outputs per call (under --out-dir):
  - wasserstein.png / wasserstein.json
  - signal_hits_distribution.png / signal_hits_distribution.json
  - events_after_cut.json (target, events kept vs indices scanned per stream)
  - mc_val_pr.json (if --mc-val provided)
  - summary.json with all numbers in one place

Usage:
  python scripts/iter_pseudolabel/diagnostics.py \
      --ckpt checkpoints/.../best.ckpt \
      --out-dir plots/iter_pseudolabel/round_0_base \
      --n-events 15000 \
      --after-cut-target 8000

``--after-cut-target`` (default 8000, i.e. ~5–10k): keep scanning MC muon / EXP
train events until this many pass the NN quality cut (≥8 pred. hits, ≥2 strings
@ 0.5). Saves counts to ``events_after_cut.json``.
"""

import argparse
import json
import sys
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import ks_2samp, wasserstein_distance
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from models.encoder import Encoder, EncoderDomainAdaptation, EncoderTwoHead

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MC_DEFAULT = "/home/plotnikovgp/baikal/Baikal-ML/pytorch_training/data/baikal_2020_sig-noise_mid-eq_normed.h5"
EXP_DEFAULT = "/home/plotnikovgp/baikal/Baikal-ML/pytorch_training/data/baikal_exp_filtered_norm_w_model_enc_v2_fix_q.h5"
MC_VAL_DEFAULT = (
    "/home/plotnikovgp/baikal/Baikal-ML/pytorch_training/data/baikal_2020_balanced_val_2k_each.h5"
)


def load_norm(h5_path):
    with h5py.File(h5_path, "r") as f:
        return f["norm_param/mean"][:].astype(np.float32), f["norm_param/std"][:].astype(np.float32)


def renorm(hits, src_mean, src_std, dst_mean, dst_std):
    return (hits * src_std + src_mean - dst_mean) / dst_std


class _DACls(torch.nn.Module):
    """Wraps EncoderDomainAdaptation to return only per-hit classification logits (B, N, 2)."""

    def __init__(self, da_model):
        super().__init__()
        self.da = da_model
        self.da.aggregate_output = False

    def forward(self, x, mask):
        out = self.da(x, mask)
        return out[0]  # (B, N, 2)


class _TwoHeadCls(torch.nn.Module):
    """Wraps EncoderTwoHead to return only the cls logits (B, N, 2)."""

    def __init__(self, two_head_model):
        super().__init__()
        self.m = two_head_model

    def forward(self, x, mask):
        full = self.m(x, mask)  # (B, N, 3) = [cls0, cls1, tres]
        return full[:, :, :2]  # (B, N, 2)


def _detect_model_type(sd_keys):
    """Guess model class from state dict key patterns."""
    key_set = set(sd_keys)
    if any(k.startswith("domain_classifier.") for k in key_set):
        return "da"
    if any(k.startswith("shared.") for k in key_set):
        return "twohead"
    return "encoder"


def load_encoder(ckpt_path, hidden_size=128, num_layers=5, dim_feedforward=512, model_type="auto"):
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]

    if model_type == "auto":
        model_type = _detect_model_type(sd.keys())
    print(f"  [load] detected model_type={model_type}")

    if model_type == "da":
        da = EncoderDomainAdaptation(
            in_features=5,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dim_feedforward_size=dim_feedforward,
            n_heads=1,
            out_size=2,
            dropout_p=0.0,
            aggregate_output=False,
        )
        da.load_state_dict(sd, strict=False)
        model = _DACls(da)
    elif model_type == "twohead":
        th = EncoderTwoHead(
            in_features=5,
            hidden_size=hidden_size,
            num_shared_layers=num_layers,
            num_cls_layers=0,
            num_tres_layers=0,
            dim_feedforward_size=dim_feedforward,
            n_heads=1,
            cls_out_size=2,
            tres_out_size=1,
            dropout_p=0.0,
            tres_head_hidden_size=256,
        )
        th.load_state_dict(sd, strict=False)
        model = _TwoHeadCls(th)
    else:
        model = Encoder(
            in_features=5,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dim_feedforward_size=dim_feedforward,
            n_heads=1,
            out_size=2,
            dropout_p=0.0,
        )
        model.load_state_dict(sd, strict=True)

    return model.to(DEVICE).eval()


def _batch_iter(h5_path, split, indices, batch_size, renorm_fn=None):
    with h5py.File(h5_path, "r") as f:
        ev_starts = f[f"{split}/ev_starts/data"][:]
        data_all = f[f"{split}/data/data"]
        channels_all = f[f"{split}/channels/data"]
        labels_all = f[f"{split}/labels/data"] if f"{split}/labels" in f else None

        for bs_start in range(0, len(indices), batch_size):
            batch_idx = indices[bs_start : bs_start + batch_size]
            hits_list, ch_list, lab_list = [], [], []
            for idx in batch_idx:
                s, e = int(ev_starts[idx]), int(ev_starts[idx + 1])
                hits = data_all[s:e].astype(np.float32)
                if renorm_fn is not None:
                    hits = renorm_fn(hits)
                hits_list.append(hits)
                ch_list.append(channels_all[s:e])
                if labels_all is not None:
                    lab_list.append(labels_all[s:e])

            max_len = max(len(h) for h in hits_list)
            bs = len(hits_list)
            x = np.zeros((bs, max_len, 5), dtype=np.float32)
            mask = np.zeros((bs, max_len), dtype=np.float32)
            ch = np.zeros((bs, max_len), dtype=np.int32)
            lab = np.zeros((bs, max_len), dtype=np.int32) if lab_list else None
            for i, h in enumerate(hits_list):
                L = len(h)
                x[i, :L] = h
                mask[i, :L] = 1.0
                ch[i, :L] = ch_list[i]
                if lab is not None:
                    lab[i, :L] = lab_list[i]
            yield (
                torch.tensor(x),
                torch.tensor(mask),
                ch,
                lab,
            )


def infer_events(model, h5_path, split, indices, renorm_fn=None, batch_size=128):
    """Return list of dicts with per-hit probs, channels and (if available) hard labels."""
    events = []
    with torch.no_grad():
        for x, mask, ch, lab in tqdm(
            _batch_iter(h5_path, split, indices, batch_size, renorm_fn),
            total=(len(indices) + batch_size - 1) // batch_size,
            desc=f"Inference {Path(h5_path).name}:{split}",
        ):
            out = model(x.to(DEVICE), mask.to(DEVICE).bool())
            if isinstance(out, tuple):
                out = out[0]
            probs = torch.sigmoid(out[:, :, 1]).cpu().numpy()
            mask_np = mask.numpy().astype(bool)
            for i in range(x.shape[0]):
                m = mask_np[i]
                ev = {"probs": probs[i][m], "channels": ch[i][m]}
                if lab is not None:
                    ev["labels"] = lab[i][m]
                events.append(ev)
    return events


def event_passes_cut(ev, threshold, min_hits, min_strings):
    sig = ev["probs"] > threshold
    n_sig = int(sig.sum())
    if n_sig < min_hits:
        return False
    n_str = len(np.unique(ev["channels"][sig] // 36))
    return n_str >= min_strings


def _select_muatm_indices(h5_path, split, max_events):
    with h5py.File(h5_path, "r") as f:
        ev_ids = f[f"{split}/ev_ids/data"][:]
    muatm_idx = np.array(
        [i for i, eid in enumerate(ev_ids) if eid.decode().startswith("muatm")],
        dtype=np.int64,
    )
    if max_events and len(muatm_idx) > max_events:
        muatm_idx = muatm_idx[:max_events]
    return muatm_idx


def _all_muatm_indices(h5_path, split):
    """All train/val event indices whose id starts with muatm (no cap)."""
    with h5py.File(h5_path, "r") as f:
        ev_ids = f[f"{split}/ev_ids/data"][:]
    return np.array(
        [i for i, eid in enumerate(ev_ids) if eid.decode().startswith("muatm")],
        dtype=np.int64,
    )


def _select_all_indices(h5_path, split, max_events):
    with h5py.File(h5_path, "r") as f:
        n = len(f[f"{split}/ev_starts/data"]) - 1
    if max_events and n > max_events:
        n = max_events
    return np.arange(n, dtype=np.int64)


def _all_event_indices(h5_path, split):
    with h5py.File(h5_path, "r") as f:
        n = len(f[f"{split}/ev_starts/data"]) - 1
    return np.arange(n, dtype=np.int64)


def infer_events_collect_after_cut(
    model,
    h5_path,
    split,
    index_pool,
    renorm_fn,
    batch_size,
    thr,
    min_hits,
    min_strings,
    target_count,
    chunk_indices=4096,
    max_indices_to_scan=None,
    desc="after-cut",
):
    """Scan index_pool in chunks; keep events that pass the NN cut until target_count.

    Returns:
        kept_events: list of length min(target_count, available), each dict with probs/channels
        n_scanned: number of event indices from index_pool that were run through the model
    """
    kept = []
    n_scanned = 0
    i = 0
    pbar = tqdm(total=target_count, desc=desc, unit="evt")
    while len(kept) < target_count and i < len(index_pool):
        if max_indices_to_scan is not None and n_scanned >= max_indices_to_scan:
            break
        chunk = index_pool[i : i + chunk_indices]
        if len(chunk) == 0:
            break
        evs = infer_events(model, h5_path, split, chunk, renorm_fn=renorm_fn, batch_size=batch_size)
        n_scanned += len(chunk)
        i += chunk_indices
        added = 0
        for e in evs:
            if event_passes_cut(e, thr, min_hits, min_strings):
                kept.append(e)
                added += 1
                if len(kept) >= target_count:
                    break
        pbar.update(added)
        if len(kept) >= target_count:
            break
    pbar.close()
    return kept[:target_count], n_scanned


def compute_wasserstein_two_sample(mc_probs_1d, exp_probs_1d, label, n_mc, n_exp):
    if mc_probs_1d.size == 0 or exp_probs_1d.size == 0:
        return {label: {"wasserstein": None, "n_mc": int(n_mc), "n_exp": int(n_exp)}}
    w = float(wasserstein_distance(mc_probs_1d, exp_probs_1d))
    return {label: {"wasserstein": w, "n_mc": int(n_mc), "n_exp": int(n_exp)}}


def plot_wasserstein(w_results, out_path, ckpt_label):
    cuts = list(w_results.keys())
    vals = [
        w_results[c]["wasserstein"] if w_results[c]["wasserstein"] is not None else 0.0
        for c in cuts
    ]
    fig, ax = plt.subplots(figsize=(7.5, 5.2), constrained_layout=True)
    x = np.arange(len(cuts))
    bars = ax.bar(x, vals, 0.55, color="#1f77b4", edgecolor="black", linewidth=0.8)
    for bar, v, c in zip(bars, vals, cuts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v,
            f"{v:.4f}\nn_mc={w_results[c]['n_mc']}\nn_exp={w_results[c]['n_exp']}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#1f77b4",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(cuts)
    ax.set_xlabel("Event quality cut (min predicted hits, min strings)")
    ax.set_ylabel("Wasserstein distance")
    ax.set_title(f"MC muon vs EXP — P(signal) alignment\n{ckpt_label}", fontsize=12)
    top = max(vals) if vals else 1.0
    ax.set_ylim(0, top * 1.4 + 1e-6)
    ax.yaxis.grid(True, alpha=0.35)
    ax.set_axisbelow(True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_signal_hit_distribution(mc_n, ex_n, out_path, ckpt_label, threshold=0.5, cut=(8, 2)):
    if not mc_n or not ex_n:
        fig, ax = plt.subplots(figsize=(7.5, 5.2))
        ax.text(0.5, 0.5, "Insufficient events after cut", ha="center", va="center")
        fig.savefig(out_path, dpi=130)
        plt.close(fig)
        return

    mc_arr = np.asarray(mc_n, dtype=np.float64)
    ex_arr = np.asarray(ex_n, dtype=np.float64)
    hi = int(max(mc_arr.max(), ex_arr.max())) + 2
    bins = np.arange(0, hi + 1) - 0.5  # bin centers on integers 0..hi-1

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11.5, 5.0),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.35, 0.65]},
    )
    ax_hist, ax_box = axes

    # Left: overlaid density histograms (filled for readability)
    ax_hist.hist(
        mc_arr,
        bins=bins,
        density=True,
        alpha=0.45,
        color="#2ca02c",
        edgecolor="#1a6b14",
        linewidth=0.6,
        label=f"MC train muon (n={len(mc_n)})",
    )
    ax_hist.hist(
        ex_arr,
        bins=bins,
        density=True,
        alpha=0.42,
        color="#1f77b4",
        edgecolor="#0d3d63",
        linewidth=0.6,
        label=f"EXP (n={len(ex_n)})",
    )
    mc_mean, ex_mean = float(mc_arr.mean()), float(ex_arr.mean())
    ax_hist.axvline(
        mc_mean, color="#2ca02c", ls="--", lw=2.0, alpha=0.9, label=f"MC mean = {mc_mean:.1f}"
    )
    ax_hist.axvline(
        ex_mean, color="#1f77b4", ls="--", lw=2.0, alpha=0.9, label=f"EXP mean = {ex_mean:.1f}"
    )
    ax_hist.set_xlabel(f"Predicted signal hits per event (@ P(signal) > {threshold})")
    ax_hist.set_ylabel("Density")
    ax_hist.set_title(
        f"After NN event cut: ≥{cut[0]} hits & ≥{cut[1]} strings",
        fontsize=12,
    )
    ax_hist.legend(loc="upper right", fontsize=9)
    ax_hist.grid(True, axis="y", alpha=0.35)
    ax_hist.set_xlim(left=-0.5)

    # Right: boxplots for compact comparison
    bp = ax_box.boxplot(
        [mc_arr, ex_arr],
        labels=["MC\nmuatm", "EXP"],
        patch_artist=True,
        medianprops=dict(color="black", linewidth=1.6),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
    )
    colors = ["#b3e0a8", "#aecce8"]
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.85)
    ax_box.set_ylabel("Hits / event")
    ax_box.set_title("Summary", fontsize=11)
    ax_box.grid(True, axis="y", alpha=0.35)
    summary = (
        f"MC:  mean {mc_mean:.1f}, med {float(np.median(mc_arr)):.0f}\n"
        f"EXP: mean {ex_mean:.1f}, med {float(np.median(ex_arr)):.0f}"
    )
    ax_box.text(
        0.02,
        0.98,
        summary,
        transform=ax_box.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.85, edgecolor="0.5"),
    )

    fig.suptitle(f"Signal hits per event — {ckpt_label}", fontsize=13, y=1.02)
    fig.savefig(out_path, dpi=145, bbox_inches="tight")
    plt.close(fig)


def mc_val_pr(mc_val_evs, thresholds=(0.5, 0.73, 0.9)):
    """Per-hit precision/recall on MC val (uses MC original labels != 0 as truth)."""
    if not mc_val_evs:
        return {}
    have_labels = "labels" in mc_val_evs[0]
    if not have_labels:
        return {}
    probs = np.concatenate([e["probs"] for e in mc_val_evs])
    true = np.concatenate([e["labels"] for e in mc_val_evs]) != 0
    out = {}
    for t in thresholds:
        pred = probs > t
        tp = int((pred & true).sum())
        fp = int((pred & ~true).sum())
        fn = int((~pred & true).sum())
        precision = tp / (tp + fp) if (tp + fp) else float("nan")
        recall = tp / (tp + fn) if (tp + fn) else float("nan")
        out[f"thr={t}"] = {
            "precision": precision,
            "recall": recall,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
    return out


def concat_hit_probs(events):
    if not events:
        return np.array([], dtype=np.float64)
    return np.concatenate([e["probs"] for e in events]).astype(np.float64)


def signal_hit_counts_preselected(events_after_cut, threshold=0.5):
    """Events already passed the quality cut; count pred. signal hits per event."""
    return [int((e["probs"] > threshold).sum()) for e in events_after_cut]


def ks_signal_hits(mc_counts, exp_counts):
    """KS test on signal-hit-per-event distributions (MC vs EXP, after cut)."""
    if not mc_counts or not exp_counts:
        return {"ks_stat": None, "ks_pvalue": None, "n_mc": 0, "n_exp": 0}
    mc = np.asarray(mc_counts, dtype=np.float64)
    exp = np.asarray(exp_counts, dtype=np.float64)
    stat, p = ks_2samp(mc, exp)
    return {
        "ks_stat": float(stat),
        "ks_pvalue": float(p),
        "n_mc": len(mc),
        "n_exp": len(exp),
        "mc_mean": float(mc.mean()),
        "exp_mean": float(exp.mean()),
        "delta_mean": float(exp.mean() - mc.mean()),
    }


def selection_efficiency(after_cut_report):
    """Fraction of events passing the NN quality cut."""
    out = {}
    for key in ("mc_muatm", "exp"):
        sec = after_cut_report.get(key, {})
        kept = sec.get("events_after_cut", 0)
        scanned = sec.get("event_indices_scanned", 1)
        out[key] = {
            "events_kept": kept,
            "events_scanned": scanned,
            "efficiency_pct": round(kept / max(scanned, 1) * 100, 2),
        }
    mc_eff = out["mc_muatm"]["efficiency_pct"]
    exp_eff = out["exp"]["efficiency_pct"]
    out["mc_to_exp_ratio"] = round(mc_eff / max(exp_eff, 1e-6), 2)
    return out


def signal_region_ks(events_after_cut_mc, events_after_cut_exp, p_min=0.1):
    """KS test on per-hit P(signal) restricted to signal-region hits (P > p_min).

    This focuses on how the model shapes the probability distribution
    for genuine signal candidates, excluding the trivially-classified noise floor.
    """
    mc_probs = concat_hit_probs(events_after_cut_mc)
    exp_probs = concat_hit_probs(events_after_cut_exp)
    mc_sig = mc_probs[mc_probs > p_min]
    exp_sig = exp_probs[exp_probs > p_min]
    if mc_sig.size == 0 or exp_sig.size == 0:
        return {"ks_stat": None, "ks_pvalue": None, "p_min": p_min, "n_mc_hits": 0, "n_exp_hits": 0}
    stat, p = ks_2samp(mc_sig, exp_sig)
    return {
        "ks_stat": float(stat),
        "ks_pvalue": float(p),
        "p_min": p_min,
        "n_mc_hits": int(mc_sig.size),
        "n_exp_hits": int(exp_sig.size),
        "mc_mean": float(mc_sig.mean()),
        "exp_mean": float(exp_sig.mean()),
        "wasserstein": float(wasserstein_distance(mc_sig, exp_sig)),
    }


def prob_bimodality(events_after_cut, label="mc"):
    """Characterize the shape of P(signal) distribution for after-cut events.

    Splits hit probs into three zones:
      noise:  [0, 0.1]
      ambig:  (0.1, 0.9)
      signal: [0.9, 1.0]
    Returns fractions and mean P in each zone.
    """
    probs = concat_hit_probs(events_after_cut)
    if probs.size == 0:
        return {"label": label, "n_hits": 0}
    noise_mask = probs <= 0.1
    signal_mask = probs >= 0.9
    ambig_mask = ~noise_mask & ~signal_mask
    n = probs.size
    return {
        "label": label,
        "n_hits": int(n),
        "frac_noise": round(float(noise_mask.sum()) / n, 4),
        "frac_ambig": round(float(ambig_mask.sum()) / n, 4),
        "frac_signal": round(float(signal_mask.sum()) / n, 4),
        "mean_all": round(float(probs.mean()), 4),
        "mean_noise_zone": round(float(probs[noise_mask].mean()), 4) if noise_mask.any() else None,
        "mean_signal_zone": round(float(probs[signal_mask].mean()), 4)
        if signal_mask.any()
        else None,
    }


def plot_prob_distribution(mc_after, exp_after, out_path, ckpt_label, p_min=0.1):
    """Per-hit P(signal) distribution for after-cut events, MC vs EXP."""
    mc_probs = concat_hit_probs(mc_after)
    exp_probs = concat_hit_probs(exp_after)
    if mc_probs.size == 0 or exp_probs.size == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    bins_full = np.linspace(0, 1, 101)
    for ax, title, lo, hi, bins in [
        (axes[0], "Full range [0, 1]", 0, 1, bins_full),
        (axes[1], f"Signal region ({p_min}, 1]", p_min, 1, np.linspace(p_min, 1, 91)),
    ]:
        mc_sel = mc_probs[(mc_probs >= lo)] if lo == 0 else mc_probs[mc_probs > lo]
        exp_sel = exp_probs[(exp_probs >= lo)] if lo == 0 else exp_probs[exp_probs > lo]
        ax.hist(
            mc_sel,
            bins=bins,
            density=True,
            alpha=0.45,
            color="#2ca02c",
            edgecolor="#1a6b14",
            linewidth=0.4,
            label=f"MC ({mc_sel.size} hits)",
        )
        ax.hist(
            exp_sel,
            bins=bins,
            density=True,
            alpha=0.42,
            color="#1f77b4",
            edgecolor="#0d3d63",
            linewidth=0.4,
            label=f"EXP ({exp_sel.size} hits)",
        )
        ax.set_xlabel("P(signal)")
        ax.set_ylabel("Density")
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(f"Per-hit P(signal) — after NN cut — {ckpt_label}", fontsize=12, y=1.01)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--mc", default=MC_DEFAULT)
    ap.add_argument("--exp", default=EXP_DEFAULT)
    ap.add_argument("--mc-val", default=MC_VAL_DEFAULT, help="optional labeled MC val h5 for PR")
    ap.add_argument("--mc-split", default="train")
    ap.add_argument("--exp-split", default="train")
    ap.add_argument(
        "--n-events",
        type=int,
        default=15000,
        help="MC muon + EXP events to infer for no-cut Wasserstein (hit-level)",
    )
    ap.add_argument(
        "--after-cut-target",
        type=int,
        default=8000,
        help="Keep scanning until this many events per stream pass the (8,2) NN cut (~5–10k)",
    )
    ap.add_argument(
        "--max-indices-scan",
        type=int,
        default=2_000_000,
        help="Safety cap: max event indices to scan per stream when filling after-cut sample",
    )
    ap.add_argument(
        "--chunk-indices", type=int, default=4096, help="Index chunk size for streaming"
    )
    ap.add_argument(
        "--cut-threshold", type=float, default=0.5, help="P(signal) threshold for cut + histogram"
    )
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--hidden-size", type=int, default=128)
    ap.add_argument("--num-layers", type=int, default=5)
    ap.add_argument("--dim-feedforward", type=int, default=512)
    ap.add_argument("--ckpt-label", default=None)
    args = ap.parse_args()

    thr = args.cut_threshold
    mh, ms = 8, 2
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    label = args.ckpt_label or args.ckpt

    model = load_encoder(args.ckpt, args.hidden_size, args.num_layers, args.dim_feedforward)

    mc_mean, mc_std = load_norm(args.mc)
    exp_mean, exp_std = load_norm(args.exp)

    def exp_renorm(h):
        return renorm(h, exp_mean, exp_std, mc_mean, mc_std)

    # --- Pool A: fixed-size sample for no-cut Wasserstein (all hit probs) ---
    print(f"[diag] No-cut sample: MC muon + EXP, up to {args.n_events} events each")
    mc_idx_a = _select_muatm_indices(args.mc, args.mc_split, args.n_events)
    exp_idx_a = _select_all_indices(args.exp, args.exp_split, args.n_events)
    print(f"  MC muatm indices: {len(mc_idx_a)}, EXP indices: {len(exp_idx_a)}")
    mc_events_nocut = infer_events(
        model, args.mc, args.mc_split, mc_idx_a, renorm_fn=None, batch_size=args.batch_size
    )
    exp_events_nocut = infer_events(
        model, args.exp, args.exp_split, exp_idx_a, renorm_fn=exp_renorm, batch_size=args.batch_size
    )

    # --- Pool B: stream until after-cut-target events per stream ---
    tgt = args.after_cut_target
    print(
        f"[diag] Collecting ≥{tgt} events after NN cut (≥{mh} pred. hits, ≥{ms} strings @ {thr})..."
    )
    mc_pool = _all_muatm_indices(args.mc, args.mc_split)
    exp_pool = _all_event_indices(args.exp, args.exp_split)
    print(f"  MC muatm index pool size = {len(mc_pool)}, EXP index pool size = {len(exp_pool)}")

    mc_after, mc_scanned = infer_events_collect_after_cut(
        model,
        args.mc,
        args.mc_split,
        mc_pool,
        renorm_fn=None,
        batch_size=args.batch_size,
        thr=thr,
        min_hits=mh,
        min_strings=ms,
        target_count=tgt,
        chunk_indices=args.chunk_indices,
        max_indices_to_scan=args.max_indices_scan,
        desc="MC muatm after-cut",
    )
    exp_after, exp_scanned = infer_events_collect_after_cut(
        model,
        args.exp,
        args.exp_split,
        exp_pool,
        renorm_fn=exp_renorm,
        batch_size=args.batch_size,
        thr=thr,
        min_hits=mh,
        min_strings=ms,
        target_count=tgt,
        chunk_indices=args.chunk_indices,
        max_indices_to_scan=args.max_indices_scan,
        desc="EXP after-cut",
    )

    def _stopped_reason(n_kept, n_scanned, pool_len):
        if n_kept >= tgt:
            return "target_reached"
        if args.max_indices_scan is not None and n_scanned >= args.max_indices_scan:
            return "max_indices_scan"
        if n_scanned >= pool_len:
            return "exhausted_index_pool"
        return "partial_unknown"

    after_cut_report = {
        "nn_cut": {
            "min_predicted_signal_hits": mh,
            "min_strings_with_pred_signal": ms,
            "probability_threshold": thr,
        },
        "target_events_after_cut_per_stream": tgt,
        "mc_muatm": {
            "events_after_cut": len(mc_after),
            "event_indices_scanned": mc_scanned,
            "muatm_index_pool_size": len(mc_pool),
            "stopped_reason": _stopped_reason(len(mc_after), mc_scanned, len(mc_pool)),
        },
        "exp": {
            "events_after_cut": len(exp_after),
            "event_indices_scanned": exp_scanned,
            "exp_index_pool_size": len(exp_pool),
            "stopped_reason": _stopped_reason(len(exp_after), exp_scanned, len(exp_pool)),
        },
    }
    (out_dir / "events_after_cut.json").write_text(json.dumps(after_cut_report, indent=2))
    print(
        f"  MC: kept {len(mc_after)}/{tgt} after cut (scanned {mc_scanned} muatm events); "
        f"EXP: kept {len(exp_after)}/{tgt} (scanned {exp_scanned} events)"
    )

    # --- Wasserstein: no_cut from pool A; (8,2) from pool B (already post-cut) ---
    print("[diag] Computing Wasserstein...")
    probs_mc_nocut = concat_hit_probs(mc_events_nocut)
    probs_exp_nocut = concat_hit_probs(exp_events_nocut)
    w1 = compute_wasserstein_two_sample(
        probs_mc_nocut,
        probs_exp_nocut,
        "no_cut",
        len(mc_events_nocut),
        len(exp_events_nocut),
    )
    probs_mc_cut = concat_hit_probs(mc_after)
    probs_exp_cut = concat_hit_probs(exp_after)
    w2 = compute_wasserstein_two_sample(
        probs_mc_cut,
        probs_exp_cut,
        f"({mh},{ms})",
        len(mc_after),
        len(exp_after),
    )
    w = {**w1, **w2}
    for k, v in w.items():
        wv = v["wasserstein"]
        print(f"  cut {k}: W = {wv:.4f}" if wv is not None else f"  cut {k}: empty")
    plot_wasserstein(w, out_dir / "wasserstein.png", label)
    (out_dir / "wasserstein.json").write_text(json.dumps(w, indent=2))

    print("[diag] Signal-hit count histogram (events already after NN cut)...")
    mc_n = signal_hit_counts_preselected(mc_after, threshold=thr)
    ex_n = signal_hit_counts_preselected(exp_after, threshold=thr)
    plot_signal_hit_distribution(
        mc_n, ex_n, out_dir / "signal_hits_distribution.png", label, threshold=thr, cut=(mh, ms)
    )
    np.savez_compressed(
        out_dir / "signal_hits_per_event.npz",
        mc=np.asarray(mc_n, dtype=np.int32),
        exp=np.asarray(ex_n, dtype=np.int32),
    )
    sh_stats = {
        "mc": {
            "n_events": len(mc_n),
            "mean": float(np.mean(mc_n)) if mc_n else None,
            "median": float(np.median(mc_n)) if mc_n else None,
            "p25": float(np.percentile(mc_n, 25)) if mc_n else None,
            "p75": float(np.percentile(mc_n, 75)) if mc_n else None,
        },
        "exp": {
            "n_events": len(ex_n),
            "mean": float(np.mean(ex_n)) if ex_n else None,
            "median": float(np.median(ex_n)) if ex_n else None,
            "p25": float(np.percentile(ex_n, 25)) if ex_n else None,
            "p75": float(np.percentile(ex_n, 75)) if ex_n else None,
        },
    }
    (out_dir / "signal_hits_distribution.json").write_text(json.dumps(sh_stats, indent=2))
    print(f"  MC mean hits = {sh_stats['mc']['mean']}, EXP mean = {sh_stats['exp']['mean']}")

    # --- NEW: KS test on signal-hit counts ---
    print("[diag] KS test on signal-hit-per-event distributions...")
    ks_hits_result = ks_signal_hits(mc_n, ex_n)
    (out_dir / "ks_signal_hits.json").write_text(json.dumps(ks_hits_result, indent=2))
    if ks_hits_result["ks_stat"] is not None:
        print(
            f"  KS stat = {ks_hits_result['ks_stat']:.4f}, p = {ks_hits_result['ks_pvalue']:.2e}, "
            f"delta_mean = {ks_hits_result['delta_mean']:+.2f}"
        )

    # --- NEW: Selection efficiency ---
    print("[diag] Selection efficiency (events passing NN cut / scanned)...")
    sel_eff = selection_efficiency(after_cut_report)
    (out_dir / "selection_efficiency.json").write_text(json.dumps(sel_eff, indent=2))
    print(
        f"  MC: {sel_eff['mc_muatm']['efficiency_pct']:.1f}%, "
        f"EXP: {sel_eff['exp']['efficiency_pct']:.1f}%, "
        f"ratio: {sel_eff['mc_to_exp_ratio']:.2f}x"
    )

    # --- NEW: KS test on P(signal) for signal-region hits (P > 0.1) ---
    print("[diag] Signal-region P(signal) KS test (P > 0.1)...")
    sr_ks = signal_region_ks(mc_after, exp_after, p_min=0.1)
    (out_dir / "signal_region_ks.json").write_text(json.dumps(sr_ks, indent=2))
    if sr_ks["ks_stat"] is not None:
        print(
            f"  KS stat = {sr_ks['ks_stat']:.4f}, p = {sr_ks['ks_pvalue']:.2e}, "
            f"W = {sr_ks['wasserstein']:.4f}"
        )

    # --- NEW: P(signal) bimodality for MC vs EXP ---
    print("[diag] P(signal) bimodality analysis...")
    bim_mc = prob_bimodality(mc_after, label="mc")
    bim_exp = prob_bimodality(exp_after, label="exp")
    bimodality = {"mc": bim_mc, "exp": bim_exp}
    (out_dir / "bimodality.json").write_text(json.dumps(bimodality, indent=2))
    print(
        f"  MC:  noise={bim_mc.get('frac_noise')}, ambig={bim_mc.get('frac_ambig')}, signal={bim_mc.get('frac_signal')}"
    )
    print(
        f"  EXP: noise={bim_exp.get('frac_noise')}, ambig={bim_exp.get('frac_ambig')}, signal={bim_exp.get('frac_signal')}"
    )

    # --- NEW: per-hit P(signal) distribution plot ---
    print("[diag] Plotting per-hit P(signal) distribution (after cut)...")
    plot_prob_distribution(mc_after, exp_after, out_dir / "prob_distribution.png", label, p_min=0.1)

    pr_results = {}
    if args.mc_val and Path(args.mc_val).exists():
        print(f"[diag] MC val PR at {args.mc_val}...")
        mc_val_n = min(2000, args.n_events)
        mc_val_idx = _select_all_indices(args.mc_val, "val", mc_val_n)
        mc_val_events = infer_events(
            model,
            args.mc_val,
            "val",
            mc_val_idx,
            renorm_fn=None,
            batch_size=args.batch_size,
        )
        pr_results = mc_val_pr(mc_val_events)
        (out_dir / "mc_val_pr.json").write_text(json.dumps(pr_results, indent=2))
        for k, v in pr_results.items():
            print(f"  {k}: P={v['precision']:.4f}  R={v['recall']:.4f}")

    summary = {
        "ckpt": str(args.ckpt),
        "label": label,
        "wasserstein": w,
        "events_after_cut": after_cut_report,
        "signal_hits_distribution": sh_stats,
        "ks_signal_hits": ks_hits_result,
        "selection_efficiency": sel_eff,
        "signal_region_ks": sr_ks,
        "bimodality": bimodality,
        "mc_val_pr": pr_results,
        "settings": {
            "n_events_nocut_wasserstein": args.n_events,
            "after_cut_target": tgt,
            "max_indices_scan": args.max_indices_scan,
            "chunk_indices": args.chunk_indices,
            "cut_threshold": thr,
            "mc_split": args.mc_split,
            "exp_split": args.exp_split,
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[diag] Done. Output in {out_dir}")


if __name__ == "__main__":
    main()
