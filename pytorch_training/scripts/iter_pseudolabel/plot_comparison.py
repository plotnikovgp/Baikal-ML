"""Side-by-side comparison plots: baseline vs hard_t0p9 r3.

All panels use the same AFTER-cut sample (≥8 pred. hits, ≥2 strings @ P>0.5).

Per column:
  1. P(signal) densities (MC muon vs EXP), log-y
  2. Ratio MC/EXP in each probability bin — normalized densities;
     error bars ~ delta method on log-ratio; dashed line at 1.0 (log-scale y).
  3. Signal hits per event — log-y

Layout: 2 columns (baseline | hard_t0p9), 3 rows.
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
from scipy.stats import ks_2samp, wasserstein_distance


def mc_exp_ratio_bins(mc_p: np.ndarray, exp_p: np.ndarray, n_bins: int = 40, eps: float = 1e-12):
    """Per-bin MC/EXP ratio of normalized fractions (sums to total hits)."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    hc_mc = np.histogram(mc_p, bins=edges)[0].astype(np.float64)
    hc_exp = np.histogram(exp_p, bins=edges)[0].astype(np.float64)
    bw = edges[1] - edges[0]
    rho_mc = hc_mc / (mc_p.size * bw)
    rho_exp = hc_exp / (exp_p.size * bw)
    # Ratio of densities ρ_mc / ρ_exp = (hc_mc/N_mc)/(hc_exp/N_exp)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = rho_mc / np.maximum(rho_exp, eps)
    centers = 0.5 * (edges[:-1] + edges[1:])
    # Rough 1σ on log-ratio: Var(log r) ~ 1/hc_mc + 1/hc_exp (hits per bin Poisson-ish)
    with np.errstate(divide="ignore", invalid="ignore"):
        vr = 1.0 / np.maximum(hc_mc, 1.0) + 1.0 / np.maximum(hc_exp, 1.0)
        log_lo = np.log(np.maximum(ratio, eps)) - np.sqrt(vr)
        log_hi = np.log(np.maximum(ratio, eps)) + np.sqrt(vr)
        y_err_lo = ratio - np.exp(log_lo)
        y_err_hi = np.exp(log_hi) - ratio
    valid = (hc_mc >= 8) & (hc_exp >= 8) & np.isfinite(ratio) & (ratio > 0)
    return centers, ratio, (y_err_lo, y_err_hi), valid


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.iter_pseudolabel.diagnostics import (  # noqa: E402
    _all_event_indices,
    _all_muatm_indices,
    concat_hit_probs,
    infer_events_collect_after_cut,
    load_encoder,
    load_norm,
    renorm,
    signal_hit_counts_preselected,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MC_DEFAULT = "/home/plotnikovgp/baikal/Baikal-ML/pytorch_training/data/baikal_2020_sig-noise_mid-eq_normed.h5"
EXP_DEFAULT = "/home/plotnikovgp/baikal/Baikal-ML/pytorch_training/data/baikal_exp_filtered_norm_w_model_enc_v2_fix_q.h5"


def collect(model, mc_path, exp_path, target=8000, thr=0.5, batch_size=128):
    mc_mean, mc_std = load_norm(mc_path)
    exp_mean, exp_std = load_norm(exp_path)

    def exp_renorm(h):
        return renorm(h, exp_mean, exp_std, mc_mean, mc_std)

    mc_pool = _all_muatm_indices(mc_path, "train")
    exp_pool = _all_event_indices(exp_path, "train")

    mc_evs, _ = infer_events_collect_after_cut(
        model,
        mc_path,
        "train",
        mc_pool,
        renorm_fn=None,
        batch_size=batch_size,
        thr=thr,
        min_hits=8,
        min_strings=2,
        target_count=target,
        desc="MC muatm after-cut",
    )
    exp_evs, _ = infer_events_collect_after_cut(
        model,
        exp_path,
        "train",
        exp_pool,
        renorm_fn=exp_renorm,
        batch_size=batch_size,
        thr=thr,
        min_hits=8,
        min_strings=2,
        target_count=target,
        desc="EXP after-cut",
    )
    return mc_evs, exp_evs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-base", required=True)
    ap.add_argument("--ckpt-hard", required=True)
    ap.add_argument("--label-base", default="Baseline")
    ap.add_argument("--label-hard", default="hard_t0p9 r3")
    ap.add_argument("--out", default="plots/iter_pseudolabel/_comparison/base_vs_hard09.png")
    ap.add_argument("--target", type=int, default=8000)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument(
        "--cut-threshold",
        type=float,
        default=0.5,
        help="Probability threshold used for 8-2 event cut and signal-hit counting.",
    )
    args = ap.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    ckpts = [
        (args.ckpt_base, args.label_base),
        (args.ckpt_hard, args.label_hard),
    ]

    all_data = []
    for ckpt_path, label in ckpts:
        print(f"\n=== {label}: {ckpt_path} ===")
        model = load_encoder(ckpt_path)
        mc_evs, exp_evs = collect(
            model,
            MC_DEFAULT,
            EXP_DEFAULT,
            target=args.target,
            thr=args.cut_threshold,
            batch_size=args.batch_size,
        )
        mc_probs = concat_hit_probs(mc_evs)
        exp_probs = concat_hit_probs(exp_evs)
        mc_nhits = signal_hit_counts_preselected(mc_evs, threshold=args.cut_threshold)
        exp_nhits = signal_hit_counts_preselected(exp_evs, threshold=args.cut_threshold)
        all_data.append(
            {
                "label": label,
                "mc_probs": mc_probs,
                "exp_probs": exp_probs,
                "mc_nhits": np.array(mc_nhits),
                "exp_nhits": np.array(exp_nhits),
            }
        )
        del model
        torch.cuda.empty_cache()

    # Layout: row 0 = P(signal) (2 cols), row 1 = ratio (single), row 2 = signal hits (2 cols)
    fig = plt.figure(figsize=(14, 14), constrained_layout=True)
    gs = fig.add_gridspec(3, 2)

    COLORS_MODEL = ["#d62728", "#1f77b4"]  # red = baseline, blue = trained

    # --- Row 0: P(signal) per-model (2 columns) ---
    for col, d in enumerate(all_data):
        ax = fig.add_subplot(gs[0, col])
        mc_p, ex_p = d["mc_probs"], d["exp_probs"]
        bins = np.linspace(0, 1, 101)
        ax.hist(
            mc_p,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.5,
            color="#2ca02c",
            label=f"MC muon ({mc_p.size} hits)",
        )
        ax.hist(
            ex_p,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.5,
            color="#1f77b4",
            label=f"EXP ({ex_p.size} hits)",
        )
        ax.set_yscale("log")
        ax.set_xlabel("ξ = P(signal)")
        ax.set_ylabel("Density")
        w = wasserstein_distance(mc_p, ex_p)
        ax.set_title(f"{d['label']} — P(signal) after cut\nW = {w:.4f}", fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # --- Row 1: ρ_MC / ρ_EXP — both models on ONE panel ---
    ax_ratio = fig.add_subplot(gs[1, :])
    for idx, d in enumerate(all_data):
        mc_p, ex_p = d["mc_probs"], d["exp_probs"]
        xc, r_r, Err, ok = mc_exp_ratio_bins(mc_p, ex_p, n_bins=40)
        mask = ok & np.isfinite(r_r)
        ylo = np.clip(Err[0][mask], 0.0, r_r[mask])
        yhi = Err[1][mask]
        r_med = np.median(r_r[mask]) if mask.any() else np.nan
        # Unweighted bin-wise deviation from ideal ratio=1.0
        mad_abs_1 = np.mean(np.abs(r_r[mask] - 1.0)) if mask.any() else np.nan
        rmse_1 = np.sqrt(np.mean((r_r[mask] - 1.0) ** 2)) if mask.any() else np.nan
        offset = (idx - 0.5) * 0.004  # tiny x-offset so points don't overlap
        ax_ratio.errorbar(
            xc[mask] + offset,
            r_r[mask],
            yerr=[ylo, yhi],
            fmt="o",
            markersize=4,
            capsize=2,
            linewidth=1.0,
            color=COLORS_MODEL[idx],
            ecolor=COLORS_MODEL[idx],
            alpha=0.85,
            label=(
                f"{d['label']}  (median={r_med:.3f}, mean|r-1|={mad_abs_1:.3f}, rmse={rmse_1:.3f})"
            ),
        )
    ax_ratio.axhline(1.0, color="dimgray", ls="--", lw=1.2, alpha=0.85)
    ax_ratio.set_yscale("log")
    ax_ratio.set_xlim(0, 1)
    ax_ratio.set_xlabel("ξ = P(signal)")
    ax_ratio.set_ylabel("ρ_MC / ρ_EXP")
    ax_ratio.set_title(
        "Density ratio MC/EXP per P(signal) bin — after NN cut\n"
        "bin-wise deviation metrics are unweighted across valid bins",
        fontsize=12,
    )
    ax_ratio.legend(fontsize=10)
    ax_ratio.grid(True, which="both", alpha=0.3)

    # --- Row 2: Signal hits per event (2 columns) ---
    for col, d in enumerate(all_data):
        ax = fig.add_subplot(gs[2, col])
        mc_n, ex_n = d["mc_nhits"], d["exp_nhits"]
        hi = int(max(mc_n.max(), ex_n.max())) + 2
        bins_n = np.arange(8, hi + 1) - 0.5
        ax.hist(
            mc_n,
            bins=bins_n,
            density=True,
            histtype="step",
            linewidth=1.5,
            color="#2ca02c",
            label=f"MC (n={len(mc_n)}, μ={mc_n.mean():.1f})",
        )
        ax.hist(
            ex_n,
            bins=bins_n,
            density=True,
            histtype="step",
            linewidth=1.5,
            color="#1f77b4",
            label=f"EXP (n={len(ex_n)}, μ={ex_n.mean():.1f})",
        )
        ax.set_yscale("log")
        ax.set_xlabel("N = signal hits per event")
        ax.set_ylabel("Density")
        ks_n, _ = ks_2samp(mc_n.astype(float), ex_n.astype(float))
        delta = ex_n.mean() - mc_n.mean()
        ax.set_title(
            f"{d['label']} — signal hits/event\nKS = {ks_n:.4f}, Δμ = {delta:+.2f}", fontsize=11
        )
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"MC muon vs EXP — after NN cut (≥8 hits, ≥2 strings @ P>{args.cut_threshold:g})",
        fontsize=13,
        y=1.01,
    )
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
