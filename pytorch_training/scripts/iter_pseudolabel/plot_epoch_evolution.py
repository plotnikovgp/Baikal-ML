"""How the MC-vs-EXP P(signal) distribution evolves over fine-tuning epochs.

Motivation: on EXP data the number of hits classified as confident signal
(xi = P(signal) ~ 1) grows with pseudo-label fine-tuning. This script tracks
that effect across checkpoints (baseline + epoch 1/3/5/10/15).

For each checkpoint we collect the AFTER-cut sample (>=8 pred. hits, >=2 strings
@ P>thr) for MC muon and EXP, then plot:
  (a) EXP P(signal) density overlaid across epochs
  (b) MC  P(signal) density overlaid across epochs (reference)
  (c) MC/EXP density ratio vs xi, one curve per epoch
  (d) summary vs epoch: fraction of hits with xi>0.9 and xi>0.99 (MC & EXP),
      and Wasserstein(MC, EXP).
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import wasserstein_distance

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.iter_pseudolabel.diagnostics import (  # noqa: E402
    _all_event_indices,
    _all_muatm_indices,
    concat_hit_probs,
    infer_events_collect_after_cut,
    load_encoder,
    load_norm,
    renorm,
)
from scripts.iter_pseudolabel.plot_comparison import mc_exp_ratio_bins  # noqa: E402

MC_DEFAULT = "/home/plotnikovgp/baikal/Baikal-ML/pytorch_training/data/baikal_2020_sig-noise_mid-eq_normed.h5"
EXP_DEFAULT = "/home/plotnikovgp/baikal/Baikal-ML/pytorch_training/data/baikal_exp_filtered_norm_w_model_enc_v2_fix_q.h5"
BASELINE = "/home/plotnikovgp/baikal/Baikal-ML/pytorch_training/checkpoints/noise_sig_experiments/k_nsol_labelneq0_hs128/best.ckpt"
CKPT_DIR = "/home/plotnikovgp/baikal/Baikal-ML/pytorch_training/checkpoints/noise_sig_experiments/pseudo_masked_0p1_0p9_epochsweep"


def collect(model, mc_path, exp_path, target, thr, batch_size):
    mc_mean, mc_std = load_norm(mc_path)
    exp_mean, exp_std = load_norm(exp_path)

    def exp_renorm(h):
        return renorm(h, exp_mean, exp_std, mc_mean, mc_std)

    mc_evs, _ = infer_events_collect_after_cut(
        model,
        mc_path,
        "train",
        _all_muatm_indices(mc_path, "train"),
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
        _all_event_indices(exp_path, "train"),
        renorm_fn=exp_renorm,
        batch_size=batch_size,
        thr=thr,
        min_hits=8,
        min_strings=2,
        target_count=target,
        desc="EXP after-cut",
    )
    return concat_hit_probs(mc_evs), concat_hit_probs(exp_evs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, nargs="+", default=[1, 3, 5, 10, 15])
    ap.add_argument("--ckpt-dir", default=CKPT_DIR)
    ap.add_argument("--baseline", default=BASELINE)
    ap.add_argument("--out", default="plots/iter_pseudolabel/_comparison/epoch_evolution.png")
    ap.add_argument("--target", type=int, default=8000)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--cut-threshold", type=float, default=0.5)
    args = ap.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    # (epoch_number, label, ckpt_path); baseline is epoch 0
    models = [(0, "baseline", args.baseline)]
    for e in args.epochs:
        models.append((e, f"epoch {e}", f"{args.ckpt_dir}/epoch{e}.ckpt"))

    series = []
    for ep, label, ckpt in models:
        if not Path(ckpt).exists():
            print(f"[skip] missing {ckpt}")
            continue
        print(f"\n=== {label}: {ckpt} ===")
        model = load_encoder(ckpt)
        mc_p, exp_p = collect(
            model, MC_DEFAULT, EXP_DEFAULT, args.target, args.cut_threshold, args.batch_size
        )
        series.append({"epoch": ep, "label": label, "mc": mc_p, "exp": exp_p})
        del model
        torch.cuda.empty_cache()

    # color by epoch (baseline grey, rest sequential)
    cmap = plt.get_cmap("viridis")
    n = len(series)
    colors = {}
    for i, s in enumerate(series):
        colors[s["epoch"]] = "0.45" if s["epoch"] == 0 else cmap(0.12 + 0.82 * i / max(n - 1, 1))

    bins = np.linspace(0, 1, 101)
    fig = plt.figure(figsize=(15, 13), constrained_layout=True)
    gs = fig.add_gridspec(3, 2)

    # (a) EXP overlay, (b) MC overlay
    for col, key, title in [(0, "exp", "EXP"), (1, "mc", "MC muon")]:
        ax = fig.add_subplot(gs[0, col])
        for s in series:
            ls = "--" if s["epoch"] == 0 else "-"
            ax.hist(
                s[key],
                bins=bins,
                density=True,
                histtype="step",
                linewidth=1.6,
                color=colors[s["epoch"]],
                linestyle=ls,
                label=s["label"],
            )
        ax.set_yscale("log")
        ax.set_xlabel("ξ = P(signal)")
        ax.set_ylabel("Density")
        ax.set_title(f"{title} — P(signal) after cut, by epoch")
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)

    # (c) MC/EXP density ratio vs xi per epoch
    ax = fig.add_subplot(gs[1, :])
    for s in series:
        xc, r, _, ok = mc_exp_ratio_bins(s["mc"], s["exp"], n_bins=50)
        m = ok & np.isfinite(r)
        ls = "--" if s["epoch"] == 0 else "-"
        ax.plot(xc[m], r[m], ls, color=colors[s["epoch"]], linewidth=1.8, label=s["label"])
    ax.axhline(1.0, color="dimgray", ls=":", lw=1.2)
    ax.set_yscale("log")
    ax.set_xlim(0, 1)
    ax.set_xlabel("ξ = P(signal)")
    ax.set_ylabel("ρ_MC / ρ_EXP")
    ax.set_title(
        "Density ratio MC/EXP per ξ bin — by epoch\n"
        "(ratio < 1 at high ξ ⇒ EXP has excess of confident-signal hits)"
    )
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, which="both", alpha=0.3)

    # (d) summary vs epoch
    ax = fig.add_subplot(gs[2, :])
    eps = [s["epoch"] for s in series]
    frac_exp_09 = [100 * (s["exp"] > 0.9).mean() for s in series]
    frac_mc_09 = [100 * (s["mc"] > 0.9).mean() for s in series]
    frac_exp_099 = [100 * (s["exp"] > 0.99).mean() for s in series]
    frac_mc_099 = [100 * (s["mc"] > 0.99).mean() for s in series]
    ax.plot(eps, frac_exp_09, "o-", color="#1f77b4", label="EXP: ξ>0.9")
    ax.plot(eps, frac_mc_09, "o--", color="#2ca02c", label="MC: ξ>0.9")
    ax.plot(eps, frac_exp_099, "s-", color="#9467bd", label="EXP: ξ>0.99")
    ax.plot(eps, frac_mc_099, "s--", color="#8c564b", label="MC: ξ>0.99")
    ax.set_xlabel("training epoch (0 = baseline)")
    ax.set_ylabel("fraction of after-cut hits [%]")
    ax.set_title("Confident-signal hit fraction vs epoch")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax2 = ax.twinx()
    W = [wasserstein_distance(s["mc"], s["exp"]) for s in series]
    ax2.plot(eps, W, "D-", color="#d62728", label="Wasserstein(MC,EXP)")
    ax2.set_ylabel("Wasserstein distance", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax2.legend(fontsize=9, loc="upper right")

    fig.suptitle(
        f"MC muon vs EXP — P(signal) evolution over fine-tuning epochs "
        f"(after ≥8 hits/≥2 strings @ P>{args.cut_threshold:g}, {args.target} evt/model)",
        fontsize=13,
        y=1.02,
    )
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
