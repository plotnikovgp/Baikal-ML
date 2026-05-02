"""Compare signal probability distributions between EXP and MC muon data
for a no-DA model and a DA model.

Handles renormalization: exp data is denormalized from its own norms
and renormalized to the training data norms for each model.
"""

import argparse
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import wasserstein_distance
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.encoder import Encoder, EncoderDomainAdaptation

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Data paths ──────────────────────────────────────────────────
EXP_DATA = "/home/plotnikovgp/baikal/Baikal-ML/pytorch_training/data/baikal_exp_filtered_norm_w_model_enc_v2_fix_q.h5"

MC_MERGED = "/home/plotnikovgp/baikal/data/baikal_mc_merged_m50-na15-nue15_norm.h5"
MC_2020 = "/home/plotnikovgp/baikal/Baikal-ML/pytorch_training/data/baikal_2020_sig-noise_mid-eq_normed.h5"

CKPT_A = "checkpoints/noise_sig_experiments/a_tres_cut_10ns_hs128/best.ckpt"
CKPT_DA = "checkpoints/noise_sig_domain_adaptation/encoder_nl5_hs512_dff512_nh1_noise_sig_da_k02_gr005_bs128_tres_cut/best_sig_noise_2020.ckpt"


def load_norm_params(h5_path):
    with h5py.File(h5_path, "r") as f:
        mean = f["norm_param/mean"][:]
        std = f["norm_param/std"][:]
    return mean.astype(np.float32), std.astype(np.float32)


def renormalize(data, src_mean, src_std, dst_mean, dst_std):
    """Denorm from src, renorm to dst: (data * src_std + src_mean - dst_mean) / dst_std"""
    return (data * src_std + src_mean - dst_mean) / dst_std


def load_events_batched(
    h5_path, split, max_events, batch_size=128, filter_muatm=False, renorm_fn=None
):
    """Load hit-level data as batched padded tensors. Yields (x, mask, channels) batches."""
    with h5py.File(h5_path, "r") as f:
        ev_starts = f[f"{split}/ev_starts/data"][:]
        n_events = len(ev_starts) - 1

        if filter_muatm:
            ev_ids = f[f"{split}/ev_ids/data"][:]
            muatm_indices = np.array([i for i in range(n_events) if ev_ids[i].startswith(b"muatm")])
            if max_events:
                muatm_indices = muatm_indices[:max_events]
            event_indices = muatm_indices
        else:
            if max_events:
                event_indices = np.arange(min(max_events, n_events))
            else:
                event_indices = np.arange(n_events)

        data_all = f[f"{split}/data/data"]
        channels_all = f[f"{split}/channels/data"]

        for batch_start in range(0, len(event_indices), batch_size):
            batch_idx = event_indices[batch_start : batch_start + batch_size]
            events = []
            ch_events = []
            for idx in batch_idx:
                s, e = int(ev_starts[idx]), int(ev_starts[idx + 1])
                hits = data_all[s:e].astype(np.float32)
                ch = channels_all[s:e]
                if renorm_fn is not None:
                    hits = renorm_fn(hits)
                events.append(hits)
                ch_events.append(ch)

            max_len = max(len(ev) for ev in events)
            bs = len(events)
            x_pad = np.zeros((bs, max_len, 5), dtype=np.float32)
            mask = np.zeros((bs, max_len), dtype=np.float32)
            ch_pad = np.zeros((bs, max_len), dtype=np.int32)
            for i, (ev, ch) in enumerate(zip(events, ch_events)):
                x_pad[i, : len(ev)] = ev
                mask[i, : len(ev)] = 1.0
                ch_pad[i, : len(ev)] = ch

            yield torch.tensor(x_pad), torch.tensor(mask), ch_pad


def run_inference_events(model, data_iter, is_da=False, threshold=0.5):
    """Run model and collect per-event signal probabilities + cut info."""
    event_probs = []
    event_num_signal_hits = []
    event_num_unique_strings = []
    model.eval()
    with torch.no_grad():
        for x, mask, channels in tqdm(data_iter, desc="Inference"):
            x_dev, mask_dev = x.to(DEVICE), mask.to(DEVICE)
            if is_da:
                output, _ = model(x_dev, mask_dev.bool())
            else:
                output = model(x_dev, mask_dev.bool())
            probs = torch.sigmoid(output[:, :, 1]).cpu().numpy()
            mask_np = mask.numpy().astype(bool)

            for i in range(x.shape[0]):
                m = mask_np[i]
                p = probs[i][m]
                ch = channels[i][m]
                preds = p > threshold
                strings = ch // 36
                signal_strings = np.unique(strings[preds])

                event_probs.append(p)
                event_num_signal_hits.append(int(preds.sum()))
                event_num_unique_strings.append(len(signal_strings))

    return event_probs, np.array(event_num_signal_hits), np.array(event_num_unique_strings)


def apply_cuts(event_probs, num_signal_hits, num_unique_strings, min_hits=0, min_strings=0):
    """Filter events by cuts and return concatenated hit probs."""
    mask = (num_signal_hits >= min_hits) & (num_unique_strings >= min_strings)
    n_pass = mask.sum()
    if n_pass == 0:
        return np.array([]), 0, len(mask)
    passing = [event_probs[i] for i in np.where(mask)[0]]
    return np.concatenate(passing), int(n_pass), len(mask)


def plot_distributions(results, output_dir, cut_label=""):
    """Plot probability distributions and compute Wasserstein distances."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f" ({cut_label})" if cut_label else ""

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    bins = np.linspace(0, 1, 100)
    summary_lines = []

    for ax_idx, (model_name, data) in enumerate(results.items()):
        ax = axes[ax_idx]
        mc_probs = data["mc_muon"]
        exp_probs = data["exp"]
        mc_info = data.get("mc_info", "")
        exp_info = data.get("exp_info", "")

        ax.hist(
            mc_probs,
            bins=bins,
            density=True,
            alpha=0.6,
            label=f"MC muon (n_hits={len(mc_probs):,}){mc_info}",
            color="#1f77b4",
        )
        ax.hist(
            exp_probs,
            bins=bins,
            density=True,
            alpha=0.6,
            label=f"EXP (n_hits={len(exp_probs):,}){exp_info}",
            color="#ff7f0e",
        )

        wd = (
            wasserstein_distance(mc_probs, exp_probs)
            if len(mc_probs) > 0 and len(exp_probs) > 0
            else float("nan")
        )
        summary_lines.append(f"{model_name}: Wasserstein = {wd:.6f}")

        ax.set_xlabel("P(signal)", fontsize=14)
        ax.set_ylabel("Density", fontsize=14)
        ax.set_title(f"{model_name}\nWasserstein = {wd:.6f}", fontsize=14, fontweight="bold")
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3, linestyle=":")
        ax.set_xlim(0, 1)

    fig.suptitle(
        f"Signal Probability Distribution: MC muon vs EXP{suffix}", fontsize=16, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(output_dir / "prob_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / 'prob_distributions.png'}")

    # Zoomed view
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    bins_zoom = np.linspace(0, 0.15, 80)

    for ax_idx, (model_name, data) in enumerate(results.items()):
        ax = axes[ax_idx]
        mc_probs = data["mc_muon"]
        exp_probs = data["exp"]

        ax.hist(mc_probs, bins=bins_zoom, density=True, alpha=0.6, label="MC muon", color="#1f77b4")
        ax.hist(exp_probs, bins=bins_zoom, density=True, alpha=0.6, label="EXP", color="#ff7f0e")

        ax.set_xlabel("P(signal)", fontsize=14)
        ax.set_ylabel("Density", fontsize=14)
        ax.set_title(f"{model_name} (zoomed, low prob region)", fontsize=14, fontweight="bold")
        ax.legend(fontsize=12)
        ax.grid(alpha=0.3, linestyle=":")

    fig.suptitle(
        f"Signal Probability (Zoomed): MC muon vs EXP{suffix}", fontsize=16, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(output_dir / "prob_distributions_zoomed.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / 'prob_distributions_zoomed.png'}")

    # CDF comparison
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    for ax_idx, (model_name, data) in enumerate(results.items()):
        ax = axes[ax_idx]
        mc_probs = data["mc_muon"]
        exp_probs = data["exp"]
        if len(mc_probs) > 0 and len(exp_probs) > 0:
            mc_sorted = np.sort(mc_probs)
            exp_sorted = np.sort(exp_probs)
            ax.plot(mc_sorted, np.linspace(0, 1, len(mc_sorted)), label="MC muon", linewidth=2)
            ax.plot(exp_sorted, np.linspace(0, 1, len(exp_sorted)), label="EXP", linewidth=2)
        ax.set_xlabel("P(signal)", fontsize=14)
        ax.set_ylabel("CDF", fontsize=14)
        ax.set_title(f"{model_name}", fontsize=14, fontweight="bold")
        ax.legend(fontsize=12)
        ax.grid(alpha=0.3, linestyle=":")
        ax.set_xlim(0, 1)

    fig.suptitle(f"CDF Comparison: MC muon vs EXP{suffix}", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "prob_cdf_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / 'prob_cdf_comparison.png'}")

    print("\n=== Summary ===")
    for line in summary_lines:
        print(f"  {line}")


def run_model_on_data(
    model,
    h5_path,
    split,
    max_events,
    batch_size,
    is_da=False,
    filter_muatm=False,
    renorm_fn=None,
    threshold=0.5,
):
    """Run inference and return per-event results."""
    data_iter = load_events_batched(
        h5_path,
        split,
        max_events,
        batch_size=batch_size,
        filter_muatm=filter_muatm,
        renorm_fn=renorm_fn,
    )
    return run_inference_events(model, data_iter, is_da=is_da, threshold=threshold)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-events", type=int, default=100_000)
    parser.add_argument("--output-dir", default="plots/exp_vs_mc_distributions")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--split", default="train")
    parser.add_argument("--min-hits", type=int, default=0)
    parser.add_argument("--min-strings", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    has_cuts = args.min_hits > 0 or args.min_strings > 0
    cut_label = f"≥{args.min_hits} signal hits, ≥{args.min_strings} strings" if has_cuts else ""

    exp_mean, exp_std = load_norm_params(EXP_DATA)
    mc_merged_mean, mc_merged_std = load_norm_params(MC_MERGED)
    mc_2020_mean, mc_2020_std = load_norm_params(MC_2020)

    print(f"EXP norm: mean={exp_mean}, std={exp_std}")
    print(f"MC merged norm: mean={mc_merged_mean}, std={mc_merged_std}")
    print(f"MC 2020 norm: mean={mc_2020_mean}, std={mc_2020_std}")
    if has_cuts:
        print(
            f"Cuts: min_hits={args.min_hits}, min_strings={args.min_strings}, threshold={args.threshold}"
        )

    results = {}

    # ── Model A (no-DA, hs=128, trained on MC merged) ──
    print("\n=== Model A: no-DA (a_tres_cut_10ns_hs128) ===")
    model_a = Encoder(
        in_features=5,
        hidden_size=128,
        num_layers=5,
        dim_feedforward_size=512,
        n_heads=1,
        out_size=2,
        dropout_p=0.0,
    )
    sd = torch.load(CKPT_A, map_location="cpu", weights_only=False)
    model_a.load_state_dict(sd)
    model_a.to(DEVICE).eval()

    def renorm_exp_to_mc_merged(data):
        return renormalize(data, exp_mean, exp_std, mc_merged_mean, mc_merged_std)

    print("  Running on MC muon (from MC merged, val split)...")
    mc_probs_a, mc_nhits_a, mc_nstr_a = run_model_on_data(
        model_a,
        MC_MERGED,
        "val",
        args.max_events,
        args.batch_size,
        is_da=False,
        filter_muatm=True,
        threshold=args.threshold,
    )
    print("  Running on EXP (renormed to MC merged norms)...")
    exp_probs_a, exp_nhits_a, exp_nstr_a = run_model_on_data(
        model_a,
        EXP_DATA,
        args.split,
        args.max_events,
        args.batch_size,
        is_da=False,
        renorm_fn=renorm_exp_to_mc_merged,
        threshold=args.threshold,
    )

    mc_flat_a, mc_pass_a, mc_total_a = apply_cuts(
        mc_probs_a, mc_nhits_a, mc_nstr_a, args.min_hits, args.min_strings
    )
    exp_flat_a, exp_pass_a, exp_total_a = apply_cuts(
        exp_probs_a, exp_nhits_a, exp_nstr_a, args.min_hits, args.min_strings
    )
    print(f"  MC muon: {mc_pass_a}/{mc_total_a} events pass cuts, {len(mc_flat_a):,} hits")
    print(f"  EXP: {exp_pass_a}/{exp_total_a} events pass cuts, {len(exp_flat_a):,} hits")

    results["No-DA (a_tres_cut_10ns)"] = {
        "mc_muon": mc_flat_a,
        "exp": exp_flat_a,
        "mc_info": f"\n{mc_pass_a}/{mc_total_a} ev",
        "exp_info": f"\n{exp_pass_a}/{exp_total_a} ev",
    }
    del model_a
    torch.cuda.empty_cache()

    # ── DA model (hs=512, trained on MC 2020 + exp) ──
    print("\n=== DA Model (k02_gr005_tres_cut) ===")
    model_da = EncoderDomainAdaptation(
        in_features=5,
        hidden_size=512,
        num_layers=5,
        dim_feedforward_size=512,
        n_heads=1,
        out_size=2,
        dropout_p=0.0,
        num_domains=2,
        domain_classifier_hidden_size=128,
        domain_classifier_layers=2,
        gradient_reversal_alpha=0.05,
        aggregate_output=False,
    )
    sd = torch.load(CKPT_DA, map_location="cpu", weights_only=False)
    model_da.load_state_dict(sd)
    model_da.to(DEVICE).eval()

    def renorm_exp_to_mc2020(data):
        return renormalize(data, exp_mean, exp_std, mc_2020_mean, mc_2020_std)

    print("  Running on MC muon (from MC 2020, test split)...")
    mc_probs_da, mc_nhits_da, mc_nstr_da = run_model_on_data(
        model_da,
        MC_2020,
        "test",
        args.max_events,
        args.batch_size,
        is_da=True,
        filter_muatm=True,
        threshold=args.threshold,
    )
    print("  Running on EXP (renormed to MC 2020 norms)...")
    exp_probs_da, exp_nhits_da, exp_nstr_da = run_model_on_data(
        model_da,
        EXP_DATA,
        args.split,
        args.max_events,
        args.batch_size,
        is_da=True,
        renorm_fn=renorm_exp_to_mc2020,
        threshold=args.threshold,
    )

    mc_flat_da, mc_pass_da, mc_total_da = apply_cuts(
        mc_probs_da, mc_nhits_da, mc_nstr_da, args.min_hits, args.min_strings
    )
    exp_flat_da, exp_pass_da, exp_total_da = apply_cuts(
        exp_probs_da, exp_nhits_da, exp_nstr_da, args.min_hits, args.min_strings
    )
    print(f"  MC muon: {mc_pass_da}/{mc_total_da} events pass cuts, {len(mc_flat_da):,} hits")
    print(f"  EXP: {exp_pass_da}/{exp_total_da} events pass cuts, {len(exp_flat_da):,} hits")

    results["DA (k02_gr005_tres_cut)"] = {
        "mc_muon": mc_flat_da,
        "exp": exp_flat_da,
        "mc_info": f"\n{mc_pass_da}/{mc_total_da} ev",
        "exp_info": f"\n{exp_pass_da}/{exp_total_da} ev",
    }
    del model_da
    torch.cuda.empty_cache()

    # ── Plot ──
    plot_distributions(results, args.output_dir, cut_label=cut_label)


if __name__ == "__main__":
    main()
