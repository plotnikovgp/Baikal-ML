"""Generate comprehensive PDF report: MC val vs EXP distributions + PR curves.

Model: noise_sig_rerun_v2/encoder_nl5_nh1_dff128_hs128_bs128/best_2020.ckpt
Features: [charge, time, x, y, z] (indices 0-4).
Signal definition for PR: |t_res| < 10 ns.
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
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.encoder import Encoder, EncoderDomainAdaptation

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MC_DEFAULT = "/home/plotnikovgp/baikal/Baikal-ML/pytorch_training/data/baikal_2020_sig-noise_mid-eq_normed.h5"
EXP_DATA = "/home/plotnikovgp/baikal/Baikal-ML/pytorch_training/data/baikal_exp_filtered_norm_w_model_enc_v2_fix_q.h5"
CKPT = "checkpoints/noise_sig_rerun_v2/encoder_nl5_nh1_dff128_hs128_bs128/best_2020.ckpt"

FEAT_CHARGE = 0
FEAT_Z = 4

MC_COL = "#2ca02c"
EXP_COL = "#1f77b4"
EVTYPE_COLS = {"muatm": "#d62728", "nuatm": "#ff7f0e", "nue2": "#9467bd"}

P_LS, R_LS = "-", "--"

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 9,
        "figure.titlesize": 15,
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


def load_mc_events(h5_path, split, max_per_type, batch_size=128, load_labels=False):
    """Load MC events with t_res (and optionally labels), sampling each event type."""
    src_mean, src_std = load_norm_params(h5_path)
    with h5py.File(h5_path, "r") as f:
        ev_starts = f[f"{split}/ev_starts/data"][:]
        data_all = f[f"{split}/data/data"]
        channels_all = f[f"{split}/channels/data"]
        t_res_all = f[f"{split}/t_res/data"]
        ev_ids = f[f"{split}/ev_ids/data"][:]
        labels_all = f[f"{split}/labels/data"] if load_labels and f"{split}/labels" in f else None
        n_total = len(ev_starts) - 1

        type_indices = {}
        for i in range(n_total):
            t = ev_ids[i].decode().split("_")[0]
            type_indices.setdefault(t, []).append(i)

        indices = []
        for t, idxs in type_indices.items():
            sel = idxs[:max_per_type]
            indices.extend(sel)
            print(f"    {t}: {len(sel)} events (of {len(idxs)} available)")
        indices.sort()

        for bs_start in range(0, len(indices), batch_size):
            batch_idx = indices[bs_start : bs_start + batch_size]
            evs_norm, evs_raw, evs_ch, evs_tres, evs_type, evs_lab = [], [], [], [], [], []
            for idx in batch_idx:
                s, e = int(ev_starts[idx]), int(ev_starts[idx + 1])
                hits = data_all[s:e].astype(np.float32)
                raw = denormalize(hits, src_mean, src_std)
                evs_norm.append(hits)
                evs_raw.append(raw)
                evs_ch.append(channels_all[s:e])
                evs_tres.append(t_res_all[s:e])
                evs_type.append(ev_ids[idx].decode().split("_")[0])
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
            )


def load_exp_events(h5_path, split, max_events, renorm_fn, batch_size=128):
    """Load EXP events (no labels)."""
    src_mean, src_std = load_norm_params(h5_path)
    with h5py.File(h5_path, "r") as f:
        ev_starts = f[f"{split}/ev_starts/data"][:]
        data_all = f[f"{split}/data/data"]
        channels_all = f[f"{split}/channels/data"]
        n = min(max_events, len(ev_starts) - 1) if max_events else len(ev_starts) - 1

        for bs_start in range(0, n, batch_size):
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


def load_model(ckpt_path, hs=128, dff=512, model_type="encoder", da_kwargs=None):
    if model_type == "encoder_da":
        da_kw = da_kwargs or {}
        model = EncoderDomainAdaptation(
            in_features=5,
            hidden_size=hs,
            num_layers=5,
            dim_feedforward_size=dff,
            n_heads=1,
            out_size=2,
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
            out_size=2,
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


def infer_mc(model, data_iter, signal_definition="tres"):
    events = []
    with torch.no_grad():
        for x_n, x_r, mask, ch, tr, ev_types, lab in tqdm(data_iter, desc="MC inference"):
            out = _get_output(model, x_n.to(DEVICE), mask.to(DEVICE).bool())
            probs = torch.sigmoid(out[:, :, 1]).cpu().numpy()
            m_np = mask.numpy().astype(bool)
            r_np = x_r.numpy()
            for i in range(x_n.shape[0]):
                m = m_np[i]
                tres_i = tr[i][m]
                true_sig = np.abs(tres_i) < 10
                if signal_definition == "tres_or_labels" and lab is not None:
                    true_sig = true_sig | (lab[i][m] != 0)
                events.append(
                    {
                        "probs": probs[i][m],
                        "charge": r_np[i, m, FEAT_CHARGE],
                        "z": r_np[i, m, FEAT_Z],
                        "channels": ch[i][m],
                        "t_res": tres_i,
                        "ev_type": ev_types[i],
                        "true_sig": true_sig,
                    }
                )
    return events


def infer_exp(model, data_iter):
    events = []
    with torch.no_grad():
        for x_n, x_r, mask, ch in tqdm(data_iter, desc="EXP inference"):
            out = _get_output(model, x_n.to(DEVICE), mask.to(DEVICE).bool())
            probs = torch.sigmoid(out[:, :, 1]).cpu().numpy()
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


def _compute_micro_pr(evs, thresholds):
    all_probs = np.concatenate([e["probs"] for e in evs])
    all_true = np.concatenate([e["true_sig"].astype(float) for e in evs])
    prec_vals, rec_vals = [], []
    for t in thresholds:
        pred = all_probs > t
        tp = ((pred == 1) & (all_true == 1)).sum()
        fp = ((pred == 1) & (all_true == 0)).sum()
        fn = ((pred == 0) & (all_true == 1)).sum()
        p = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        prec_vals.append(p)
        rec_vals.append(r)
    return np.asarray(prec_vals), np.asarray(rec_vals)


def _plot_pr_panels(pdf, mc_events, ylim, suptitle):
    thresholds = np.linspace(0.01, 0.99, 200)
    ev_types = ["muatm", "nuatm", "nue2"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    for ax, evtype in zip(axes, ev_types):
        evs = [e for e in mc_events if e["ev_type"] == evtype]
        if not evs:
            ax.set_title(f"{evtype}: no events")
            continue
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
        ax.set_title(f"{evtype} ({len(evs)} events)")
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
    parser.add_argument("--checkpoint", default=CKPT)
    parser.add_argument("--model-type", default="encoder", choices=["encoder", "encoder_da"])
    parser.add_argument("--model-name", default=None, help="Display name for title")
    parser.add_argument("--mc-data-path", default=MC_DEFAULT, help="MC h5 file for inference")
    parser.add_argument("--hs", type=int, default=128)
    parser.add_argument("--dff", type=int, default=512)
    parser.add_argument(
        "--signal-definition",
        default="tres",
        choices=["tres", "tres_or_labels"],
        help="GT signal definition: |t_res|<10 (tres) or |t_res|<10 OR |label|!=0 (tres_or_labels)",
    )
    args = parser.parse_args()

    mc_per_type = args.mc_events_per_type or args.max_events
    exp_max = args.exp_max_events or args.max_events

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    exp_mean, exp_std = load_norm_params(EXP_DATA)
    mc_mean, mc_std = load_norm_params(args.mc_data_path)

    def renorm_exp(data):
        return renormalize(data, exp_mean, exp_std, mc_mean, mc_std)

    signal_desc = "|t_res| < 10 ns"
    if args.signal_definition == "tres_or_labels":
        signal_desc = "|t_res| < 10 ns OR |label| > 0"
    load_labels = args.signal_definition == "tres_or_labels"

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
    del model, mc_batches, exp_batches
    torch.cuda.empty_cache()

    mc_muon = [e for e in mc_events if e["ev_type"] == "muatm"]
    print(f"\nMC total: {len(mc_events)}, muon: {len(mc_muon)}")
    print(f"EXP total: {len(exp_events)}")

    cuts = [(0, 0), (8, 2)]
    thr = args.threshold

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
        print(f"  After cut ≥{mh}h ≥{ms}s (@thr={thr}): MC muon={mc_keep}, EXP={ex_keep}")

    print(f"\nGenerating PDF report: {out_path}")
    with PdfPages(str(out_path)) as pdf:
        # Title
        model_name = args.model_name or Path(args.checkpoint).parent.name
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

    print(f"\nDone. Report saved to: {out_path}")


if __name__ == "__main__":
    main()
