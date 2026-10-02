"""Spatial-cluster metric for MC vs EXP signal hits.

Idea: signal hits from a single particle should be geometrically connected
(one blob/track).  On EXP data we sometimes see stray signal hits far from the
main activation cluster.  We quantify this by counting the number of
connected spatial clusters of network-tagged signal hits per event.

Clustering = connected components: two signal hits are joined if they are
within `eps` meters of each other (DBSCAN with min_samples=1).

Per-event metrics:
  - n_clusters       : number of connected components
  - stray_frac       : fraction of signal hits NOT in the largest cluster
  - max_dist         : distance [m] from main-cluster centroid to farthest
                       signal hit

Outputs (under --out-dir):
  - cluster_metrics.png       distributions MC vs EXP (raw counts, equal events)
  - eps_scan.png              fraction of events with >1 cluster vs eps
  - examples_3d.png           representative 3D event displays
  - cluster_metrics.json      summary numbers

Usage:
  python scripts/cluster_metric_mc_exp.py \
      --ckpt checkpoints/noise_sig_experiments/k_nsol_labelneq0_hs128/best.ckpt \
      --out-dir plots/cluster_metric --target 4000 --eps 60
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
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.stats import ks_2samp
from sklearn.cluster import DBSCAN
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.encoder import Encoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MC_COL, EXP_COL = "#2ca02c", "#1f77b4"
MC_DEFAULT = "/home/plotnikovgp/baikal/Baikal-ML/pytorch_training/data/baikal_2020_sig-noise_mid-eq_normed.h5"
EXP_DEFAULT = "/home/plotnikovgp/baikal/Baikal-ML/pytorch_training/data/baikal_exp_filtered_norm_w_model_enc_v2_fix_q.h5"


def load_norm(h5_path):
    with h5py.File(h5_path, "r") as f:
        return (
            f["norm_param/mean"][:].astype(np.float32),
            f["norm_param/std"][:].astype(np.float32),
        )


def load_encoder(ckpt_path, hidden_size=128, num_layers=5, dim_feedforward=512):
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
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


def infer_events_xyz(model, h5_path, split, indices, renorm_fn, batch_size=128):
    """Per-event dicts with probs, channels, and physical xyz [m]."""
    events = []
    mean, std = load_norm(h5_path)
    with h5py.File(h5_path, "r") as f:
        ev_starts = f[f"{split}/ev_starts/data"][:]
        data_all = f[f"{split}/data/data"]
        channels_all = f[f"{split}/channels/data"]
        labels_all = f[f"{split}/labels/data"] if f"{split}/labels" in f else None

        for bs in range(0, len(indices), batch_size):
            bidx = indices[bs : bs + batch_size]
            hits_list, ch_list, xyz_list, lab_list = [], [], [], []
            q_list, t_list = [], []
            for idx in bidx:
                s, e = int(ev_starts[idx]), int(ev_starts[idx + 1])
                hits = data_all[s:e].astype(np.float32)
                raw = hits * std + mean
                model_in = renorm_fn(hits) if renorm_fn is not None else hits
                hits_list.append(model_in)
                ch_list.append(channels_all[s:e])
                xyz_list.append(raw[:, 2:5])
                q_list.append(raw[:, 0])
                t_list.append(raw[:, 1])
                lab_list.append(labels_all[s:e] if labels_all is not None else None)

            max_len = max(len(h) for h in hits_list)
            B = len(hits_list)
            x = np.zeros((B, max_len, 5), dtype=np.float32)
            mask = np.zeros((B, max_len), dtype=np.float32)
            for i, h in enumerate(hits_list):
                x[i, : len(h)] = h
                mask[i, : len(h)] = 1.0

            with torch.no_grad():
                out = model(torch.tensor(x).to(DEVICE), torch.tensor(mask).to(DEVICE).bool())
                if isinstance(out, tuple):
                    out = out[0]
            probs = torch.sigmoid(out[:, :, 1]).cpu().numpy()
            mask_np = mask.astype(bool)
            for i in range(B):
                m = mask_np[i]
                ev = {
                    "probs": probs[i][m],
                    "channels": ch_list[i],
                    "xyz": xyz_list[i],
                    "q": q_list[i],
                    "t": t_list[i],
                }
                if lab_list[i] is not None:
                    ev["labels"] = lab_list[i]
                events.append(ev)
    return events


def event_passes_cut(ev, thr, min_hits=8, min_strings=2):
    sig = ev["probs"] > thr
    if int(sig.sum()) < min_hits:
        return False
    return len(np.unique(ev["channels"][sig] // 36)) >= min_strings


# Baikal-GVD geometry: strings 60 m apart (horizontal), OMs 15 m apart (vertical).
SCALE_H = 60.0  # m, horizontal inter-string distance
SCALE_V = 15.0  # m, vertical inter-OM distance
C_LIGHT = 0.3  # m/ns, used to convert hit time to a pseudo-distance (ct)
USE_TIME = True  # include time (as c*t) as 4th coordinate
USE_OM_UNITS = True  # True: anisotropic OM-units (X,Y/60, Z/15); False: isotropic meters


def _scaled_features(xyz, t, use_time=True):
    """Build clustering features.

    OM-units mode (USE_OM_UNITS=True): anisotropic, X,Y / 60 m, Z / 15 m, and
    time as c*t / 15 m (light travel between adjacent OMs ~ 1 unit).

    Meters mode (USE_OM_UNITS=False): isotropic Euclidean space [X, Y, Z, c*t]
    all in meters, so c*t competes with real distances on equal footing and the
    speed of light is accounted for consistently. eps is then in meters.
    """
    if USE_OM_UNITS:
        feats = [xyz[:, 0] / SCALE_H, xyz[:, 1] / SCALE_H, xyz[:, 2] / SCALE_V]
        if use_time:
            feats.append((C_LIGHT * t) / SCALE_V)
    else:
        feats = [xyz[:, 0], xyz[:, 1], xyz[:, 2]]
        if use_time:
            feats.append(C_LIGHT * t)
    return np.stack(feats, axis=1).astype(np.float32)


def cluster_signal_hits(xyz_sig, eps, t_sig=None, min_samples=1, use_time=None):
    """Connected-components labels in detector module-units (optionally 4D w/ time)."""
    if use_time is None:
        use_time = USE_TIME
    if len(xyz_sig) == 0:
        return np.array([], dtype=int)
    feats = _scaled_features(xyz_sig, t_sig, use_time and t_sig is not None)
    return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(feats)


def pca_axis(xyz):
    """Return (centroid, unit principal axis, linearity) for points xyz."""
    c = xyz.mean(axis=0)
    X = xyz - c
    cov = X.T @ X / len(X)
    w, V = np.linalg.eigh(cov)  # ascending eigenvalues
    v = V[:, -1]  # principal axis
    eigs = w[::-1]
    linearity = float(eigs[0] / (eigs.sum() + 1e-9))
    return c, v, linearity


def perp_dist(points, c, v):
    """Perpendicular distance of points to the line through c along unit v."""
    d = points - c
    along = d @ v
    proj = np.outer(along, v)
    return np.linalg.norm(d - proj, axis=1)


def event_pca_metrics(ev, sig_mask, eps, use_time=None, outlier_m=60.0):
    """PCA-based stray metrics in (X, Y, Z[, c*t]) space [meters].

    The axis is fit on the main cluster (so a stray hit does not bias the
    principal direction); residuals are measured for all signal hits.  Adding
    c*t breaks the same-Z degeneracy that biases a positions-only fit."""
    if use_time is None:
        use_time = USE_TIME
    xyz = ev["xyz"][sig_mask]
    t = ev["t"][sig_mask]
    if len(xyz) < 3:
        return {"max_perp": 0.0, "rms_perp": 0.0, "linearity": 1.0, "n_outliers": 0, "axis": None}
    # PCA features in meters: positions plus ct (so timing competes with space)
    if use_time:
        feats = np.column_stack([xyz, C_LIGHT * t]).astype(np.float64)
    else:
        feats = xyz.astype(np.float64)
    labels = cluster_signal_hits(xyz, eps, t_sig=t, use_time=use_time)
    uniq, counts = np.unique(labels, return_counts=True)
    main = uniq[counts.argmax()]
    fit = feats[labels == main]
    fit = fit if len(fit) >= 3 else feats
    c, v, linearity = pca_axis(fit)
    perp = perp_dist(feats, c, v)
    return {
        "max_perp": float(perp.max()),
        "rms_perp": float(np.sqrt((perp**2).mean())),
        "linearity": linearity,
        "n_outliers": int((perp > outlier_m).sum()),
        "axis": (c[:3], v[:3]),  # spatial projection for plotting
    }


def sig_pred(thr, q_min=0.0):
    return lambda ev: (ev["probs"] > thr) & (ev["q"] > q_min)


def sig_gt(q_min=0.0):
    return lambda ev: (ev["labels"] != 0) & (ev["q"] > q_min)


def event_metrics(ev, sig_mask, eps, use_time=None):
    """Cluster metrics given a boolean signal mask over the event's hits."""
    if use_time is None:
        use_time = USE_TIME
    xyz = ev["xyz"][sig_mask]
    t = ev["t"][sig_mask]
    if len(xyz) == 0:
        return {
            "n_clusters": 0,
            "stray_frac": 0.0,
            "max_dist": 0.0,
            "n_sig": 0,
            "labels": np.array([], int),
            "xyz_sig": xyz,
        }
    labels = cluster_signal_hits(xyz, eps, t_sig=t, use_time=use_time)
    uniq, counts = np.unique(labels, return_counts=True)
    n_clusters = len(uniq)
    main = uniq[counts.argmax()]
    stray_frac = 1.0 - counts.max() / len(labels)
    centroid = xyz[labels == main].mean(axis=0)
    max_dist = float(np.linalg.norm(xyz - centroid, axis=1).max())
    return {
        "n_clusters": int(n_clusters),
        "stray_frac": float(stray_frac),
        "max_dist": max_dist,
        "n_sig": int(sig_mask.sum()),
        "labels": labels,
        "xyz_sig": xyz,
    }


def collect_after_cut(
    model, h5_path, split, index_pool, renorm_fn, target, thr, batch_size=128, chunk=4096
):
    kept = []
    pbar = tqdm(total=target, desc=f"{Path(h5_path).name}", unit="evt")
    for i in range(0, len(index_pool), chunk):
        evs = infer_events_xyz(
            model, h5_path, split, index_pool[i : i + chunk], renorm_fn, batch_size
        )
        for e in evs:
            if event_passes_cut(e, thr):
                kept.append(e)
                pbar.update(1)
                if len(kept) >= target:
                    pbar.close()
                    return kept
    pbar.close()
    return kept


def muatm_indices(h5_path, split):
    return prefix_indices(h5_path, split, "muatm")


def prefix_indices(h5_path, split, prefix):
    with h5py.File(h5_path, "r") as f:
        ev_ids = f[f"{split}/ev_ids/data"][:]
    return np.array(
        [i for i, e in enumerate(ev_ids) if e.decode().startswith(prefix)], dtype=np.int64
    )


def all_indices(h5_path, split):
    with h5py.File(h5_path, "r") as f:
        n = len(f[f"{split}/ev_starts/data"]) - 1
    return np.arange(n, dtype=np.int64)


# ----------------------------- plotting -----------------------------------


def plot_distributions(series, eps, thr, out_path, tag="", unit="OM-units"):
    """series: list of dicts {label, color, metrics}."""
    FS = 11
    kw = dict(histtype="step", linewidth=2.0)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), constrained_layout=True)

    all_nc = [np.array([m["n_clusters"] for m in s["metrics"]]) for s in series]
    all_md = np.concatenate([[m["max_dist"] for m in s["metrics"]] for s in series])

    # (a) n_clusters
    ax = axes[0]
    hi = int(max(nc.max() for nc in all_nc))
    bins = np.arange(0.5, hi + 1.5)
    for s, nc in zip(series, all_nc):
        ax.hist(
            nc,
            bins=bins,
            color=s["color"],
            label=f"{s['label']} (>1: {100 * (nc > 1).mean():.1f}%)",
            **kw,
        )
    ax.set_yscale("log")
    ax.set_xlabel("number of signal-hit clusters", fontsize=FS)
    ax.set_ylabel("events", fontsize=FS)
    ax.set_title(f"Cluster count (eps={eps:g} {unit})", fontsize=FS + 1)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (b) stray fraction
    ax = axes[1]
    bins = np.linspace(0, 0.6, 31)
    for s in series:
        sf = np.array([m["stray_frac"] for m in s["metrics"]])
        ax.hist(sf, bins=bins, color=s["color"], label=f"{s['label']} (μ={sf.mean():.3f})", **kw)
    ax.set_yscale("log")
    ax.set_xlabel("stray-hit fraction (not in main cluster)", fontsize=FS)
    ax.set_ylabel("events", fontsize=FS)
    ax.set_title("Stray-hit fraction", fontsize=FS + 1)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (c) max distance
    ax = axes[2]
    hi = np.percentile(all_md, 99)
    bins = np.linspace(0, hi, 41)
    for s in series:
        md = np.array([m["max_dist"] for m in s["metrics"]])
        ax.hist(
            md, bins=bins, color=s["color"], label=f"{s['label']} (med={np.median(md):.0f} m)", **kw
        )
    ax.set_yscale("log")
    ax.set_xlabel("max dist to main-cluster centroid [m]", fontsize=FS)
    ax.set_ylabel("events", fontsize=FS)
    ax.set_title("Max stray distance", fontsize=FS + 1)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    n = len(series[0]["metrics"])
    fig.suptitle(
        f"Signal-hit clustering — {' vs '.join(s['label'] for s in series)} "
        f"({n} evt each, 8-2 cut @ ξ>{thr})" + (f"  [{tag}]" if tag else ""),
        fontsize=13,
        y=1.04,
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_pca_distributions(series, eps, thr, out_path, tag=""):
    """series dicts must have 'pca' list of per-event PCA metrics."""
    FS = 11
    kw = dict(histtype="step", linewidth=2.0)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), constrained_layout=True)

    all_mp = np.concatenate([[m["max_perp"] for m in s["pca"]] for s in series])

    # (a) max perpendicular distance to principal axis
    ax = axes[0]
    hi = np.percentile(all_mp, 99)
    bins = np.linspace(0, hi, 41)
    for s in series:
        mp = np.array([m["max_perp"] for m in s["pca"]])
        ax.hist(
            mp, bins=bins, color=s["color"], label=f"{s['label']} (med={np.median(mp):.0f} m)", **kw
        )
    ax.set_yscale("log")
    ax.set_xlabel("max perpendicular dist to axis [m]", fontsize=FS)
    ax.set_ylabel("events", fontsize=FS)
    ax.set_title("Max off-axis (stray) distance", fontsize=FS + 1)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (b) linearity (explained variance of principal axis)
    ax = axes[1]
    bins = np.linspace(0.3, 1.0, 36)
    for s in series:
        lin = np.array([m["linearity"] for m in s["pca"]])
        ax.hist(lin, bins=bins, color=s["color"], label=f"{s['label']} (μ={lin.mean():.2f})", **kw)
    ax.set_yscale("log")
    ax.set_xlabel("linearity  λ₁/Σλ", fontsize=FS)
    ax.set_ylabel("events", fontsize=FS)
    ax.set_title("Track-likeness (PCA linearity)", fontsize=FS + 1)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (c) number of off-axis outlier hits (>60 m)
    ax = axes[2]
    hi = int(max(max(m["n_outliers"] for m in s["pca"]) for s in series))
    bins = np.arange(-0.5, max(hi, 4) + 1.5)
    for s in series:
        no = np.array([m["n_outliers"] for m in s["pca"]])
        ax.hist(
            no,
            bins=bins,
            color=s["color"],
            label=f"{s['label']} (>0: {100 * (no > 0).mean():.1f}%)",
            **kw,
        )
    ax.set_yscale("log")
    ax.set_xlabel("# off-axis hits (>60 m from axis)", fontsize=FS)
    ax.set_ylabel("events", fontsize=FS)
    ax.set_title("Off-axis hit count", fontsize=FS + 1)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    n = len(series[0]["pca"])
    fig.suptitle(
        f"PCA off-axis (stray) metric — {' vs '.join(s['label'] for s in series)} "
        f"({n} evt each, 8-2 cut @ ξ>{thr})" + (f"  [{tag}]" if tag else ""),
        fontsize=13,
        y=1.04,
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_eps_scan(series, eps_values, thr, out_path, unit="OM-units"):
    """series: list of dicts {label, color, events, sig_fn}."""
    markers = ["o-", "s-", "^-", "d-", "v-"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    for si, s in enumerate(series):
        frac, strayf = [], []
        for eps in eps_values:
            ms = [event_metrics(e, s["sig_fn"](e), eps) for e in s["events"]]
            frac.append(np.mean([m["n_clusters"] > 1 for m in ms]))
            strayf.append(np.mean([m["stray_frac"] for m in ms]))
        mk = markers[si % len(markers)]
        axes[0].plot(eps_values, 100 * np.array(frac), mk, color=s["color"], label=s["label"])
        axes[1].plot(eps_values, strayf, mk, color=s["color"], label=s["label"])

    axes[0].set_xlabel(f"eps [{unit}]")
    axes[0].set_ylabel("events with >1 cluster [%]")
    axes[0].set_title("Fragmentation vs clustering scale")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].set_xlabel(f"eps [{unit}]")
    axes[1].set_ylabel("mean stray-hit fraction")
    axes[1].set_title("Mean stray-hit fraction vs eps")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def _plot_event_3d(ax, ev, sig, eps, title):
    xyz = ev["xyz"]
    # noise hits
    ax.scatter(xyz[~sig, 0], xyz[~sig, 1], xyz[~sig, 2], c="0.7", s=6, alpha=0.25, label="noise")
    # signal hits colored by cluster
    xyz_sig = xyz[sig]
    labels = cluster_signal_hits(xyz_sig, eps, t_sig=ev["t"][sig])
    uniq, counts = np.unique(labels, return_counts=True)
    main = uniq[counts.argmax()]
    cmap = plt.get_cmap("tab10")
    for j, lab in enumerate(uniq):
        m = labels == lab
        is_main = lab == main
        ax.scatter(
            xyz_sig[m, 0],
            xyz_sig[m, 1],
            xyz_sig[m, 2],
            color=cmap(j % 10),
            s=46 if is_main else 70,
            marker="o" if is_main else "X",
            edgecolors="black",
            linewidths=0.4,
            label=f"{'main' if is_main else 'stray'} ({m.sum()})",
        )
    # PCA principal axis (time-aware fit on main cluster, projected to space)
    if (labels == main).sum() >= 3:
        ts = ev["t"][sig]
        feats = np.column_stack([xyz_sig, C_LIGHT * ts]) if USE_TIME else xyz_sig
        c, v, _ = pca_axis(feats[labels == main])
        proj = (feats - c) @ v
        cs, vs = c[:3], v[:3]
        line = cs + np.outer(np.linspace(proj.min(), proj.max(), 2), vs)
        ax.plot(
            line[:, 0],
            line[:, 1],
            line[:, 2],
            color="black",
            lw=1.4,
            ls="--",
            alpha=0.7,
            label="PCA axis",
        )
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("X [m]", fontsize=7)
    ax.set_ylabel("Y [m]", fontsize=7)
    ax.set_zlabel("Z [m]", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.legend(fontsize=6, loc="upper left")


def plot_examples_3d(series, thr, eps, out_path, n_each=4):
    """One row per series; most-fragmented examples (highest max_dist, >1 cluster)."""

    def pick(s):
        scored = []
        for e in s["events"]:
            sig = s["sig_fn"](e)
            m = event_metrics(e, sig, eps)
            if m["n_clusters"] > 1:
                scored.append((m["max_dist"], e, sig, m))
        scored.sort(key=lambda t: t[0], reverse=True)
        return scored[:n_each]

    n_rows = len(series)
    fig = plt.figure(figsize=(5 * n_each, 5 * n_rows))
    for row, s in enumerate(series):
        for col, (_, e, sig, m) in enumerate(pick(s)):
            ax = fig.add_subplot(n_rows, n_each, row * n_each + col + 1, projection="3d")
            _plot_event_3d(
                ax,
                e,
                sig,
                eps,
                f"{s['label']}  k={m['n_clusters']}, "
                f"stray={m['stray_frac']:.2f}, d={m['max_dist']:.0f} m",
            )

    fig.suptitle(
        f"Most-fragmented signal-hit events (eps={eps} OM-units, ξ>{thr})  "
        f"rows: {', '.join(s['label'] for s in series)}",
        fontsize=13,
        y=0.99,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ckpt", default="checkpoints/noise_sig_experiments/k_nsol_labelneq0_hs128/best.ckpt"
    )
    ap.add_argument(
        "--mode", default="mc_exp", choices=["mc_exp", "particles", "eas_gt_pred", "eas_gt_mc_exp"]
    )
    ap.add_argument("--mc", default=MC_DEFAULT)
    ap.add_argument("--exp", default=EXP_DEFAULT)
    ap.add_argument("--out-dir", default="plots/cluster_metric")
    ap.add_argument("--target", type=int, default=4000)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument(
        "--space",
        choices=["om", "m"],
        default="om",
        help="clustering space: 'om' = anisotropic OM-units, 'm' = isotropic meters (X,Y,Z,c*t)",
    )
    ap.add_argument(
        "--eps",
        type=float,
        default=None,
        help="clustering radius; OM-units if --space om (default 2.5), "
        "meters if --space m (default 60)",
    )
    ap.add_argument(
        "--q-min", type=float, default=0.0, help="keep only signal hits with charge Q > q_min"
    )
    ap.add_argument(
        "--no-time",
        action="store_true",
        help="disable time (c*t) as 4th coordinate in clustering/PCA",
    )
    ap.add_argument("--batch-size", type=int, default=128)
    args = ap.parse_args()

    global USE_TIME, USE_OM_UNITS
    USE_TIME = not args.no_time
    USE_OM_UNITS = args.space == "om"
    unit = "OM-units" if USE_OM_UNITS else "m"
    if args.eps is None:
        args.eps = 2.5 if USE_OM_UNITS else 60.0
    eps_values = (
        [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 8.0]
        if USE_OM_UNITS
        else [15.0, 30.0, 45.0, 60.0, 90.0, 120.0, 150.0, 200.0]
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading model...")
    model = load_encoder(args.ckpt)

    pred_fn = sig_pred(args.thr, args.q_min)
    tag = (
        f"{Path(args.ckpt).parent.name} | space={unit} | "
        f"time={'on' if USE_TIME else 'off'} | Q>{args.q_min:g}"
    )

    if args.mode == "eas_gt_pred":
        # One EAS (muatm) sample (collected via network cut); two signal definitions.
        pool = prefix_indices(args.mc, "train", "muatm")
        print(f"Collecting {args.target} EAS events after 8-2 cut (pool={len(pool)})...")
        evs = collect_after_cut(
            model, args.mc, "train", pool, None, args.target, args.thr, args.batch_size
        )
        print(f"  got {len(evs)}")
        series = [
            {
                "label": "GT (label≠0)",
                "color": "#d62728",
                "events": evs,
                "sig_fn": sig_gt(args.q_min),
            },
            {
                "label": f"Network (ξ>{args.thr})",
                "color": "#1f77b4",
                "events": evs,
                "sig_fn": pred_fn,
            },
        ]
    elif args.mode == "eas_gt_mc_exp":
        # 3 lines: GT MC EAS, network MC EAS, network EXP (EAS-like), all at thr.
        mc_mean, mc_std = load_norm(args.mc)
        exp_mean, exp_std = load_norm(args.exp)

        def renorm_exp(h):
            return (h * exp_std + exp_mean - mc_mean) / mc_std

        mc_pool = prefix_indices(args.mc, "train", "muatm")
        print(f"Collecting {args.target} MC EAS events after 8-2 cut (pool={len(mc_pool)})...")
        mc_evs = collect_after_cut(
            model, args.mc, "train", mc_pool, None, args.target, args.thr, args.batch_size
        )
        print(f"  got {len(mc_evs)}")
        exp_pool = all_indices(args.exp, "train")
        print(f"Collecting {args.target} EXP events after 8-2 cut (pool={len(exp_pool)})...")
        exp_evs = collect_after_cut(
            model, args.exp, "train", exp_pool, renorm_exp, args.target, args.thr, args.batch_size
        )
        print(f"  got {len(exp_evs)}")
        series = [
            {
                "label": "GT MC EAS (label≠0)",
                "color": "#d62728",
                "events": mc_evs,
                "sig_fn": sig_gt(args.q_min),
            },
            {
                "label": f"Network MC EAS (ξ>{args.thr})",
                "color": "#2ca02c",
                "events": mc_evs,
                "sig_fn": pred_fn,
            },
            {
                "label": f"Network EXP (ξ>{args.thr})",
                "color": "#1f77b4",
                "events": exp_evs,
                "sig_fn": pred_fn,
            },
        ]
    else:
        # Build streams: (label, color, h5_path, index_pool, renorm_fn)
        if args.mode == "particles":
            streams = [
                (
                    "EAS (muatm)",
                    "#2ca02c",
                    args.mc,
                    prefix_indices(args.mc, "train", "muatm"),
                    None,
                ),
                (
                    "atm nu (nuatm)",
                    "#ff7f0e",
                    args.mc,
                    prefix_indices(args.mc, "train", "nuatm"),
                    None,
                ),
                ("nue2", "#9467bd", args.mc, prefix_indices(args.mc, "train", "nue2"), None),
            ]
        else:
            mc_mean, mc_std = load_norm(args.mc)
            exp_mean, exp_std = load_norm(args.exp)

            def renorm_exp(h):
                return (h * exp_std + exp_mean - mc_mean) / mc_std

            streams = [
                ("MC EAS", MC_COL, args.mc, muatm_indices(args.mc, "train"), None),
                ("EXP", EXP_COL, args.exp, all_indices(args.exp, "train"), renorm_exp),
            ]
        series = []
        for label, color, h5_path, pool, renorm_fn in streams:
            print(f"Collecting {args.target} '{label}' events after 8-2 cut (pool={len(pool)})...")
            evs = collect_after_cut(
                model, h5_path, "train", pool, renorm_fn, args.target, args.thr, args.batch_size
            )
            print(f"  got {len(evs)}")
            series.append({"label": label, "color": color, "events": evs, "sig_fn": pred_fn})

    n = min(len(s["events"]) for s in series)
    for s in series:
        s["events"] = s["events"][:n]
        s["metrics"] = [event_metrics(e, s["sig_fn"](e), args.eps, USE_TIME) for e in s["events"]]
        s["pca"] = [event_pca_metrics(e, s["sig_fn"](e), args.eps, USE_TIME) for e in s["events"]]
    print(f"Using {n} events each")

    plot_distributions(series, args.eps, args.thr, out_dir / "cluster_metrics.png", tag, unit)
    plot_pca_distributions(series, args.eps, args.thr, out_dir / "pca_metrics.png", tag)
    plot_eps_scan(series, eps_values, args.thr, out_dir / "eps_scan.png", unit)
    plot_examples_3d(series, args.thr, args.eps, out_dir / "examples_3d.png", n_each=4)

    summary = {
        "n_events": n,
        "eps": args.eps,
        "eps_unit": unit,
        "space": args.space,
        "scale_h_m": SCALE_H,
        "scale_v_m": SCALE_V,
        "use_time": USE_TIME,
        "c_light_m_per_ns": C_LIGHT,
        "q_min": args.q_min,
        "ckpt": args.ckpt,
        "threshold": args.thr,
        "series": {},
    }
    for s in series:
        nc = np.array([m["n_clusters"] for m in s["metrics"]])
        summary["series"][s["label"]] = {
            "mean_n_clusters": float(nc.mean()),
            "frac_multicluster": float((nc > 1).mean()),
            "mean_stray_frac": float(np.mean([m["stray_frac"] for m in s["metrics"]])),
            "median_max_dist_m": float(np.median([m["max_dist"] for m in s["metrics"]])),
            "pca_median_max_perp_m": float(np.median([m["max_perp"] for m in s["pca"]])),
            "pca_mean_linearity": float(np.mean([m["linearity"] for m in s["pca"]])),
            "pca_frac_with_offaxis": float(np.mean([m["n_outliers"] > 0 for m in s["pca"]])),
        }
    (out_dir / "cluster_metrics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
