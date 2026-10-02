"""Aggregate per-round diagnostics across cells into one comparison.

Reads:  plots/iter_pseudolabel/<cell>/iter_<k>/summary.json
        plots/iter_pseudolabel/round_0_base/summary.json
Writes: plots/iter_pseudolabel/_summary/{table.csv, wass_vs_round.png}

Usage:
    python scripts/iter_pseudolabel/summary.py \
        --plots-root plots/iter_pseudolabel \
        --out-dir   plots/iter_pseudolabel/_summary
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_summary(path: Path):
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def extract_metrics(data: dict | None) -> dict:
    """Translate diagnostics summary.json to flat scalars used by the table/plot."""
    if not data:
        return {}
    w = data.get("wasserstein") or {}
    sh = data.get("signal_hits_distribution") or {}
    pr = data.get("mc_val_pr") or {}
    ks_h = data.get("ks_signal_hits") or {}
    sel = data.get("selection_efficiency") or {}
    sr = data.get("signal_region_ks") or {}
    bim = data.get("bimodality") or {}

    after_cut_keys = [k for k in w if k != "no_cut"]
    after_cut_key = after_cut_keys[0] if after_cut_keys else None

    def _w(k):
        return (w.get(k) or {}).get("wasserstein")

    out = {
        "wass_no_cut": _w("no_cut"),
        "wass_after_cut": _w(after_cut_key) if after_cut_key else None,
        "after_cut_label": after_cut_key,
        "mc_mean_sig_hits": (sh.get("mc") or {}).get("mean"),
        "exp_mean_sig_hits": (sh.get("exp") or {}).get("mean"),
        "delta_sig_hits": ks_h.get("delta_mean"),
        "ks_sig_hits_stat": ks_h.get("ks_stat"),
        "ks_sig_hits_p": ks_h.get("ks_pvalue"),
        "mc_sel_eff_pct": (sel.get("mc_muatm") or {}).get("efficiency_pct"),
        "exp_sel_eff_pct": (sel.get("exp") or {}).get("efficiency_pct"),
        "sel_eff_ratio": sel.get("mc_to_exp_ratio"),
        "sr_ks_stat": sr.get("ks_stat"),
        "sr_wasserstein": sr.get("wasserstein"),
        "mc_frac_ambig": (bim.get("mc") or {}).get("frac_ambig"),
        "exp_frac_ambig": (bim.get("exp") or {}).get("frac_ambig"),
        "mc_frac_signal": (bim.get("mc") or {}).get("frac_signal"),
        "exp_frac_signal": (bim.get("exp") or {}).get("frac_signal"),
    }
    if pr:
        first = next(iter(pr.values()))
        out["mc_val_precision"] = first.get("precision")
        out["mc_val_recall"] = first.get("recall")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plots-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    baseline = load_summary(args.plots_root / "round_0_base" / "summary.json")

    rows = []
    if baseline is not None:
        rows.append({"cell": "round_0_base", "round": 0, **extract_metrics(baseline)})

    for cell_dir in sorted(args.plots_root.iterdir()):
        if not cell_dir.is_dir() or cell_dir.name in ("_summary", "round_0_base"):
            continue
        has_iter = False
        for iter_dir in sorted(cell_dir.iterdir()):
            if not iter_dir.is_dir() or not iter_dir.name.startswith("iter_"):
                continue
            data = load_summary(iter_dir / "summary.json")
            if data is None:
                continue
            has_iter = True
            try:
                k = int(iter_dir.name.replace("iter_", ""))
            except ValueError:
                continue
            rows.append({"cell": cell_dir.name, "round": k, **extract_metrics(data)})
        if not has_iter:
            data = load_summary(cell_dir / "summary.json")
            if data is not None:
                rows.append({"cell": cell_dir.name, "round": 0, **extract_metrics(data)})

    table_csv = args.out_dir / "table.csv"
    fieldnames = [
        "cell",
        "round",
        "wass_no_cut",
        "wass_after_cut",
        "after_cut_label",
        "mc_mean_sig_hits",
        "exp_mean_sig_hits",
        "delta_sig_hits",
        "ks_sig_hits_stat",
        "ks_sig_hits_p",
        "mc_sel_eff_pct",
        "exp_sel_eff_pct",
        "sel_eff_ratio",
        "sr_ks_stat",
        "sr_wasserstein",
        "mc_frac_ambig",
        "exp_frac_ambig",
        "mc_frac_signal",
        "exp_frac_signal",
        "mc_val_precision",
        "mc_val_recall",
    ]
    with table_csv.open("w", newline="") as f:
        w_csv = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w_csv.writeheader()
        for r in rows:
            w_csv.writerow(r)

    iterable_cells = sorted(
        {
            r["cell"]
            for r in rows
            if r["cell"] != "round_0_base"
            and r["cell"] not in ("da_k0p0001", "tres_abs_merged", "zm_hs128")
        }
    )
    ref_cells = [
        c
        for c in ("da_k0p0001", "tres_abs_merged", "zm_hs128")
        if any(r["cell"] == c for r in rows)
    ]

    base_metrics = extract_metrics(baseline) if baseline is not None else {}

    metric_panels = [
        ("wass_no_cut", "Wasserstein (no cut)", "Wasserstein"),
        ("wass_after_cut", "Wasserstein (8h, 2s cut)", "Wasserstein"),
        ("ks_sig_hits_stat", "KS stat: signal hits/event", "KS statistic"),
        ("delta_sig_hits", "Δ mean signal hits (EXP−MC)", "Δ hits"),
        ("sr_ks_stat", "Signal-region P(sig) KS stat", "KS statistic"),
        ("sr_wasserstein", "Signal-region P(sig) Wasserstein", "Wasserstein"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(20, 10), constrained_layout=True)
    axes = axes.flatten()

    for ax, (key, title, ylabel) in zip(axes, metric_panels):
        for cell in iterable_cells:
            cell_rows = sorted([r for r in rows if r["cell"] == cell], key=lambda r: r["round"])
            xs = [r["round"] for r in cell_rows]
            ys = [r.get(key) for r in cell_rows]
            ax.plot(xs, ys, marker="o", label=cell, linewidth=1.5)

        if base_metrics and base_metrics.get(key) is not None:
            ax.axhline(base_metrics[key], color="k", ls="--", alpha=0.5, label="base ckpt")

        for rc in ref_cells:
            rc_row = next((r for r in rows if r["cell"] == rc), None)
            if rc_row and rc_row.get(key) is not None:
                ax.axhline(rc_row[key], ls=":", alpha=0.6, label=rc)

        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Round")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, ncol=2)

    fig.suptitle("Iterative Pseudolabel Sweep — Metric Comparison", fontsize=14)
    fig.savefig(args.out_dir / "metrics_vs_round.png", dpi=130)
    plt.close(fig)

    # Selection efficiency comparison bar chart
    all_cells_for_bar = ["round_0_base"] + ref_cells + iterable_cells
    fig2, ax2 = plt.subplots(figsize=(14, 5), constrained_layout=True)
    labels_list, mc_vals, exp_vals = [], [], []
    for cell in all_cells_for_bar:
        cell_rows = [r for r in rows if r["cell"] == cell]
        if not cell_rows:
            continue
        last = sorted(cell_rows, key=lambda r: r["round"])[-1]
        if last.get("mc_sel_eff_pct") is None:
            continue
        lbl = f"{cell}\nr{last['round']}" if last["round"] > 0 else cell
        labels_list.append(lbl)
        mc_vals.append(last["mc_sel_eff_pct"])
        exp_vals.append(last["exp_sel_eff_pct"])

    import numpy as np

    x = np.arange(len(labels_list))
    w = 0.35
    ax2.bar(x - w / 2, mc_vals, w, label="MC muatm", color="#2ca02c", alpha=0.75)
    ax2.bar(x + w / 2, exp_vals, w, label="EXP", color="#1f77b4", alpha=0.75)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels_list, fontsize=8, rotation=30, ha="right")
    ax2.set_ylabel("Selection efficiency (%)")
    ax2.set_title("NN Cut Efficiency (8h, 2s @ P>0.5)")
    ax2.legend()
    ax2.grid(True, axis="y", alpha=0.3)
    fig2.savefig(args.out_dir / "selection_efficiency.png", dpi=130)
    plt.close(fig2)

    print(f"Wrote {table_csv}")
    print(f"Wrote {args.out_dir / 'metrics_vs_round.png'}")
    print(f"Wrote {args.out_dir / 'selection_efficiency.png'}")


if __name__ == "__main__":
    main()
