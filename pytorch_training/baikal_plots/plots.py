"""Reusable publication-quality plot functions for Baikal-ML analyses.

All functions accept plain numpy arrays / lists — no coupling to specific
data loaders or model code. Import the ``style`` module and call
``style.apply()`` once before using these.

Typical usage::

    from baikal_plots import style, plots
    style.apply()

    plots.pr_vs_threshold(
        thresholds, precisions, recalls,
        title="EAS", out_path="pr_eas.png",
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

from . import style

# ===================================================================
# Precision / Recall vs threshold
# ===================================================================


def pr_vs_threshold(
    thresholds: np.ndarray,
    precision: np.ndarray,
    recall: np.ndarray,
    *,
    title: str = "",
    y_min: float = style.METRIC_Y_MIN,
    ax: Optional[plt.Axes] = None,
    out_path: Optional[str] = None,
    prec_err: Optional[np.ndarray] = None,
    rec_err: Optional[np.ndarray] = None,
    legend_loc: str = "lower center",
) -> plt.Axes:
    """Precision & recall vs classification threshold on a single axes.

    Parameters
    ----------
    thresholds : 1-D array of threshold values (x-axis).
    precision, recall : 1-D arrays (same length as *thresholds*).
    prec_err, rec_err : optional error bars → drawn as shaded bands.
    title : panel title.
    y_min : lower y-axis bound.
    ax : existing axes to draw on; a new figure is created if *None*.
    out_path : if given, saves the figure and closes it.
    """
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(5.6, 4.6), constrained_layout=True)

    if prec_err is not None:
        m = np.isfinite(precision)
        ax.fill_between(
            thresholds[m],
            np.clip(precision[m] - prec_err[m], 0, 1),
            np.clip(precision[m] + prec_err[m], 0, 1),
            color=style.PRECISION_COL,
            alpha=0.18,
            linewidth=0,
        )
    if rec_err is not None:
        m = np.isfinite(recall)
        ax.fill_between(
            thresholds[m],
            np.clip(recall[m] - rec_err[m], 0, 1),
            np.clip(recall[m] + rec_err[m], 0, 1),
            color=style.RECALL_COL,
            alpha=0.18,
            linewidth=0,
        )

    ax.plot(thresholds, precision, lw=style.CURVE_LW, color=style.PRECISION_COL, label="precision")
    ax.plot(thresholds, recall, lw=style.CURVE_LW, ls="--", color=style.RECALL_COL, label="recall")
    ax.set_xlabel(style.XI_LABEL)
    if title:
        ax.set_title(title)
    style.threshold_xaxis(ax)
    style.metric_yaxis(ax, y_min=y_min)
    ax.legend(loc=legend_loc)

    if out_path and own_fig:
        style.savefig(fig, out_path)
    return ax


def pr_panels(
    panel_data: list[dict],
    *,
    out_path: Optional[str] = None,
    y_min: float = 0.85,
    panel_size: tuple[float, float] = (5.6, 4.6),
) -> plt.Figure:
    """Multiple side-by-side P/R-vs-threshold panels.

    Parameters
    ----------
    panel_data : list of dicts, each with keys:
        - ``thresholds``, ``precision``, ``recall`` (required)
        - ``title`` (optional, default "")
        - ``prec_err``, ``rec_err`` (optional)
    out_path : saves and closes figure if given.

    Returns the Figure object.
    """
    n = len(panel_data)
    fig, axes = style.panel_grid(n, panel_size=panel_size)
    for ax, pd in zip(axes, panel_data):
        pr_vs_threshold(
            pd["thresholds"],
            pd["precision"],
            pd["recall"],
            title=pd.get("title", ""),
            y_min=y_min,
            ax=ax,
            prec_err=pd.get("prec_err"),
            rec_err=pd.get("rec_err"),
        )
    if out_path:
        style.savefig(fig, out_path)
    return fig


# ===================================================================
# Metric vs continuous variable (energy, |t_res|, etc.)
# ===================================================================


def metric_vs_variable(
    x: np.ndarray,
    precision: np.ndarray,
    recall: np.ndarray,
    *,
    xlabel: str = "",
    title: str = "",
    y_min: float = style.METRIC_Y_MIN,
    ax: Optional[plt.Axes] = None,
    out_path: Optional[str] = None,
    marker_prec: str = "o",
    marker_rec: str = "s",
    legend_loc: str = "lower right",
) -> plt.Axes:
    """Precision & recall vs a continuous variable (e.g. energy, |t_res| bin).

    Parameters
    ----------
    x : 1-D array of bin centers or category positions.
    precision, recall : 1-D arrays.
    xlabel : x-axis label.
    """
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)

    ax.plot(
        x,
        precision,
        f"{marker_prec}-",
        lw=style.CURVE_LW,
        color=style.PRECISION_COL,
        label="precision",
        markersize=7,
    )
    ax.plot(
        x,
        recall,
        f"{marker_rec}--",
        lw=style.CURVE_LW,
        color=style.RECALL_COL,
        label="recall",
        markersize=7,
    )
    if xlabel:
        ax.set_xlabel(xlabel)
    if title:
        ax.set_title(title)
    ax.minorticks_on()
    style.metric_yaxis(ax, y_min=y_min)
    ax.legend(loc=legend_loc)

    if out_path and own_fig:
        style.savefig(fig, out_path)
    return ax


def metric_vs_variable_panels(
    panel_data: list[dict],
    *,
    out_path: Optional[str] = None,
    panel_size: tuple[float, float] = (5.6, 4.6),
) -> plt.Figure:
    """Multiple side-by-side metric-vs-variable panels.

    Parameters
    ----------
    panel_data : list of dicts, each with keys:
        - ``x``, ``precision``, ``recall`` (required)
        - ``xlabel``, ``title``, ``y_min``, ``legend_loc`` (optional)
    """
    n = len(panel_data)
    fig, axes = style.panel_grid(n, panel_size=panel_size)
    for ax, pd in zip(axes, panel_data):
        metric_vs_variable(
            pd["x"],
            pd["precision"],
            pd["recall"],
            xlabel=pd.get("xlabel", ""),
            title=pd.get("title", ""),
            y_min=pd.get("y_min", style.METRIC_Y_MIN),
            ax=ax,
            legend_loc=pd.get("legend_loc", "lower right"),
        )
    if out_path:
        style.savefig(fig, out_path)
    return fig


# ===================================================================
# Score distributions (MC vs EXP)
# ===================================================================


def score_distributions(
    scores_dict: dict[str, np.ndarray],
    *,
    colors: Optional[dict[str, str]] = None,
    xlabel: str = "",
    title: str = "",
    log_y: bool = True,
    bins: int | np.ndarray = 100,
    ax: Optional[plt.Axes] = None,
    out_path: Optional[str] = None,
    density: bool = True,
) -> plt.Axes:
    """Overlay step histograms of scores for multiple datasets.

    Parameters
    ----------
    scores_dict : ``{"MC EAS": array, "EXP": array, ...}``.
    colors : per-key colors; defaults to MC_COL / EXP_COL for first two.
    """
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)

    default_cols = [style.MC_COL, style.EXP_COL] + list(style.EVTYPE_COLS.values())
    if colors is None:
        colors = {}
    for i, name in enumerate(scores_dict):
        col = colors.get(name, default_cols[i % len(default_cols)])
        ax.hist(
            scores_dict[name],
            bins=bins,
            density=density,
            histtype="step",
            lw=style.HIST_LW,
            color=col,
            label=name,
        )
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel or style.XI_LABEL)
    ax.set_ylabel("Density" if density else "Count")
    if title:
        ax.set_title(title)
    style.threshold_xaxis(ax)
    ax.legend(loc="upper center")
    ax.grid(True, which="major", alpha=0.3)

    if out_path and own_fig:
        style.savefig(fig, out_path)
    return ax


# ===================================================================
# Prediction error plot (regression)
# ===================================================================


def prediction_error(
    true_vals: np.ndarray,
    pred_vals: np.ndarray,
    *,
    bins: Optional[np.ndarray] = None,
    log_x: Optional[bool] = None,
    xlabel: str = r"true $|t_\mathrm{res}|$, ns",
    ylabel: str = r"$|\,\hat{t}_\mathrm{res} - t_\mathrm{res}\,|$, ns",
    title: str = "",
    ax: Optional[plt.Axes] = None,
    out_path: Optional[str] = None,
) -> plt.Axes:
    """Median/mean prediction error vs binned true value, with IQR band.

    Parameters
    ----------
    true_vals, pred_vals : 1-D arrays of true and predicted values.
    bins : bin edges for |true_vals|.  Default: [0,2,5,10,20,40,80,160,320].
    log_x : use log scale for x-axis.  Auto-detected from bin range if None.
    """
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)

    finite = np.isfinite(true_vals) & np.isfinite(pred_vals)
    if not finite.any():
        return ax
    t = np.abs(true_vals[finite])
    p = np.abs(pred_vals[finite])
    err = np.abs(t - p)

    if bins is None:
        bins = np.array([0, 2, 5, 10, 20, 40, 80, 160, 320], dtype=float)
    bins = np.asarray(bins, dtype=float)
    if log_x is None:
        log_x = bool(bins.max() / max(bins[bins > 0].min(), 1e-9) > 30)

    medians, means, q25, q75 = [], [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (t >= lo) & (t < hi)
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
    valid = np.isfinite(medians)

    if log_x:
        centers = np.array([np.sqrt(max(lo, 0.5) * hi) for lo, hi in zip(bins[:-1], bins[1:])])
    else:
        centers = 0.5 * (bins[:-1] + bins[1:])

    ax.fill_between(
        centers[valid],
        q25[valid],
        q75[valid],
        color=style.PRECISION_COL,
        alpha=0.18,
        label="IQR",
        linewidth=0,
    )
    ax.plot(
        centers[valid],
        medians[valid],
        "o-",
        lw=style.CURVE_LW,
        color=style.PRECISION_COL,
        label="median",
        markersize=7,
    )
    ax.plot(
        centers[valid],
        means[valid],
        "s--",
        lw=2.2,
        color=style.RECALL_COL,
        label="mean",
        markersize=7,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if log_x:
        ax.set_xscale("log")
        ax.set_xlim(left=0.8)
    style.minor_grid(ax)
    ax.legend(loc="lower right" if log_x else "upper left")

    if out_path and own_fig:
        style.savefig(fig, out_path)
    return ax


# ===================================================================
# Grouped bar chart (e.g. Wasserstein distance comparison)
# ===================================================================


def grouped_bars(
    categories: Sequence[str],
    groups: dict[str, Sequence[float]],
    *,
    colors: Optional[dict[str, str]] = None,
    ylabel: str = "",
    title: str = "",
    annotate: bool = True,
    fmt: str = ".4f",
    ax: Optional[plt.Axes] = None,
    out_path: Optional[str] = None,
    bar_width: float = 0.35,
) -> plt.Axes:
    """Side-by-side grouped bar chart.

    Parameters
    ----------
    categories : x-axis tick labels (e.g. ["no cut", "8-2 cut"]).
    groups : ``{"No DA": [0.12, 0.08], "With DA": [0.05, 0.03]}``.
    colors : per-group colors.
    annotate : write values on top of bars.
    """
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)

    default_cols = [style.EXP_COL, style.PRECISION_COL, style.RECALL_COL, style.MC_COL]
    if colors is None:
        colors = {}
    n_groups = len(groups)
    x = np.arange(len(categories))
    offsets = np.linspace(-(n_groups - 1) / 2, (n_groups - 1) / 2, n_groups) * bar_width

    for i, (gname, vals) in enumerate(groups.items()):
        col = colors.get(gname, default_cols[i % len(default_cols)])
        bars = ax.bar(x + offsets[i], vals, bar_width * 0.9, label=gname, color=col, alpha=0.85)
        if annotate:
            for bar, v in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{v:{fmt}}",
                    ha="center",
                    va="bottom",
                    fontsize=11,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    if out_path and own_fig:
        style.savefig(fig, out_path)
    return ax


# ===================================================================
# Distribution comparison (MC vs EXP histograms)
# ===================================================================


def compare_distributions(
    data_dict: dict[str, np.ndarray],
    *,
    colors: Optional[dict[str, str]] = None,
    xlabel: str = "",
    ylabel: str = "Density",
    title: str = "",
    bins: int | np.ndarray = 50,
    log_y: bool = False,
    log_x: bool = False,
    density: bool = True,
    histtype: str = "step",
    ax: Optional[plt.Axes] = None,
    out_path: Optional[str] = None,
) -> plt.Axes:
    """Overlay histograms from multiple datasets.

    Parameters
    ----------
    data_dict : ``{"MC EAS": array, "EXP": array}``.
    """
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)

    default_cols = [style.MC_COL, style.EXP_COL] + list(style.EVTYPE_COLS.values())
    if colors is None:
        colors = {}
    for i, (name, vals) in enumerate(data_dict.items()):
        col = colors.get(name, default_cols[i % len(default_cols)])
        ax.hist(
            vals,
            bins=bins,
            density=density,
            histtype=histtype,
            lw=style.HIST_LW,
            color=col,
            label=name,
        )
    if log_y:
        ax.set_yscale("log")
    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend()
    ax.grid(True, which="major", alpha=0.3)

    if out_path and own_fig:
        style.savefig(fig, out_path)
    return ax


# ===================================================================
# Scatter / 2D comparison (e.g. UMAP, pred vs true)
# ===================================================================


def scatter_comparison(
    datasets: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    colors: Optional[dict[str, str]] = None,
    xlabel: str = "UMAP-1",
    ylabel: str = "UMAP-2",
    title: str = "",
    alpha: float = 0.3,
    s: float = 3,
    ax: Optional[plt.Axes] = None,
    out_path: Optional[str] = None,
) -> plt.Axes:
    """Overlay scatter plots from multiple datasets.

    Parameters
    ----------
    datasets : ``{"MC EAS": (x, y), "EXP": (x, y)}``.
    """
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)

    default_cols = [style.MC_COL, style.EXP_COL] + list(style.EVTYPE_COLS.values())
    if colors is None:
        colors = {}
    for i, (name, (xd, yd)) in enumerate(datasets.items()):
        col = colors.get(name, default_cols[i % len(default_cols)])
        ax.scatter(xd, yd, c=col, s=s, alpha=alpha, label=name, rasterized=True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend(markerscale=3)
    ax.grid(True, alpha=0.2)

    if out_path and own_fig:
        style.savefig(fig, out_path)
    return ax
