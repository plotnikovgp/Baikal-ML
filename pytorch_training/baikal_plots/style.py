"""Baikal-ML publication-quality plot styling.

Usage:
    from baikal_plots import style
    style.apply()  # call once at the start of your script

All constants are available as module-level attributes:
    style.MC_COL, style.EXP_COL, style.PRECISION_COL, ...
"""

from pathlib import Path
from typing import Optional, Sequence

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------
MC_COL = "#2ca02c"
EXP_COL = "#1f77b4"

EVTYPE_COLS = {
    "muatm": "#d62728",
    "nuatm": "#ff7f0e",
    "nue2": "#9467bd",
}

PRECISION_COL = "#D55E00"
RECALL_COL = "#009E73"

# ---------------------------------------------------------------------------
# Event-type labels (LaTeX-compatible for matplotlib)
# ---------------------------------------------------------------------------
EVTYPE_LABELS = {
    "muatm": "EAS",
    "nuatm": r"atmospheric $\nu_\mu$",
    "nue2": r"cosmogenic $\nu_\mu$",
}

# ---------------------------------------------------------------------------
# Axis-tuning defaults
# ---------------------------------------------------------------------------
XI_LABEL = r"$\xi$"

METRIC_Y_MIN = 0.7
METRIC_Y_MAX = 1.01
METRIC_Y_MAJOR = 0.05
METRIC_Y_MINOR = 0.01

THRESHOLD_X_MAJOR = 0.2
THRESHOLD_X_MINOR = 0.05

# Line styles for precision / recall
P_LINESTYLE = "-"
R_LINESTYLE = "--"

# Default line width for curves
CURVE_LW = 2.4
HIST_LW = 2.4

# Default DPI for saved PNGs
SAVE_DPI = 160

# ---------------------------------------------------------------------------
# rcParams (global matplotlib style)
# ---------------------------------------------------------------------------
RC_PARAMS = {
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

_applied = False


def apply(backend: str = "Agg"):
    """Apply Baikal plot style globally. Safe to call multiple times."""
    global _applied
    if not _applied:
        matplotlib.use(backend)
        _applied = True
    plt.rcParams.update(RC_PARAMS)


# ---------------------------------------------------------------------------
# Axis helpers
# ---------------------------------------------------------------------------
def metric_yaxis(ax, y_min: float = METRIC_Y_MIN, y_max: float = METRIC_Y_MAX):
    """Configure y-axis for metric plots (precision, recall, etc.).

    Automatically picks major/minor tick spacing based on the axis span.
    Adds both major and minor gridlines.
    """
    ax.set_ylim(y_min, y_max)
    span = y_max - y_min
    if span <= 0.10:
        major, minor = 0.01, 0.005
    elif span <= 0.25:
        major, minor = 0.02, 0.01
    else:
        major, minor = METRIC_Y_MAJOR, METRIC_Y_MINOR
    ax.yaxis.set_major_locator(MultipleLocator(major))
    ax.yaxis.set_minor_locator(MultipleLocator(minor))
    ax.tick_params(which="major", length=6)
    ax.tick_params(which="minor", length=3)
    ax.grid(True, which="major", alpha=0.35)
    ax.grid(True, which="minor", alpha=0.18, linestyle=":")


def threshold_xaxis(ax, x_min: float = 0.0, x_max: float = 1.0):
    """Configure x-axis for threshold/score sweep plots.

    Sets range [x_min, x_max] with standard major/minor ticks and gridlines.
    """
    ax.set_xlim(x_min, x_max)
    ax.xaxis.set_major_locator(MultipleLocator(THRESHOLD_X_MAJOR))
    ax.xaxis.set_minor_locator(MultipleLocator(THRESHOLD_X_MINOR))
    ax.tick_params(which="major", length=6)
    ax.tick_params(which="minor", length=3)
    ax.grid(True, which="major", axis="x", alpha=0.35)
    ax.grid(True, which="minor", axis="x", alpha=0.18, linestyle=":")


def minor_grid(ax, which: str = "both"):
    """Add minor gridlines to an axis."""
    ax.minorticks_on()
    ax.grid(True, which="major", alpha=0.35)
    ax.grid(True, which="minor", alpha=0.18, linestyle=":")


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------
def panel_grid(
    n_panels: int,
    panel_size: tuple[float, float] = (5.6, 4.6),
    **subplot_kw,
) -> tuple[plt.Figure, list[plt.Axes]]:
    """Create a 1×N panel figure with constrained layout.

    Returns (fig, [ax1, ax2, ...]).
    """
    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(panel_size[0] * n_panels, panel_size[1]),
        constrained_layout=True,
        **subplot_kw,
    )
    if n_panels == 1:
        axes = [axes]
    return fig, list(axes)


def savefig(fig: plt.Figure, path, dpi: int = SAVE_DPI, close: bool = True):
    """Save figure as PNG with tight bounding box. Creates parent dirs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    if close:
        plt.close(fig)
    print(f"[baikal_plots] Saved {path}")


# ---------------------------------------------------------------------------
# Legend helpers
# ---------------------------------------------------------------------------
def mc_exp_legend(ax, mc_label: str = "MC EAS", exp_label: str = "EXP", **kwargs):
    """Add a standard MC vs EXP legend."""
    from matplotlib.lines import Line2D

    handles = [
        Line2D([], [], color=MC_COL, lw=CURVE_LW, label=mc_label),
        Line2D([], [], color=EXP_COL, lw=CURVE_LW, label=exp_label),
    ]
    ax.legend(handles=handles, **kwargs)


def pr_legend(ax, **kwargs):
    """Add a standard Precision / Recall legend."""
    from matplotlib.lines import Line2D

    handles = [
        Line2D([], [], color=PRECISION_COL, ls=P_LINESTYLE, lw=CURVE_LW, label="Precision"),
        Line2D([], [], color=RECALL_COL, ls=R_LINESTYLE, lw=CURVE_LW, label="Recall"),
    ]
    ax.legend(handles=handles, **kwargs)
