```python
import sys
sys.path.insert(0, "/path/to/pytorch_training")
```

Dependencies: `matplotlib`, `numpy`.

## Quick start

```python
from baikal_plots import style, plots
import numpy as np

# Apply global style (call once)
style.apply()

# --- Precision / Recall vs threshold ---
thresholds = np.linspace(0.01, 0.99, 80)
precision = np.random.uniform(0.85, 1.0, len(thresholds))
recall = np.random.uniform(0.80, 1.0, len(thresholds))

plots.pr_vs_threshold(thresholds, precision, recall,
                      title="EAS", out_path="pr_eas.png")
```

## Available plot functions

### `plots.pr_vs_threshold`
Precision & recall vs classification threshold with styled axes and gridlines.

```python
plots.pr_vs_threshold(thresholds, precision, recall,
                      title="EAS", y_min=0.85,
                      prec_err=p_err, rec_err=r_err,  # optional error bands
                      out_path="pr.png")
```

### `plots.pr_panels`
Multiple side-by-side P/R panels (e.g. one per event type):

```python
plots.pr_panels([
    {"thresholds": thr, "precision": p1, "recall": r1, "title": "EAS"},
    {"thresholds": thr, "precision": p2, "recall": r2, "title": r"atm $\nu_\mu$"},
    {"thresholds": thr, "precision": p3, "recall": r3, "title": r"cosmo $\nu_\mu$"},
], y_min=0.85, out_path="pr_panels.png")
```

### `plots.metric_vs_variable`
Precision & recall vs any continuous variable (energy, |t_res|, etc.):

```python
plots.metric_vs_variable(
    energy_centers, precision, recall,
    xlabel=r"$\log_{10}(E\,/\,\mathrm{GeV})$",
    title="EAS", out_path="pr_vs_energy.png",
)
```

### `plots.metric_vs_variable_panels`
Multiple panels of metric-vs-variable:

```python
plots.metric_vs_variable_panels([
    {"x": centers, "precision": p, "recall": r, "title": "EAS",
     "xlabel": r"$\log_{10}(E)$"},
    # ...
], out_path="energy_panels.png")
```

### `plots.score_distributions`
Overlay step histograms of classification scores (MC vs EXP):

```python
plots.score_distributions(
    {"MC EAS": mc_scores, "EXP": exp_scores},
    xlabel=r"$\xi$", log_y=True,
    out_path="score_dist.png",
)
```

### `plots.prediction_error`
Regression error (median, mean, IQR) vs binned true value:

```python
plots.prediction_error(
    true_tres, pred_tres,
    bins=np.array([0, 2, 5, 10, 20, 40, 80, 160, 320]),
    out_path="tres_error.png",
)
```

### `plots.grouped_bars`
Side-by-side grouped bar chart (e.g. Wasserstein distance):

```python
plots.grouped_bars(
    ["no cut", "8-2 cut"],
    {"No DA": [0.12, 0.08], "With DA": [0.05, 0.03]},
    ylabel="Wasserstein distance",
    out_path="wasserstein.png",
)
```

### `plots.compare_distributions`
Generic histogram comparison for any variable:

```python
plots.compare_distributions(
    {"MC": charge_mc, "EXP": charge_exp},
    xlabel="Charge", log_y=True, bins=60,
    out_path="charge_dist.png",
)
```

### `plots.scatter_comparison`
Overlay scatter plots (e.g. UMAP embeddings):

```python
plots.scatter_comparison(
    {"MC EAS": (umap_mc[:, 0], umap_mc[:, 1]),
     "EXP":   (umap_exp[:, 0], umap_exp[:, 1])},
    out_path="umap.png",
)
```

## Styling API

All constants and helpers are in `style`:

```python
from baikal_plots import style

# Colors
style.MC_COL          # "#2ca02c" (green)
style.EXP_COL         # "#1f77b4" (blue)
style.PRECISION_COL   # "#D55E00" (orange)
style.RECALL_COL      # "#009E73" (teal)
style.EVTYPE_COLS     # {"muatm": red, "nuatm": orange, "nue2": purple}
style.EVTYPE_LABELS   # {"muatm": "EAS", "nuatm": r"atmospheric $\nu_\mu$", ...}

# Axis helpers (use on your own axes)
style.metric_yaxis(ax, y_min=0.85)    # smart major/minor ticks + grid
style.threshold_xaxis(ax)              # [0,1] with major/minor ticks + grid
style.minor_grid(ax)                   # add minor gridlines

# Figure helpers
fig, axes = style.panel_grid(3)        # 1×3 figure with constrained layout
style.savefig(fig, "out.png")          # save at 160 dpi, tight bbox

# Legend shortcuts
style.mc_exp_legend(ax)                # MC EAS / EXP legend
style.pr_legend(ax)                    # Precision / Recall legend
```

## Customization

Override any constant before plotting:

```python
style.MC_COL = "#ff0000"
style.SAVE_DPI = 300
style.apply()
```

Or modify `rcParams` directly after `style.apply()`:

```python
import matplotlib.pyplot as plt
style.apply()
plt.rcParams["font.size"] = 20
```
