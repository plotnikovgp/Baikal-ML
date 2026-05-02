"""baikal_plots — publication-quality plotting for Baikal-ML.

Quick start::

    from baikal_plots import style, plots
    style.apply()

    # P/R vs threshold
    plots.pr_vs_threshold(thresholds, precision, recall, out_path="pr.png")

    # MC vs EXP score distributions
    plots.score_distributions({"MC EAS": mc_scores, "EXP": exp_scores}, out_path="scores.png")

    # Prediction error
    plots.prediction_error(true_tres, pred_tres, out_path="tres_error.png")

    # Grouped bars (e.g. Wasserstein distance)
    plots.grouped_bars(["no cut", "8-2"], {"No DA": [0.1, 0.08], "DA": [0.05, 0.03]},
                       ylabel="Wasserstein distance", out_path="wass.png")
"""

from . import plots, style

__all__ = ["style", "plots"]
