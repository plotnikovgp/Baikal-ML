from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


class BasePlotter:
    def __init__(self, save_dir: Path):
        self.save_dir = save_dir

    def plot(self, data: dict):
        pass


TITILE_FONT_SIZE = 16
LABEL_FONT_SIZE = 14
TICK_FONT_SIZE = 12
LEGEND_FONT_SIZE = 12
CMAP = "viridis"
PERCENTILE_COLOR = "#FF00FF"


class AngleUncertaintyPlotter(BasePlotter):
    def _plot_coordinate_uncertainties(self, data: dict):
        """
        Plot uncertainty evaluation for X, Y, Z coordinates.
        Shows predicted sigma vs. actual error with 68.2%       central percentile lines.

        Args:
            data: Dictionary containing 'pred_sigma2', 'y_pred', and 'y_true'
        """
        import matplotlib.colors as colors

        pred_xyz = data["y_pred"]
        true_xyz = data["y_true"]
        error_xyz = true_xyz - pred_xyz

        # Get predicted sigmas (convert from variance to std dev)
        sigma_xyz = np.sqrt(data["pred_sigma2"])

        # Create figure with 3 subplots for x, y, z
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(
            "Uncertainty Predictor Evaluation for Coordinates",
            fontsize=TITILE_FONT_SIZE,
            fontweight="bold",
        )

        coord_labels = ["x", "y", "z"]

        for i, (ax, coord) in enumerate(zip(axs, coord_labels)):
            sigma = sigma_xyz[:, i]
            error = error_xyz[:, i]
            # remove nan values
            mask = ~np.isnan(sigma) & ~np.isnan(error)
            sigma = sigma[mask]
            error = error[mask]
            # Create 2D histogram
            hist = ax.hist2d(
                sigma,
                error,
                bins=[60, 30],
                density=True,
                norm=colors.LogNorm(vmin=1e-3),
                cmap=CMAP,
            )
            ax.grid(True, linestyle=":")
            ax.set_xlabel(
                f"Predicted $\\mathbf{{\\sigma}}$ ({coord})",
                fontsize=LABEL_FONT_SIZE,
                fontweight="bold",
            )
            ax.set_ylabel(f"{coord} - {coord}*", fontsize=LABEL_FONT_SIZE, fontweight="bold")
            cbar = fig.colorbar(hist[3], ax=ax)
            cbar.ax.tick_params(labelsize=TICK_FONT_SIZE)

            max_sigma = np.percentile(sigma, 97)
            x = np.linspace(0, max_sigma, 30)
            ax.plot(x, x, "--", color="black", linewidth=2, label="One sigma")
            ax.plot(x, -x, "--", color="black", linewidth=2)

            # Calculate percentiles
            plus_quantile = []
            minus_quantile = []
            x_quantile = []

            for x_value_0, x_value_1 in zip(x[0:-1], x[1:]):
                mask = (sigma >= x_value_0) & (sigma < x_value_1)
                if mask.sum() < 6:  # minimum sample size
                    continue
                plus_quantile.append(np.quantile(error[mask], 0.5 + 0.341))
                minus_quantile.append(np.quantile(error[mask], 0.5 - 0.341))
                x_quantile.append((x_value_0 + x_value_1) / 2)

            # Plot percentiles
            # Update the percentile lines
            if len(x_quantile) > 0:
                # Update the percentile lines
                ax.plot(
                    x_quantile,
                    plus_quantile,
                    linestyle="-",
                    color=PERCENTILE_COLOR,
                    linewidth=3,
                    label="68.2% central percentile",
                )
                ax.plot(
                    x_quantile,
                    minus_quantile,
                    linestyle="-",
                    color=PERCENTILE_COLOR,
                    linewidth=3,
                )

            # Set limits
            ax.set_xlim(0, max_sigma)

        # After plotting all elements, add a legend to each plot
        for i, ax in enumerate(axs):
            # Only add legend to the last plot to avoid redundancy
            if i == 2:  # Add only to the z-axis plot
                legend = ax.legend(
                    fontsize=LEGEND_FONT_SIZE,
                    frameon=True,
                    fancybox=True,
                    framealpha=0.9,
                    edgecolor="gray",
                    loc="upper right",
                )
                legend.get_frame().set_linewidth(1.5)

        plt.tight_layout()
        plt.savefig(
            self.save_dir / "coordinate_uncertainty_evaluation.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def _plot_angle_uncertainties(self, data: dict):
        import matplotlib.colors as colors

        # Extract theta and phi data
        sigma_theta = np.sqrt(data["pred_theta_sigma2"])
        error_theta = data["true_theta"] - data["pred_theta"]

        sigma_phi = np.sqrt(data["pred_phi_sigma2"])
        error_phi = data["true_phi"] - data["pred_phi"]

        # Create figure
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(
            "Uncertainty Predictor Evaluation for Angular Components",
            fontsize=TITILE_FONT_SIZE,
            fontweight="bold",
        )

        angle_data = [
            (sigma_theta, error_theta, "theta", "Polar Angle ($\\mathbf{{\\theta}}$)"),
            (sigma_phi, error_phi, "phi", "Azimuthal Angle ($\\mathbf{{\\phi}}$)"),
        ]

        for i, (sigma, error, symbol, title) in enumerate(angle_data):
            ax = axs[i]

            # Filter out NaN or Inf values
            valid_mask = ~(np.isnan(sigma) | np.isnan(error) | np.isinf(sigma) | np.isinf(error))
            sigma_valid = sigma[valid_mask]
            error_valid = error[valid_mask]

            if len(sigma_valid) == 0:
                ax.text(
                    0.5,
                    0.5,
                    "No valid data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                continue

            # Set limits based on quantiles
            if symbol == "theta":
                sigma_min, sigma_max = np.percentile(sigma_valid, [3, 97])
                error_min, error_max = np.percentile(error_valid, [3, 97])
            else:
                sigma_min, sigma_max = np.percentile(sigma_valid, [3, 93])
                error_min, error_max = np.percentile(error_valid, [3, 93])

            # Ensure limits are symmetric for error
            error_limit = max(abs(error_min), abs(error_max))

            # Create 2D histogram with quantile-based limits
            norm = colors.LogNorm(vmin=1e-5) if symbol == "phi" else colors.LogNorm(vmin=1e-4)
            hist = ax.hist2d(
                sigma_valid,
                error_valid,
                bins=50,
                range=[[0, sigma_max], [-error_limit, error_limit]],
                density=True,
                norm=norm,
                cmap=CMAP,
            )
            ax.grid(True, linestyle=":")
            ax.set_xlabel(
                f"Predicted $\\mathbf{{\\sigma}}$({symbol})",
                fontsize=LABEL_FONT_SIZE,
                fontweight="bold",
            )
            ax.set_ylabel(f"{symbol} - {symbol}*", fontsize=LABEL_FONT_SIZE, fontweight="bold")

            # Add colorbar
            cbar = fig.colorbar(hist[3], ax=ax)
            cbar.ax.tick_params(labelsize=TICK_FONT_SIZE)

            # Add reference lines - only up to the max sigma value
            x = np.linspace(0, sigma_max, 100)
            ax.plot(x, x, "--", color="black", linewidth=2, label="One sigma")
            ax.plot(x, -x, "--", color="black", linewidth=2)

            # Calculate percentiles
            plus_quantile = []
            minus_quantile = []
            x_quantile = []

            # Use more bins for better resolution in percentile calculation
            bin_edges = np.linspace(0, sigma_max, 25)

            for x_value_0, x_value_1 in zip(bin_edges[:-1], bin_edges[1:]):
                mask = (sigma_valid >= x_value_0) & (sigma_valid < x_value_1)
                if mask.sum() < 10:  # Increased minimum sample size
                    continue
                plus_quantile.append(np.quantile(error_valid[mask], 0.5 + 0.341))
                minus_quantile.append(np.quantile(error_valid[mask], 0.5 - 0.341))
                x_quantile.append((x_value_0 + x_value_1) / 2)

            # Plot percentiles
            if len(x_quantile) > 0:
                # Update the percentile lines
                ax.plot(
                    x_quantile,
                    plus_quantile,
                    linestyle="-",
                    color=PERCENTILE_COLOR,
                    linewidth=3,
                    label="68.2% central percentile",
                )
                ax.plot(
                    x_quantile,
                    minus_quantile,
                    linestyle="-",
                    color=PERCENTILE_COLOR,
                    linewidth=3,
                )

            # Set limits explicitly
            ax.set_xlim(0, sigma_max)
            ax.set_ylim(-error_limit, error_limit)

        # After plotting all elements, add a legend to the phi plot
        axs[1].legend(
            fontsize=LEGEND_FONT_SIZE,
            frameon=True,
            fancybox=True,
            framealpha=0.9,
            edgecolor="gray",
            loc="upper right",
        )
        legend = axs[1].get_legend()
        legend.get_frame().set_linewidth(1.5)

        plt.tight_layout()
        plt.savefig(
            self.save_dir / "angle_uncertainty_evaluation.png",
            dpi=300,
            bbox_inches="tight",
        )

        # # Print diagnostics
        # print(f"Theta sigma range: {sigma_theta.min():.4f} to {sigma_theta.max():.4f}")
        # print(f"Theta error range: {error_theta.min():.4f} to {error_theta.max():.4f}")
        # print(f"Phi sigma range: {sigma_phi.min():.4f} to {sigma_phi.max():.4f}")
        # print(f"Phi error range: {error_phi.min():.4f} to {error_phi.max():.4f}")

        plt.close()

    def _plot_cmp(self, x, y, ax, title):
        ax.hexbin(x, y, gridsize=50, cmap=CMAP, bins="log")
        alpha = 0.05
        min_v = min(np.percentile(x, alpha), np.percentile(y, alpha))
        max_v = max(np.percentile(x, 100 - alpha), np.percentile(y, 100 - alpha))
        ax.plot([min_v, max_v], [min_v, max_v], "k--", lw=1.5, label="$y=x$")
        ax.set_xlabel(rf"$\log \sigma_{{\mathrm{{pred}},\,{title}}}$", fontweight="bold")
        ax.set_ylabel(rf"$\log \sigma_{{\mathrm{{true}},\,{title}}}$", fontweight="bold")
        ax.set_title(f"{title}", fontweight="bold")
        ax.legend()

    def _plot_coordinate_uncertainty_metrics(self, data: dict):
        fig, ax = plt.subplots(ncols=3, nrows=1, figsize=(15, 5))
        # x, y, z of predicted sigma2 vs x, y, z of true sigma2
        x = data["pred_sigma2"]
        y = data["true_sigma2"]
        self._plot_cmp(np.log(x[:, 0]), np.log(y[:, 0]), ax[0], "x")
        self._plot_cmp(np.log(x[:, 1]), np.log(y[:, 1]), ax[1], "y")
        self._plot_cmp(np.log(x[:, 2]), np.log(y[:, 2]), ax[2], "z")

        fig.suptitle("Coordinate Uncertainty Comparison", fontsize=16, fontweight="bold")

        plt.savefig(
            self.save_dir / "angle_uncertainty_metrics_xyz.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

    def _plot_angle_uncertainty_metrics(self, data: dict):
        fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(10, 5))
        # x = predicted_sigma2_phi, y = true_sigma2_phi
        x = data["pred_phi_sigma2"]
        y = data["true_phi_sigma2"]
        self._plot_cmp(np.log(x), np.log(y), ax[0], "phi")

        x = data["pred_theta_sigma2"]
        y = data["true_theta_sigma2"]
        self._plot_cmp(np.log(x), np.log(y), ax[1], "theta")

        fig.suptitle("Angular Uncertainty Comparison", fontsize=16, fontweight="bold")

        plt.savefig(
            self.save_dir / "angle_uncertainty_metrics.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

    def plot(self, data: dict):
        sns.set_style("whitegrid")
        plt.rcParams.update(
            {
                "font.family": "sans-serif",
                "font.sans-serif": ["Arial", "DejaVu Sans"],
                "font.size": TICK_FONT_SIZE,
                "axes.titlesize": TITILE_FONT_SIZE,
                "axes.labelsize": LABEL_FONT_SIZE,
                "xtick.labelsize": TICK_FONT_SIZE,
                "ytick.labelsize": TICK_FONT_SIZE,
                "legend.fontsize": LEGEND_FONT_SIZE,
            }
        )
        # self._plot_coordinate_uncertainty_metrics(data)
        self._plot_angle_uncertainty_metrics(data)
        # self._plot_coordinate_uncertainties(data)
        self._plot_angle_uncertainties(data)
