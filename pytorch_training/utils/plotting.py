import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm


def plot_precision_recall_by_event_type(
    y_pred_prob,
    y_true,
    ev_types,
    thresholds=None,
    save_path=None,
    figsize=(16, 5),
    ev_starts=None,
    subsample=5,
):
    y_pred_prob = np.asarray(y_pred_prob)
    y_true = np.asarray(y_true)
    ev_types = np.asarray(ev_types, dtype=int)

    if thresholds is None:
        thresholds = np.arange(0.2, 0.8, 0.03)

    if len(ev_types) != len(y_pred_prob):
        if ev_starts is None:
            print(
                f"Warning: ev_types length ({len(ev_types)}) != y_pred_prob length ({len(y_pred_prob)})"
            )
            print("Need ev_starts to map events to hits. Skipping plot.")
            return None

        ev_starts = np.asarray(ev_starts)
        if len(ev_types) == len(ev_starts) - 1:
            ev_types_expanded = np.zeros(len(y_pred_prob), dtype=int)
            for i in range(len(ev_starts) - 1):
                ev_types_expanded[ev_starts[i] : ev_starts[i + 1]] = ev_types[i]
            ev_types = ev_types_expanded
        else:
            raise ValueError("ev_starts length must be len(ev_types)+1")

    def particle_type_from_id(ev_id):
        if ev_id == 0:
            return "muatm"
        elif ev_id == 1:
            return "nuatm"
        elif ev_id == 2:
            return "nue2"
        return "unknown"

    ev_types_str = np.array([particle_type_from_id(eid) for eid in ev_types])
    particle_types = ["muatm", "nuatm", "nue2"]

    fig, axes = plt.subplots(1, 3, figsize=figsize, tight_layout=True)

    for idx, part_type in enumerate(particle_types):
        ax = axes[idx]

        mask = ev_types_str == part_type
        if not mask.any():
            ax.text(
                0.5,
                0.5,
                f"No {part_type} events found",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=14,
            )
            ax.set_title(f"{part_type.upper()}", fontsize=14, fontweight="bold")
            continue

        probs_type = y_pred_prob[mask][::subsample]
        labels_type = y_true[mask][::subsample]

        precisions, recalls, f1s = [], [], []

        for thresh in tqdm(thresholds, desc=f"Computing {part_type}", leave=False):
            preds = (probs_type > thresh).astype(int)
            precisions.append(precision_score(labels_type, preds, zero_division=0))
            recalls.append(recall_score(labels_type, preds, zero_division=0))
            f1s.append(f1_score(labels_type, preds, zero_division=0))

        precisions = np.array(precisions)
        recalls = np.array(recalls)
        f1s = np.array(f1s)

        ax.plot(
            thresholds,
            precisions,
            label="Precision",
            linewidth=1.5,
            color="blue",
            linestyle="-",
            alpha=0.8,
        )
        ax.plot(
            thresholds,
            recalls,
            label="Recall",
            linewidth=1.5,
            color="blue",
            linestyle="--",
            alpha=0.8,
        )

        best_idx = np.argmax(f1s)
        best_thresh = thresholds[best_idx]
        best_p, best_r = precisions[best_idx], recalls[best_idx]

        ax.axvline(x=best_thresh, color="blue", linestyle=":", alpha=0.4)
        ax.text(
            0.02,
            0.95,
            f"t={best_thresh:.2f}, P={best_p:.3f}, R={best_r:.3f}",
            transform=ax.transAxes,
            fontsize=9,
            color="blue",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        ax.set_xlabel("Threshold", fontsize=12)
        ax.set_ylabel("Score", fontsize=12)
        ax.set_title(f"{part_type.upper()}\n(n={mask.sum():,} hits)", fontsize=14)
        ax.legend(fontsize=10, loc="lower right")
        ax.grid(alpha=0.3, linestyle=":")
        ax.set_xlim(thresholds.min(), thresholds.max())

        min_val = min(precisions.min(), recalls.min())
        ax.set_ylim(max(0, min_val - 0.05), 1.02)

    plt.suptitle("Precision & Recall by Event Type", fontsize=16, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.show()
    return fig


def plot_threshold_curves(y_pred_prob, y_true, thresholds=None, save_path=None):
    if thresholds is None:
        thresholds = np.arange(0.1, 0.9, 0.02)

    y_pred_prob = np.asarray(y_pred_prob)
    y_true = np.asarray(y_true)

    precisions, recalls, f1s = [], [], []

    for thresh in thresholds:
        preds = (y_pred_prob > thresh).astype(int)
        precisions.append(precision_score(y_true, preds, zero_division=0))
        recalls.append(recall_score(y_true, preds, zero_division=0))
        f1s.append(f1_score(y_true, preds, zero_division=0))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(thresholds, precisions, label="Precision", linewidth=2)
    ax.plot(thresholds, recalls, label="Recall", linewidth=2)
    ax.plot(thresholds, f1s, label="F1", linewidth=2, linestyle="--")

    best_idx = np.argmax(f1s)
    ax.axvline(x=thresholds[best_idx], color="gray", linestyle=":", alpha=0.5)
    ax.scatter(
        [thresholds[best_idx]],
        [f1s[best_idx]],
        color="red",
        s=100,
        zorder=5,
        label=f"Best F1 @ {thresholds[best_idx]:.2f}",
    )

    ax.set_xlabel("Threshold", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Precision, Recall, F1 vs Threshold", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.show()
    return fig


def plot_prediction_distribution(y_pred_prob, y_true, bins=50, save_path=None):
    y_pred_prob = np.asarray(y_pred_prob)
    y_true = np.asarray(y_true, dtype=bool)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(y_pred_prob[y_true], bins=bins, alpha=0.6, label="Positive class", density=True)
    ax.hist(y_pred_prob[~y_true], bins=bins, alpha=0.6, label="Negative class", density=True)

    ax.set_xlabel("Predicted Probability", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Prediction Distribution by Class", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.show()
    return fig


def plot_regression_scatter(y_pred, y_true, save_path=None):
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(y_true, y_pred, alpha=0.3, s=10)

    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="y=x")

    ax.set_xlabel("True", fontsize=12)
    ax.set_ylabel("Predicted", fontsize=12)
    ax.set_title("Predicted vs True", fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.show()
    return fig
