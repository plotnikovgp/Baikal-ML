"""Evaluate uncertainty quality for angle reconstruction models.

Loads a trained UncertaintyPredictor (or any model that outputs direction + log_sigma2),
runs inference on the validation set, computes uncertainty-related metrics, and generates
diagnostic plots:
  - 2D histogram: predicted sigma vs. true angular error
  - Uncertainty cut curve: median/68% error vs. fraction of events passing the cut
  - Error vs. uncertainty binned plot with uncertainty distribution

Usage:
    python scripts/eval_uncertainty.py \
        --data /path/to/data.h5 \
        --checkpoint /path/to/best.ckpt \
        --output-dir plots/uncertainty_eval

    Or using an existing Hydra experiment config:
    python scripts/eval_uncertainty.py \
        --data /home/plotnikovgp/baikal/data/baikal_multi_0324_flat_treck-cascade_h8_s2_pureMC_normed-3strings-10hits.h5 \
        --checkpoint checkpoints/angle_reconstruction_sigma_v5_mc_0924/encoder_nl5_hs512_dff512_nh1_lr1e4_no-eas_10-3_mae_sigma_tune_new/best.ckpt \
        --output-dir plots/uncertainty_eval
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_utils import BaikalDatasetAngles, create_dataloaders
from data_utils.preprocessors import AnglePreprocessorWithTres, DataPrefilter
from metrics.plots import AngleUncertaintyPlotter
from models import Encoder, UncertaintyPredictor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DEFAULT_ENCODER_PARAMS = dict(
    in_features=5,
    hidden_size=512,
    num_layers=5,
    dim_feedforward_size=512,
    n_heads=1,
    out_size=3,
    dropout_p=0.0,
    use_cls_token=True,
    return_only_cls_token=True,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate uncertainty predictor")
    parser.add_argument("--data", type=str, required=True, help="Path to HDF5 dataset")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output-dir", type=str, default="plots/uncertainty_eval")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--predictor-hidden-size", type=int, default=512)
    parser.add_argument("--num-predictor-layers", type=int, default=1)
    parser.add_argument("--tres-cut", type=float, default=100000.0)
    parser.add_argument(
        "--max-batches", type=int, default=None, help="Limit number of batches for inference"
    )
    parser.add_argument("--n-heads", type=int, default=1)
    return parser.parse_args()


def build_model(
    checkpoint_path: str,
    encoder_params: dict,
    predictor_hidden_size: int,
    num_predictor_layers: int,
) -> torch.nn.Module:
    encoder = Encoder(**encoder_params)
    model = UncertaintyPredictor(
        model=encoder,
        hidden_size=predictor_hidden_size,
        num_predictor_layers=num_predictor_layers,
    )

    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


def get_predictions(model, dataloader, max_batches=None):
    all_preds = []
    all_trues = []

    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Inference")):
            if max_batches is not None and i >= max_batches:
                break
            if isinstance(batch, (list, tuple)):
                x, y_true, mask = batch[0], batch[1], batch[2]
            else:
                x, y_true, mask = batch
            x = x.to(DEVICE)
            mask = mask.to(DEVICE).bool()

            output = model(x, mask)
            all_preds.append(output.cpu().numpy())
            all_trues.append(y_true.numpy())

    return np.concatenate(all_preds), np.concatenate(all_trues)


def compute_ence(angular_error, mean_sigma, n_bins=20):
    """Expected Normalized Calibration Error.

    Bins events by predicted sigma, compares RMSE of actual errors (RMV)
    to mean predicted sigma in each bin. Perfect calibration => ENCE = 0.
    """
    bin_edges = np.percentile(mean_sigma, np.linspace(0, 100, n_bins + 1))
    bin_edges[-1] += 1e-6
    ence = 0.0
    total = 0
    for i in range(n_bins):
        mask = (mean_sigma >= bin_edges[i]) & (mean_sigma < bin_edges[i + 1])
        if mask.sum() < 2:
            continue
        rmse = np.sqrt(np.mean(angular_error[mask] ** 2))
        rmv = np.mean(mean_sigma[mask])
        ence += mask.sum() * abs(rmse - rmv) / rmv
        total += mask.sum()
    return ence / total if total > 0 else float("nan")


def compute_and_print_metrics(y_pred_full, y_true):
    direction = y_pred_full[:, :3]
    log_sigma2 = y_pred_full[:, 3:]
    sigma = np.sqrt(np.exp(log_sigma2))

    dot = np.clip(np.sum(y_true * direction, axis=1), -1.0, 1.0)
    angular_error = np.rad2deg(np.arccos(dot))

    mean_sigma = sigma.mean(axis=1)

    rho, p_value = spearmanr(mean_sigma, angular_error)
    ence = compute_ence(angular_error, mean_sigma)

    print("\n=== Uncertainty Evaluation Metrics ===")
    print(f"Total events:              {len(angular_error)}")
    print(f"Angular error median:      {np.median(angular_error):.3f} deg")
    print(f"Angular error 68%:         {np.percentile(angular_error, 68):.3f} deg")
    print(f"Angular error 90%:         {np.percentile(angular_error, 90):.3f} deg")
    print(f"Mean sigma range:          [{mean_sigma.min():.4f}, {mean_sigma.max():.4f}]")
    print(f"Mean sigma median:         {np.median(mean_sigma):.4f}")
    print(f"Spearman corr (σ, err):    {rho:.4f}  (p={p_value:.2e})")
    print(f"ENCE (20 bins):            {ence:.4f}")

    for frac in [0.5, 0.7, 0.9]:
        thr = np.percentile(mean_sigma, frac * 100)
        mask = mean_sigma <= thr
        print(
            f"  Cut {frac * 100:.0f}% events: "
            f"median error = {np.median(angular_error[mask]):.3f} deg, "
            f"68% error = {np.percentile(angular_error[mask], 68):.3f} deg"
        )


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    encoder_params = {**DEFAULT_ENCODER_PARAMS, "n_heads": args.n_heads}
    print(f"Loading model from {args.checkpoint}")
    model = build_model(
        args.checkpoint, encoder_params, args.predictor_hidden_size, args.num_predictor_layers
    )

    print(f"Loading data from {args.data}")
    preprocessor = AnglePreprocessorWithTres(args.tres_cut, DataPrefilter())
    dataloaders = create_dataloaders(
        path_to_data=args.data,
        DatasetType=BaikalDatasetAngles,
        batch_size=args.batch_size,
        is_graph=False,
        is_classification=False,
        preprocessor=preprocessor,
    )

    loader = dataloaders[args.split]
    y_pred_full, y_true = get_predictions(model, loader, max_batches=args.max_batches)

    direction = y_pred_full[:, :3]
    log_sigma2 = y_pred_full[:, 3:]

    compute_and_print_metrics(y_pred_full, y_true)

    sigma_xyz = np.sqrt(np.exp(log_sigma2))
    sigma_x, sigma_y, sigma_z = sigma_xyz[:, 0], sigma_xyz[:, 1], sigma_xyz[:, 2]

    x, y, z = direction[:, 0], direction[:, 1], direction[:, 2]
    theta_pred = np.arccos(np.clip(z, -1, 1))
    phi_pred = np.arctan2(y, x)

    x_t, y_t, z_t = y_true[:, 0], y_true[:, 1], y_true[:, 2]
    theta_true = np.arccos(np.clip(z_t, -1, 1))
    phi_true = np.arctan2(y_t, x_t)

    sin_theta = np.clip(np.sin(theta_pred), 1e-6, None)
    sigma_theta_rad = sigma_z / sin_theta
    r_xy_sq = np.clip(x**2 + y**2, 1e-12, None)
    sigma_phi_rad = np.sqrt(y**2 * sigma_x**2 + x**2 * sigma_y**2) / r_xy_sq

    sigma_theta_deg = np.rad2deg(sigma_theta_rad)
    sigma_phi_deg = np.rad2deg(sigma_phi_rad)

    theta_error = np.rad2deg(theta_true - theta_pred)
    phi_error = np.rad2deg(phi_true - phi_pred)
    phi_error = (phi_error + 180) % 360 - 180

    plotter_data = {
        "y_pred": direction,
        "y_true": y_true,
        "log_sigma2": log_sigma2,
        "pred_sigma2": np.exp(log_sigma2),
        "pred_theta_sigma2": sigma_theta_deg**2,
        "pred_phi_sigma2": sigma_phi_deg**2,
        "true_theta_sigma2": theta_error**2,
        "true_phi_sigma2": phi_error**2,
        "true_theta": np.rad2deg(theta_true),
        "true_phi": np.rad2deg(phi_true),
        "pred_theta": np.rad2deg(theta_pred),
        "pred_phi": np.rad2deg(phi_pred),
    }

    print(f"\nGenerating plots in {output_dir}")
    plotter = AngleUncertaintyPlotter(save_dir=output_dir)
    plotter.plot(plotter_data)

    print("Done!")
    print(f"  - {output_dir / 'uncertainty_cut_curve.png'}")
    print(f"  - {output_dir / 'error_vs_uncertainty.png'}")
    print(f"  - {output_dir / 'angle_uncertainty_evaluation.png'}")
    print(f"  - {output_dir / 'angle_uncertainty_metrics.png'}")


if __name__ == "__main__":
    main()
