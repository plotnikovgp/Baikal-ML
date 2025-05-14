import argparse
import typing as tp
from pathlib import Path

import numpy as np
import torch
import yaml
import wandb
import pytorch_warmup as warmup
import logging

from data_utils import *
from metrics import *
from models import uncertainty_loss, load_model
from training import train, train_iters, validate

DEVICE = "cuda"
SEED = 42

torch.autograd.set_detect_anomaly(True)
torch.set_num_threads(4)

def fix_seed(seed: int = SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-dw", "--disable-wandb", action="store_true")
    parser.add_argument("-c", "--config", help="path to config file", required=True)
    return parser.parse_args()


def validate_config(parsed_config: tp.Dict[str, tp.Any]):
    required_keys = [
        "path_to_data",
        "model_type",
        "batch_size",
        "lr",
        "model_params",
        "is_graph",
    ]
    for key in required_keys:
        if key not in parsed_config:
            raise ValueError(f"Required key={key} wasn't provided in config")


def create_preprocessor(train_type, is_graph, config):
    if train_type == "noise_sig":
        return (
            NoiseSigGraphPreprocessor(config["knn_neighbours"])
            if is_graph
            else NoiseSigPreprocessor()
        )
    elif train_type == "track_cascade":
        return (
            TrackCascadeGraphPreprocessor(config["knn_neighbours"], config["tres_cut"])
            if is_graph
            else TrackCascadePreprocessor(config["tres_cut"])
        )
    elif train_type == "tres":
        return (
            TresGraphPreprocessor(config["knn_neighbours"])
            if is_graph
            else TresPreprocessor()
        )
    elif train_type == "tres_and_track_cascade":
        return (
            TresAndTrackCascadeGraphPreprocessor(
                config["knn_neighbours"], config["tres_cut"]
            )
            if is_graph
            else TresAndTrackCascadePreprocessor(config["tres_cut"])
        )
    elif train_type == "energy_reconstruction":
        data_prefilter = DataPrefilter(**(config.get("data_prefilter_params", {})))
        return EnergyPreprocessor(data_prefilter)
    elif train_type == "energy_reconstruction_domain_adaptation":
        data_prefilter = DataPrefilter(**(config.get("data_prefilter_params", {})))
        return EnergyPreprocessor(data_prefilter)
    elif train_type in [
        "angle_reconstruction",
        "angle_reconstruction_old",
        "angle_reconstruction_domain_adaptation",
        "angle_reconstruction_sigma_tune",
    ]:
        if not is_graph:
            data_prefilter = DataPrefilter(**(config.get("data_prefilter_params", {})))
            return AnglePreprocessorWithTres(config["tres_cut"], data_prefilter)
        else:
            return AngleGraphPreprocessor(config["knn_neighbours"])
    elif train_type == "direction":
        return DirectionPreprocessor()
    elif train_type == "angle_and_track_cascade":
        return AngleAndTrackCascadePreprocessor(config["tres_cut"])
    elif train_type == "track_cascade_domain_adaptation":
        # Select your preprocessor for track_cascade with domain adaptation
        return (
            TrackCascadeGraphPreprocessor(
                config["knn_neighbours"], config.get("tres_cut", None)
            )
            if is_graph
            else TrackCascadePreprocessor(config.get("tres_cut", None))
        )
    else:
        raise ValueError(f"Unknown train_type: {train_type}")


def load_state_dict_partial(
    model: torch.nn.Module, state_dict: dict, strict: bool = False
) -> None:
    """
    Load state dict partially, ignoring non-matching keys.

    Args:
        model: The model to load weights into
        state_dict: The state dict to load
        strict: If True, requires exact matching of keys. If False, ignores non-matching keys.
    """
    model_state_dict = model.state_dict()
    missing_keys = []
    unexpected_keys = []

    for key in state_dict:
        if key not in model_state_dict:
            unexpected_keys.append(key)
            continue
        if state_dict[key].shape != model_state_dict[key].shape:
            logging.warning(
                f"Shape mismatch for key {key}: expected {model_state_dict[key].shape}, got {state_dict[key].shape}"
            )
            continue
        model_state_dict[key] = state_dict[key]

    for key in model_state_dict:
        if key not in state_dict:
            missing_keys.append(key)

    if missing_keys:
        logging.warning(f"Missing keys in state_dict: {missing_keys}")
    if unexpected_keys:
        logging.warning(f"Unexpected keys in state_dict: {unexpected_keys}")

    model.load_state_dict(model_state_dict, strict=strict)


def main():
    fix_seed()
    args = parse_args()

    config_path = args.config if "/" in args.config else "train_configs/" + args.config

    with open(config_path, "r") as f:
        train_params = yaml.safe_load(f)

    train_params_str = yaml.dump(train_params)

    train_type = train_params.get("train_type")
    is_graph = train_params.get("is_graph")
    is_classification = False
    model = (
        load_model(train_params["model_type"], train_params["model_params"]).to(DEVICE)
        if train_type != "angle_reconstruction_sigma_tune"
        else None
    )

    if train_params.get("from_checkpoint"):
        state_dict = torch.load(train_params["from_checkpoint"])
        load_state_dict_partial(model, state_dict, strict=False)

    track_cascade_model = None
    # with open("train_configs/encoder_track_cascade.yaml", "r") as f:
    #     track_cascade_params = yaml.safe_load(f)
    # track_cascade_model = load_model(track_cascade_params["model_type"], track_cascade_params["model_params"])
    # track_cascade_state_dict = torch.load("/home/plotnikovgp/baikal/Baikal-ML/pytorch_training/checkpoints/track_cascade_enc_hyps_mc_0924/encoder_nl5_dff512_hs512_try_on_other_data/best.ckpt")
    # load_state_dict_partial(track_cascade_model, track_cascade_state_dict, strict=True)
    # track_cascade_model.to(DEVICE)
    # track_cascade_model.eval()

    if train_type == "noise_sig":
        DatasetType = BaikalDataset
        is_classification = True
        criterion = torch.nn.CrossEntropyLoss()
        metrics_calc_fun = binary_clf_metrics
    elif train_type == "track_cascade":
        DatasetType = (
            BaikalDatasetTrackCascade
            if not is_graph
            else BaikalDatasetTrackCascadeSingle
        )
        metrics_calc_fun = track_cascade_clf_metrics
        is_classification = True
        criterion = torch.nn.CrossEntropyLoss()
    elif train_type == "tres":
        DatasetType = BaikalDatasetTres
        metrics_calc_fun = regression_metrics
        criterion = torch.nn.MSELoss()
    elif train_type == "tres_and_track_cascade":
        DatasetType = BaikalDatasetTrackCascade
        metrics_calc_fun = tres_and_track_cascade_metrics

        ce_ = torch.nn.CrossEntropyLoss()
        mse_ = torch.nn.MSELoss()

        def criterion(y_pred, y_true):
            ce_loss = ce_(y_pred[:, :-1], y_true[:, 0].long())
            mse_loss = mse_(y_pred[:, -1], y_true[:, -1])
            return ce_loss + train_params["tres_mse_coef"] * mse_loss

    elif (
        train_type == "angle_reconstruction" or train_type == "angle_reconstruction_old"
    ):
        if not is_graph:
            DatasetType = (
                BaikalDatasetAngles
                if "old" not in train_type
                else BaikalDatasetAnglesOld
            )
        else:
            DatasetType = (
                BaikalDatasetAnglesSingle
                if "old" not in train_type
                else BaikalDatasetAnglesOldSingle
            )
        train_type = "angle_reconstruction"
        metrics_calc_fun = angle_reconstruction_metrics

        def criterion(y_pred, y_true):
            return torch.abs(y_pred - y_true).mean()

    elif train_type == "direction":
        DatasetType = BaikalDatasetAngles
        metrics_calc_fun = direction_metrics
        dist_loss_coef = train_params["distance_loss_coef"]

        def criterion(y_pred, y_true):
            angle_pred = y_pred[:, :3]
            angle_true = y_true[:, :3]
            point_pred = y_pred[:, 3:]
            point_true = y_true[:, 3:]
            angle_loss = torch.nn.L1Loss()(angle_pred, angle_true)
            w = point_pred - point_true
            n = torch.cross(angle_pred, angle_true)
            norm_n = torch.norm(n, dim=1, keepdim=True)
            norm_n_clamped = torch.clamp(norm_n, min=1e-6)
            distance_loss = (
                torch.abs(torch.sum(w * n, dim=1, keepdim=True)) / norm_n_clamped
            )
            distance_loss = distance_loss.mean()
            return angle_loss + distance_loss * dist_loss_coef

    elif train_type == "angle_reconstruction_sigma_tune":
        DatasetType = BaikalDatasetAngles
        encoder_model = load_model("encoder", train_params["encoder_model_params"])
        encoder_model.load_state_dict(torch.load(train_params["encoder_model_path"]))
        encoder_model = encoder_model.to(DEVICE)
        model = load_model(train_params["model_type"], {"model": encoder_model}).to(
            DEVICE
        )
        metrics_calc_fun = angle_uncertainty_metrics
        criterion = uncertainty_loss
    elif train_type == "angle_and_track_cascade":
        DatasetType = BaikalDatasetAnglesAndTrackCascade
        ce_ = torch.nn.CrossEntropyLoss()
        mse_ = torch.nn.MSELoss()

        def criterion(output, y_true):
            ce_loss = ce_(output[:, :2], y_true[:, 0].long())
            mse_loss = mse_(output[:, 0, 2:].reshape(-1, 2), y_true[:, :2])
            return ce_loss + train_params["mse_coef"] * mse_loss

    elif train_type == "angle_reconstruction_domain_adaptation":
        if not is_graph:
            DatasetType = BaikalDatasetAngles
        else:
            DatasetType = BaikalDatasetAnglesSingle

        # Use the same metrics function as regular angle reconstruction
        metrics_calc_fun = angle_reconstruction_metrics

        # Custom criterion for domain adaptation that handles labeled/unlabeled datasets
        domain_adaptation_loss_k = train_params.get("domain_adaptation_loss_k", 0.1)
        label_dataset_name = train_params.get("label_dataset_name", None)

        loss_fn = torch.nn.L1Loss()
        def criterion(y_pred, y_true, domain_pred=None, domain_true=None):
            if domain_true is None:
                return {"loss": loss_fn(y_pred, y_true)}

            mask = domain_true == dataset_names.index(label_dataset_name)
            if mask.sum() > 0:
                angle_loss = loss_fn(y_pred[mask], y_true[mask])
            else:
                angle_loss = torch.tensor(0.0, device=y_pred.device)

            domain_loss = torch.nn.functional.cross_entropy(domain_pred, domain_true)
            total_loss = angle_loss + domain_adaptation_loss_k * domain_loss
            return {
                "loss": total_loss,  # For backward pass
                "angle_loss": angle_loss.detach(),
                "domain_loss": domain_loss.detach(),
            }

    elif train_type == "track_cascade_domain_adaptation":
        if not is_graph:
            DatasetType = BaikalDatasetTrackCascade
        else:
            DatasetType = BaikalDatasetTrackCascadeSingle

        metrics_calc_fun = track_cascade_clf_metrics

        # For domain adaptation
        domain_adaptation_loss_k = train_params.get("domain_adaptation_loss_k", 0.1)
        label_dataset_name = train_params.get("label_dataset_name", None)

        ce_ = torch.nn.CrossEntropyLoss()

        def criterion(y_pred, y_true, domain_pred=None, domain_true=None):
            """
            y_pred: main output (logits for track_cascade classification)
            y_true: main labels
            domain_pred: domain classifier output (logits)
            domain_true: domain labels (ints)
            """
            if domain_true is None:
                return {"loss": ce_(y_pred, y_true.long())}

            mask = domain_true == dataset_names.index(label_dataset_name)
            if mask.sum() > 0:
                track_cascade_loss = ce_(y_pred[mask], y_true[mask].long())
            else:
                track_cascade_loss = torch.tensor(0.0, device=y_pred.device)

            domain_loss = torch.nn.functional.cross_entropy(domain_pred, domain_true)
            total_loss = track_cascade_loss + domain_adaptation_loss_k * domain_loss
            return {
                "loss": total_loss,  # For backward pass
                "track_cascade_loss": track_cascade_loss.detach(),
                "domain_loss": domain_loss.detach(),
            }

    elif train_type == "energy_reconstruction":
        DatasetType = BaikalDatasetEnergy
        metrics_calc_fun = regression_metrics
        criterion = torch.nn.MSELoss()
    elif train_type == "energy_reconstruction_domain_adaptation":
        DatasetType = BaikalDatasetEnergy
        metrics_calc_fun = regression_metrics

        # For domain adaptation
        domain_adaptation_loss_k = train_params.get("domain_adaptation_loss_k", 0.1)
        label_dataset_name = train_params.get("label_dataset_name", None)

        # Move the criterion definition to after dataset_names is populated
        # We'll initialize it to None here and define it properly later
        def log_cosh_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
            def _log_cosh(x: torch.Tensor) -> torch.Tensor:
                return x + torch.nn.functional.softplus(-2.0 * x) - math.log(2.0)

            return torch.mean(_log_cosh(y_pred - y_true))

        if train_params.get("use_cosh_loss", False):
            energy_loss_ = log_cosh_loss
        else:
            energy_loss_ = torch.nn.MSELoss() # torch.nn.L1Loss() 

        def criterion(y_pred, y_true, domain_pred=None, domain_true=None):
            """
            y_pred: main output (energy prediction)
            y_true: main labels (energy values)
            domain_pred: domain classifier output (logits)
            domain_true: domain labels (ints)
            """
            assert label_dataset_name is not None
            y_pred = y_pred.squeeze()
            y_true = y_true.squeeze()

            if domain_true is None:
                return {"loss": energy_loss_(y_pred, y_true)}

            mask = domain_true == dataset_names.index(label_dataset_name)
            if mask.sum() > 0:
                energy_loss = energy_loss_(y_pred[mask], y_true[mask])
            else:
                energy_loss = torch.tensor(0.0, device=y_pred.device)

            domain_loss = torch.nn.functional.cross_entropy(domain_pred, domain_true)
            total_loss = energy_loss + domain_adaptation_loss_k * domain_loss
            return {
                "loss": total_loss,  # For backward pass
                "energy_loss": energy_loss.detach(),
                "domain_loss": domain_loss.detach(),
            }

    else:
        raise ValueError("unknown train_type")

    default_preprocessor = create_preprocessor(train_type, is_graph, train_params)

    dataset_names = None

    if "dataset_configs" in train_params:
        dataset_names = []

        for i, config in enumerate(train_params["dataset_configs"]):
            for key, value in train_params.items():
                if (
                    key not in ["dataset_configs", "dataset_weights", "dataset_names"]
                    and key not in config
                ):
                    config[key] = value

            dataset_name = config.get("name", f"dataset_{i}")
            dataset_names.append(dataset_name)

            if "DatasetType" not in config:
                config["DatasetType"] = DatasetType

            if "preprocessor" not in config:
                config["preprocessor"] = create_preprocessor(
                    train_type, config.get("is_graph", is_graph), config
                )

        dataloaders = create_multi_dataset_dataloader(
            dataset_configs=train_params["dataset_configs"],
            probabilities=train_params.get("dataset_weights", None),
            batch_size=train_params["batch_size"],
            num_workers=train_params.get("num_workers", 1),
            prefetch_factor=train_params.get("prefetch_factor", 2),
            persistent_workers=train_params.get("persistent_workers", True),
            pin_memory=train_params.get("pin_memory", True),
            cache_datasets=train_params.get("cache_datasets", False),
            return_datasets=True,
        )
    else:
        dataloaders = create_dataloaders(
            DatasetType=DatasetType,
            path_to_data=train_params["path_to_data"],
            is_graph=train_params["is_graph"],
            batch_size=train_params["batch_size"],
            val_subset_cut=train_params["val_subset_cut"],
            is_classification=is_classification,
            preprocessor=default_preprocessor,
            num_workers=train_params.get("num_workers", 1),
            prefetch_factor=train_params.get("prefetch_factor", 2),
            persistent_workers=train_params.get("persistent_workers", True),
            pin_memory=train_params.get("pin_memory", True),
            cache_datasets=train_params.get("cache_datasets", False),
            events_amount=train_params.get("events_amount", None),
        )

    n_params = sum(p.numel() for p in model.parameters())
    print("n_params M", n_params / 1e6)
    print(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=train_params["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", factor=0.5, min_lr=1e-3, patience=128
    )
    warmup_scheduler = warmup.ExponentialWarmup(
        optimizer, train_params.get("warmup_steps", 0)
    )

    train_dataset = (
        dataloaders.get("train_dataset") or dataloaders.get("train_datasets", [None])[0]
    )

    train_fun_kwargs = dict(
        optimizer=optimizer,
        dataset=train_dataset,
        train_loader=dataloaders["train"],
        criterion=criterion,
        warmup_scheduler=warmup_scheduler,
        warmup_steps=train_params.get("warmup_steps", 0),
        scheduler=scheduler,
        is_graph=is_graph,
        track_cascade_model=track_cascade_model,
        num_iters=train_params.get("num_train_steps_per_validation", 256),
        metrics_calc_fun=metrics_calc_fun,
        is_classification=is_classification,
        is_track_cascade_tres_train=(train_type == "tres_and_track_cascade"),
        is_angle_reconstruction=(train_type == "angle_reconstruction" or train_type == "angle_reconstruction_domain_adaptation"),
        is_angle_and_track_cascade=(train_type == "angle_and_track_cascade"),
        is_angle_reconstruction_sigma_tune=(
            train_type == "angle_reconstruction_sigma_tune"
        ),
        is_direction=(train_type == "direction"),
        is_domain_adaptation="domain_adaptation" in train_type,
        is_energy_reconstruction="energy_reconstruction" in train_type,
        grad_clip_value=train_params.get("grad_clip_value", None),
        accumulate_grad_steps=train_params.get("accumulate_grad_steps", 1),
        min_recall=train_params.get("min_recall", None),
    )
    validate_fun_kwargs = dict(
        val_loader=dataloaders["val"],
        dataset=dataloaders.get("val_dataset")
        or dataloaders.get("val_datasets", [None])[0],
        criterion=criterion,
        is_graph=is_graph,
        metrics_calc_fun=metrics_calc_fun,
        track_cascade_model=track_cascade_model,
        is_classification=is_classification,
        is_track_cascade_tres_train=(train_type == "tres_and_track_cascade"),
        is_angle_reconstruction=(train_type == "angle_reconstruction" or train_type == "angle_reconstruction_domain_adaptation"),
        is_angle_reconstruction_sigma_tune=(
            train_type == "angle_reconstruction_sigma_tune"
        ),
        is_angle_and_track_cascade=(train_type == "angle_and_track_cascade"),
        is_direction=(train_type == "direction"),
        is_domain_adaptation="domain_adaptation" in train_type,
        is_energy_reconstruction="energy_reconstruction" in train_type,
        dataset_names=dataset_names,  # Use custom dataset names
        min_recall=train_params.get("min_recall", None),
    )

    if not args.disable_wandb:
        wandb.init(
            project=train_params["exp_project"],
            name=train_params["exp_name"],
            config=train_params,
        )

    save_dir = (
        Path("checkpoints") / train_params["exp_project"] / train_params["exp_name"]
    )
    Path.mkdir(save_dir, parents=True, exist_ok=True)
    train(
        model,
        train_fun=train_iters,
        train_fun_kwargs=train_fun_kwargs,
        validate_fun=validate,
        validate_fun_kwargs=validate_fun_kwargs,
        use_wandb=not args.disable_wandb,
        model_save_dir=save_dir,
        train_params_str=train_params_str,
        save_best_per_dataset=train_params.get("save_best_per_dataset", False),
        dataset_names=dataset_names,  # Use custom dataset names
    )


if __name__ == "__main__":
    main()
