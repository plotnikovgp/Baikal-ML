import argparse
import typing as tp
from pathlib import Path

import numpy as np
import torch
import yaml
import wandb
import pytorch_warmup as warmup

from data_utils import *
from metrics import *
from models import uncertainty_loss, load_model
from training import train, train_iters, validate

DEVICE = "cuda"
SEED = 42

torch.autograd.set_detect_anomaly(True)

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
    required_keys = ["path_to_data", "model_type", "batch_size", "lr", "model_params", "is_graph"]
    for key in required_keys:
        if key not in parsed_config:
            raise ValueError(f"Required key={key} wasn't provided in config")


def create_preprocessor(train_type, is_graph, config):
    """
    Create a preprocessor based on the training type and configuration.
    This allows for creating custom preprocessors for individual datasets.
    
    Args:
        train_type: The type of training being performed
        is_graph: Whether the model is using graph-based data
        config: Configuration dictionary with parameters for the preprocessor
        
    Returns:
        A preprocessor instance configured with the given parameters
    """
    if train_type == "noise_sig":
        return (
            NoiseSigGraphPreprocessor(config["knn_neighbours"])
            if is_graph
            else NoiseSigPreprocessor()
        )
    elif train_type == "track_cascade":
        return (
            TrackCascadeGraphPreprocessor(
                config["knn_neighbours"], config["tres_cut"]
            )
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
    elif train_type == "angle_reconstruction" or train_type == "angle_reconstruction_old" or train_type == "angle_reconstruction_domain_adaptation":
        if not is_graph:
            data_prefilter = DataPrefilter(**(config.get("data_prefilter_params", {})))
            return AnglePreprocessorWithTres(config["tres_cut"], data_prefilter)
        else:
            return AngleGraphPreprocessor(config["knn_neighbours"])
    elif train_type == "direction":
        return DirectionPreprocessor()
    elif train_type == "angle_reconstruction_sigma_tune":
        return AnglePreprocessor()
    elif train_type == "angle_and_track_cascade":
        return AngleAndTrackCascadePreprocessor(config["tres_cut"])
    else:
        raise ValueError(f"Unknown train_type: {train_type}")


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
    model = load_model(train_params["model_type"], train_params["model_params"]).to(DEVICE) if train_type != "angle_reconstruction_sigma_tune" else None

    if train_params.get("from_checkpoint"):
        model.load_state_dict(torch.load(train_params["from_checkpoint"]))
    model.eval()
    track_cascade_model = None
    # with open("train_configs/encoder_track_cascade.yaml", "r") as f:
    #     track_cascade_params = yaml.safe_load(f)
    # track_cascade_model = load_model(track_cascade_params["model_type"], track_cascade_params["model_params"])
    # track_cascade_model.load_state_dict(
    #     torch.load("/home/plotnikovgp/baikal/Baikal-ML/pytorch_training/checkpoints/track_cascade_enc_hyps_mc_0924/encoder_nl5_dff512_hs1024_nh2_bs128_tres=10.0_new_data/best.ckpt")
    # )
    # track_cascade_model.to(DEVICE)
    # track_cascade_model.eval()

    if train_type == "noise_sig":
        DatasetType = BaikalDataset
        is_classification = True
        criterion = torch.nn.CrossEntropyLoss()
        metrics_calc_fun = binary_clf_metrics
    elif train_type == "track_cascade":
        DatasetType = BaikalDatasetTrackCascade
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
    elif train_type == "angle_reconstruction" or train_type == "angle_reconstruction_old":
        if not is_graph:
            DatasetType = BaikalDatasetAngles if "old" not in train_type else BaikalDatasetAnglesOld
        else:
            DatasetType = BaikalDatasetAnglesSingle if "old" not in train_type else BaikalDatasetAnglesOldSingle
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
            distance_loss = torch.abs(torch.sum(w * n, dim=1, keepdim=True)) / norm_n_clamped
            distance_loss = distance_loss.mean()
            return angle_loss + distance_loss * dist_loss_coef
    elif train_type == "angle_reconstruction_sigma_tune":
        DatasetType = BaikalDatasetAngles
        encoder_model = load_model("encoder", train_params["encoder_model_params"])
        encoder_model.load_state_dict(torch.load(train_params["encoder_model_path"]))
        encoder_model = encoder_model.to(DEVICE)
        model = load_model(train_params["model_type"], {"model": encoder_model}).to(DEVICE)
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
        
        def criterion(y_pred, y_true, domain_pred=None, domain_true=None):
            # Main task loss (angle reconstruction) - only applied to labeled samples
            if label_dataset_name is not None:
                mask = domain_true == dataset_names.index(label_dataset_name)
                if mask.sum() > 0:
                    angle_loss = torch.abs(y_pred[mask] - y_true[mask]).mean()
                else:
                    angle_loss = torch.tensor(0.0, device=y_pred.device)
            else:
                angle_loss = torch.abs(y_pred - y_true).mean()
            
            # Domain adaptation loss
            if domain_pred is not None and domain_true is not None:
                domain_loss = torch.nn.functional.cross_entropy(domain_pred, domain_true)
                return angle_loss + domain_adaptation_loss_k * domain_loss
            else:
                return angle_loss
    else:
        raise ValueError("unknown train_type")
    
    # Create a default preprocessor for single dataset mode
    default_preprocessor = create_preprocessor(train_type, is_graph, train_params)
    
    # Dataset names for validation
    dataset_names = None

    # Check if using multi-dataset setup
    if "dataset_configs" in train_params:
        # Get dataset names if provided
        dataset_names = []
        
        # Configure each dataset
        for i, config in enumerate(train_params["dataset_configs"]):
            # Apply common configuration to each dataset unless it's explicitly overridden
            for key, value in train_params.items():
                if key not in ["dataset_configs", "dataset_weights", "dataset_names"] and key not in config:
                    config[key] = value
            
            # Get dataset name (use provided name or default to "dataset_{i}")
            dataset_name = config.get("name", f"dataset_{i}")
            dataset_names.append(dataset_name)
            
            # Set DatasetType based on train_type if not specified
            if "DatasetType" not in config:
                config["DatasetType"] = DatasetType
            
            # Create individual preprocessor for each dataset with its own parameters
            if "preprocessor" not in config:
                # Create a preprocessor using this dataset's specific configuration
                config["preprocessor"] = create_preprocessor(
                    train_type, 
                    config.get("is_graph", is_graph),
                    config
                )
                
        # Create multi-dataset dataloaders
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
        # Original single dataset setup
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
    warmup_scheduler = warmup.ExponentialWarmup(optimizer, train_params.get("warmup_steps", 0))
    
    # Use train_dataset or first train_datasets from multi-dataset setup
    train_dataset = dataloaders.get("train_dataset") or dataloaders.get("train_datasets", [None])[0]
    
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
        is_angle_reconstruction=(train_type == "angle_reconstruction"),
        is_angle_and_track_cascade=(train_type == "angle_and_track_cascade"),
        is_angle_reconstruction_sigma_tune=(train_type == "angle_reconstruction_sigma_tune"),
        is_direction=(train_type == "direction"),
        is_domain_adaptation=(train_type == "angle_reconstruction_domain_adaptation"),
        grad_clip_value=train_params.get("grad_clip_value", None),
        accumulate_grad_steps=train_params.get("accumulate_grad_steps", 1),
    )

    validate_fun_kwargs = dict(
        val_loader=dataloaders["val"],
        dataset=dataloaders.get("val_dataset") or dataloaders.get("val_datasets", [None])[0],
        criterion=criterion,
        is_graph=is_graph,
        metrics_calc_fun=metrics_calc_fun,
        track_cascade_model=track_cascade_model,
        is_classification=is_classification,
        is_track_cascade_tres_train=(train_type == "tres_and_track_cascade"),
        is_angle_reconstruction=(train_type == "angle_reconstruction"),
        is_angle_reconstruction_sigma_tune=(train_type == "angle_reconstruction_sigma_tune"),
        is_angle_and_track_cascade=(train_type == "angle_and_track_cascade"),
        is_direction=(train_type == "direction"),
        is_domain_adaptation=(train_type == "angle_reconstruction_domain_adaptation"),
        dataset_names=dataset_names,  # Use custom dataset names
    )

    if not args.disable_wandb:
        wandb.init(
            project=train_params["exp_project"],
            name=train_params["exp_name"],
            config=train_params,
        )


    save_dir = Path("checkpoints") / train_params["exp_project"] / train_params["exp_name"] 
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
