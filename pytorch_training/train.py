import logging
from pathlib import Path

import h5py
import hydra
import numpy as np
import pytorch_warmup as warmup
import torch
from clearml import Task
from omegaconf import DictConfig, OmegaConf

from data_utils import BaikalDatasetNoLabels, create_dataloaders, create_multi_dataset_dataloader
from data_utils.preprocessors import NoLabelsPerHitPreprocessor, NoLabelsPreprocessor
from models import load_model
from train_types import get_train_type
from training.trainer import Trainer

DEVICE = "cuda"

torch.autograd.set_detect_anomaly(True)
torch.set_num_threads(4)


def fix_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def load_state_dict_partial(model: torch.nn.Module, state_dict: dict, strict: bool = False) -> None:
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


def create_dataloaders_from_config(cfg: DictConfig, train_type_handler):
    is_graph = cfg.train_type.is_graph
    data_cfg = cfg.data
    training_cfg = cfg.training

    if data_cfg.get("datasets"):
        dataset_names = []
        DatasetType = train_type_handler.get_dataset_type()
        dataset_configs = []

        for ds_cfg in data_cfg.datasets:
            ds_config = OmegaConf.to_container(ds_cfg, resolve=True)
            dataset_names.append(ds_config.get("name", f"dataset_{len(dataset_names)}"))

            if ds_config.get("DatasetType") in ["no_labels", "no_labels_per_hit"]:
                ds_config["DatasetType"] = BaikalDatasetNoLabels
            else:
                ds_config["DatasetType"] = DatasetType

            if ds_config.get("preprocessor") == "no_labels":
                ds_config["preprocessor"] = NoLabelsPreprocessor()
            elif ds_config.get("preprocessor") == "no_labels_per_hit":
                ds_config["preprocessor"] = NoLabelsPerHitPreprocessor()
            else:
                preproc_cfg = OmegaConf.to_container(cfg, resolve=True)
                preproc_cfg.update(ds_config)
                ds_config["preprocessor"] = train_type_handler.get_preprocessor(preproc_cfg)

            renorm_to = ds_config.pop("renorm_to", None)
            if renorm_to:
                with h5py.File(ds_config["path_to_data"], "r") as f:
                    src_mean = torch.tensor(f["norm_param/mean"][:].astype(np.float32))
                    src_std = torch.tensor(f["norm_param/std"][:].astype(np.float32))
                with h5py.File(renorm_to, "r") as f:
                    dst_mean = torch.tensor(f["norm_param/mean"][:].astype(np.float32))
                    dst_std = torch.tensor(f["norm_param/std"][:].astype(np.float32))
                ds_config["renorm_params"] = (src_mean, src_std, dst_mean, dst_std)
                logging.info(f"Renormalization enabled: {ds_config['path_to_data']} -> {renorm_to}")

            dataset_configs.append(ds_config)

        dataloaders = create_multi_dataset_dataloader(
            dataset_configs=dataset_configs,
            probabilities=data_cfg.get("weights"),
            batch_size=training_cfg.batch_size,
            num_workers=data_cfg.get("num_workers", 1),
            prefetch_factor=data_cfg.get("prefetch_factor", 2),
            persistent_workers=data_cfg.get("persistent_workers", True),
            pin_memory=data_cfg.get("pin_memory", True),
            cache_datasets=data_cfg.get("cache_datasets", False),
            return_datasets=True,
            events_amount=data_cfg.get("events_amount"),
            set_tres_stats=data_cfg.get("set_tres_stats", False),
        )
    else:
        dataset_names = None
        preproc_cfg = OmegaConf.to_container(cfg, resolve=True)
        default_preprocessor = train_type_handler.get_preprocessor(preproc_cfg)
        DatasetType = train_type_handler.get_dataset_type()

        dataloaders = create_dataloaders(
            DatasetType=DatasetType,
            path_to_data=data_cfg.path,
            is_graph=is_graph,
            batch_size=training_cfg.batch_size,
            val_subset_cut=data_cfg.val_subset_cut,
            is_classification=train_type_handler.is_classification,
            preprocessor=default_preprocessor,
            num_workers=data_cfg.get("num_workers", 1),
            prefetch_factor=data_cfg.get("prefetch_factor", 2),
            persistent_workers=data_cfg.get("persistent_workers", True),
            pin_memory=data_cfg.get("pin_memory", True),
            cache_datasets=data_cfg.get("cache_datasets", False),
            events_amount=data_cfg.get("events_amount"),
            set_tres_stats=data_cfg.get("set_tres_stats", False),
        )

    return dataloaders, dataset_names


def setup_logging(cfg: DictConfig, is_val_mode: bool) -> Task | None:
    if is_val_mode:
        return None

    use_clearml = cfg.get("use_clearml", True)

    if use_clearml:
        try:
            task = Task.init(
                project_name=cfg.exp_project,
                task_name=cfg.exp_name,
                auto_connect_frameworks={"pytorch": True, "matplotlib": True},
            )
            task.connect(OmegaConf.to_container(cfg, resolve=True))
            print(f"ClearML task initialized: {task.id}")
            return task
        except Exception as e:
            print(f"ClearML initialization failed: {e}")
            print("Training will continue without experiment tracking.")
            return None
    return None


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    fix_seed(cfg.random_seed)

    is_val_mode = cfg.val_mode
    train_type_name = cfg.train_type.name

    save_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

    train_params = OmegaConf.to_container(cfg, resolve=True)
    train_params["train_type_name"] = train_type_name
    train_params["is_graph"] = cfg.train_type.is_graph
    train_params["val_mode"] = is_val_mode

    train_type_handler = get_train_type(train_type_name, train_params, DEVICE, save_dir)

    model = load_model(cfg.model.type, OmegaConf.to_container(cfg.model.params, resolve=True)).to(
        DEVICE
    )

    if cfg.from_checkpoint:
        state_dict = torch.load(cfg.from_checkpoint, weights_only=False)
        load_state_dict_partial(model, state_dict, strict=True)

    dataloaders, dataset_names = create_dataloaders_from_config(cfg, train_type_handler)

    if dataset_names:
        train_type_handler.set_dataset_names(dataset_names)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"n_params M: {n_params / 1e6:.2f}")
    print(model)

    training_cfg = cfg.training
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_cfg.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=training_cfg.get("scheduler_patience", 1000),
        factor=0.5,
        min_lr=1e-4,
    )
    warmup_scheduler = warmup.ExponentialWarmup(
        optimizer, warmup_period=training_cfg.get("warmup_steps", 0)
    )

    clearml_task = setup_logging(cfg, is_val_mode)

    trainer = Trainer(
        model=model,
        train_type=train_type_handler,
        optimizer=optimizer,
        warmup_scheduler=warmup_scheduler,
        warmup_steps=training_cfg.get("warmup_steps", 0),
        scheduler=scheduler,
        accumulate_grad_steps=training_cfg.get("accumulate_grad_steps", 1),
        grad_clip_value=training_cfg.get("grad_clip_value"),
        clearml_task=clearml_task,
        model_save_dir=str(save_dir),
        valid_main_metric=training_cfg.get("valid_main_metric", "loss"),
    )

    if "train_datasets" in dataloaders:
        train_dataset_size = sum(len(ds) for ds in dataloaders["train_datasets"])
    elif "train_dataset" in dataloaders:
        train_dataset_size = len(dataloaders["train_dataset"])
    else:
        train_dataset_size = None

    trainer.train(
        train_loader=dataloaders["train"],
        val_loader=dataloaders["val"],
        train_params_str=OmegaConf.to_yaml(cfg),
        num_iters=training_cfg.get("num_train_steps_per_validation", 256),
        dataset_names=dataset_names,
        save_best_per_dataset=training_cfg.get("save_best_per_dataset", False),
        val_mode=is_val_mode,
        min_recall=training_cfg.get("min_recall"),
        train_dataset_size=train_dataset_size,
    )


if __name__ == "__main__":
    main()
