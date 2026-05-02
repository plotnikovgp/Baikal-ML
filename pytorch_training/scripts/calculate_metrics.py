import argparse

import numpy as np
import torch
import yaml
from tqdm import tqdm

from data_utils import *
from metrics import AngleReconstructionMetrics
from models import load_model

DEVICE = "cuda"


def fix_seed():
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="/home/plotnikovgp/baikal/Baikal-ML/pytorch_training/train_configs/encoder_angle.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        "-cp",
        type=str,
        default="/home/plotnikovgp/baikal/Baikal-ML/pytorch_training/models/model_old.pth",
    )
    args = parser.parse_args()
    return args


def get_predictions(model, dataloader, device="cuda"):
    model.eval()
    y_preds = []
    y_trues = []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            if isinstance(batch, list):
                batch = batch[0]
            x, y_true, mask = batch[0], batch[1], batch[2]
            x = x.to(device)
            mask = mask.to(device)

            output = model(x, mask)
            output = output / output.norm(dim=1, keepdim=True)

            y_preds.append(output.cpu())
            y_trues.append(y_true)

    return torch.cat(y_preds), torch.cat(y_trues)


def main():
    fix_seed()
    args = parse_args()
    with open(args.config, "r") as f:
        train_params = yaml.safe_load(f)

    train_type = train_params.get("train_type")
    is_graph = train_params.get("is_graph")

    model = load_model(train_params["model_type"], train_params["model_params"]).to(DEVICE)
    model.load_state_dict(torch.load(args.checkpoint))
    model.eval()

    if train_type == "angle_reconstruction":
        DatasetType = BaikalDatasetAngles
        preprocessor = (
            AngleGraphPreprocessor(train_params["knn_neighbours"])
            if is_graph
            else AnglePreprocessor()
        )
        metrics_calc = AngleReconstructionMetrics()
    else:
        raise ValueError(f"Unsupported train_type: {train_type}")

    dataloaders = create_dataloaders(
        path_to_data=train_params["path_to_data"],
        is_graph=train_params["is_graph"],
        batch_size=train_params["batch_size"],
        DatasetType=DatasetType,
        is_classification=False,
        preprocessor=preprocessor,
        set_tres_stats=False,
    )

    y_pred, y_true = get_predictions(model, dataloaders["val"], DEVICE)
    metrics = metrics_calc(y_pred.numpy(), y_true.numpy())
    print("Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
