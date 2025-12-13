import typing as tp

import torch.nn as nn

from .cnn import CNNDomainAdaptation, CNNModel, CNNModelWithAttention
from .encoder import *
from .gat import GAT
from .gcn import GCN
from .gincn import GINCN
from .graphnet import GraphnetAndEncoderStack, GraphnetDynedge
from .lstm import LSTM
from .uncertainty_predictor import UncertaintyPredictor
from .unet import UNetModel


def load_model(model_type: str, model_kwargs: dict[str, tp.Any]) -> nn.Module:
    if model_type == "encoder":
        return Encoder(**model_kwargs)
    elif model_type == "encoder_domain_adaptation":
        return EncoderDomainAdaptation(**model_kwargs)
    elif model_type == "graphnet":
        return GraphnetDynedge(**model_kwargs)
    elif model_type == "graphnet_and_encoder_stack":
        return GraphnetAndEncoderStack(**model_kwargs)
    elif model_type == "lstm":
        return LSTM(**model_kwargs)
    elif model_type == "gat":
        return GAT(**model_kwargs)
    elif model_type == "gcn":
        return GCN(**model_kwargs)
    elif model_type == "gin":
        return GINCN(**model_kwargs)
    elif model_type == "uncertainty_predictor":
        return UncertaintyPredictor(**model_kwargs)
    elif model_type == "unet":
        return UNetModel(**model_kwargs)
    elif model_type == "cnn":
        return CNNModel(**model_kwargs)
    elif model_type == "cnn_attention":
        return CNNModelWithAttention(**model_kwargs)
    elif model_type == "cnn_domain_adaptation":
        return CNNDomainAdaptation(**model_kwargs)
    else:
        raise NotImplementedError
    # elif model
