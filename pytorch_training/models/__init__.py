from .encoder import *
from .graphnet import GraphnetDynedge, GraphnetAndEncoderStack
from .lstm import LSTM
from .gat import GAT
from .gincn import GINCN
from .uncertainty_predictor import UncertaintyPredictor, uncertainty_loss
from .unet import UNetModel
from .cnn import CNNModel, CNNModelWithAttention, CNNDomainAdaptation
import torch.nn as nn
import typing as tp


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
