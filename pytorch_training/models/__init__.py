from .encoder import *
from .graphnet import GraphnetDynedge, GraphnetAndEncoderStack
from .lstm import LSTM
from .gat import GAT
from .gincn import GINCN
from .uncertainty_predictor import UncertaintyPredictor, uncertainty_loss
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
    else:
        raise NotImplementedError
    # elif model
