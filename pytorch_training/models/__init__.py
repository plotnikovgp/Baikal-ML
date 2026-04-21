import logging
import typing as tp

import torch
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
        encoder_params = model_kwargs.pop("encoder_params", {})
        encoder_checkpoint = model_kwargs.pop("encoder_checkpoint", None)
        encoder = Encoder(**encoder_params)
        if encoder_checkpoint:
            state_dict = torch.load(encoder_checkpoint, map_location="cpu", weights_only=False)
            encoder.load_state_dict(state_dict)
            logging.info(f"Loaded encoder weights from {encoder_checkpoint}")
        return UncertaintyPredictor(model=encoder, **model_kwargs)
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
