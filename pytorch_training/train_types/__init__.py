from pathlib import Path
from typing import Any, Dict, Type

from .angle import (
    AngleAndTrackCascadeTrainType,
    AngleReconstructionDomainAdaptationTrainType,
    AngleReconstructionTrainType,
)
from .base import BaseTrainType
from .direction import DirectionTrainType
from .energy import EnergyReconstructionDomainAdaptationTrainType, EnergyReconstructionTrainType
from .noise_sig import (
    NoiseSigDomainAdaptationTrainType,
    NoiseSigOriginalLabelsDomainAdaptationTrainType,
    NoiseSigOriginalLabelsTrainType,
    NoiseSigOrLabelsTrainType,
    NoiseSigTrainType,
    TresRegressionSoftLossTrainType,
    TresRegressionTrainType,
)
from .track_cascade import TrackCascadeDomainAdaptationTrainType, TrackCascadeTrainType
from .tres import TresAndTrackCascadeTrainType, TresDomainAdaptationTrainType, TresTrainType

TRAIN_TYPE_REGISTRY: Dict[str, Type[BaseTrainType]] = {
    "noise_sig": NoiseSigTrainType,
    "noise_sig_domain_adaptation": NoiseSigDomainAdaptationTrainType,
    "noise_sig_original_labels": NoiseSigOriginalLabelsTrainType,
    "noise_sig_original_labels_da": NoiseSigOriginalLabelsDomainAdaptationTrainType,
    "noise_sig_or_labels": NoiseSigOrLabelsTrainType,
    "tres_regression": TresRegressionTrainType,
    "tres_regression_soft": TresRegressionSoftLossTrainType,
    "track_cascade": TrackCascadeTrainType,
    "track_cascade_domain_adaptation": TrackCascadeDomainAdaptationTrainType,
    "tres": TresTrainType,
    "tres_domain_adaptation": TresDomainAdaptationTrainType,
    "tres_and_track_cascade": TresAndTrackCascadeTrainType,
    "angle_reconstruction": AngleReconstructionTrainType,
    "angle_reconstruction_old": AngleReconstructionTrainType,
    "angle_reconstruction_domain_adaptation": AngleReconstructionDomainAdaptationTrainType,
    "angle_and_track_cascade": AngleAndTrackCascadeTrainType,
    "direction": DirectionTrainType,
    "energy_reconstruction": EnergyReconstructionTrainType,
    "energy_reconstruction_domain_adaptation": EnergyReconstructionDomainAdaptationTrainType,
}


def get_train_type(
    train_type_name: str, train_params: Dict[str, Any], device: str = "cuda", save_dir: Path = None
) -> BaseTrainType:
    if train_type_name not in TRAIN_TYPE_REGISTRY:
        raise ValueError(
            f"Unknown train_type: {train_type_name}. Available: {list(TRAIN_TYPE_REGISTRY.keys())}"
        )

    train_type_class = TRAIN_TYPE_REGISTRY[train_type_name]

    # Some train types need save_dir
    if train_type_name in [
        "angle_reconstruction",
        "angle_reconstruction_old",
        "angle_reconstruction_domain_adaptation",
        "energy_reconstruction",
        "energy_reconstruction_domain_adaptation",
        "direction",
    ]:
        return train_type_class(train_params, device, save_dir)

    return train_type_class(train_params, device)


__all__ = [
    "BaseTrainType",
    "get_train_type",
    "TRAIN_TYPE_REGISTRY",
]
