"""OaPA: Observation-anchored Prediction Aggregation."""

from .aggregation import OaPA, VPA_FEATURES, VisibilityPredictionAggregator
from .backbone import BackboneConfig, OaPABackbone, build_oapa_backbone
from .pipeline import OaPAVPA
from .objectives import (
    DiscriminativeContrastiveRegularization,
    ModalityPrioritizedDistillation,
)

__all__ = [
    "BackboneConfig",
    "DiscriminativeContrastiveRegularization",
    "OaPAVPA",
    "ModalityPrioritizedDistillation",
    "OaPA",
    "OaPABackbone",
    "VisibilityPredictionAggregator",
    "VPA_FEATURES",
    "build_oapa_backbone",
]
