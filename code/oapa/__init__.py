"""OaPA: Observation-anchored Prediction Aggregation."""

from .aggregation import OaPA, VisibilityPredictionAggregator
from .backbone import BackboneConfig, OaPABackbone, build_oapa_backbone
from .objectives import (
    DiscriminativeContrastiveRegularization,
    ModalityPrioritizedDistillation,
)

__all__ = [
    "BackboneConfig",
    "DiscriminativeContrastiveRegularization",
    "ModalityPrioritizedDistillation",
    "OaPA",
    "OaPABackbone",
    "VisibilityPredictionAggregator",
    "build_oapa_backbone",
]
