"""Configuration and input validation shared by the OaPA model."""

from dataclasses import dataclass

import torch


MODALITIES = ("T", "V", "A")


@dataclass
class BackboneConfig:
    """Configuration for text-video-audio or text-image OaPA backbones."""

    input_dims: tuple = (384, 512, 768)
    sequence_lengths: tuple = (32, 16, 1)
    hidden: int = 128
    heads: int = 4
    layers: int = 2
    memory_layers: int = 3
    encoder_layers: int = 1
    dropout: float = 0.1
    prompt_length: int = 8
    topics: int = 8
    conditional: bool = True
    use_completion: bool = True

    def __post_init__(self):
        if len(self.input_dims) not in (2, 3) or len(self.input_dims) != len(self.sequence_lengths):
            raise ValueError("input_dims and sequence_lengths must have length two or three")
        if min(self.input_dims) < 1 or min(self.sequence_lengths) < 1:
            raise ValueError("Feature dimensions and sequence lengths must be positive")
        if self.hidden < 4 or self.hidden % self.heads:
            raise ValueError("hidden must be divisible by heads")
        if min(self.layers, self.memory_layers, self.encoder_layers, self.prompt_length) < 1:
            raise ValueError("Model depths and prompt_length must be positive")
        if not 0 <= self.dropout < 1 or self.topics < 2:
            raise ValueError("Invalid dropout or topic count")


def validate_mask(mask, modalities, batch=None):
    if mask.dtype != torch.bool or mask.ndim != 2 or mask.shape[1] != modalities:
        raise ValueError(f"Mask must be a bool tensor [batch, {modalities}]")
    if batch is not None and mask.shape[0] != batch:
        raise ValueError("Mask batch size mismatch")
    if not mask.any(1).all():
        raise ValueError("At least one modality must be visible")


def sanitize(features, visible_mask, dims):
    """Zero hidden features before any learned operation."""

    validate_mask(visible_mask, len(dims), len(visible_mask))
    if len(features) != len(dims):
        raise ValueError("Feature/modality count mismatch")
    clean = []
    for k, (x, dim) in enumerate(zip(features, dims)):
        if x.ndim not in (2, 3) or x.shape[0] != len(visible_mask) or x.shape[-1] != dim:
            raise ValueError(f"Modality {k} must have shape [batch, length, {dim}]")
        if not x.is_floating_point():
            raise ValueError("Features must be floating point")
        x = torch.where(visible_mask[:, k].reshape(-1, *([1] * (x.ndim - 1))), x, torch.zeros_like(x))
        if not torch.isfinite(x).all():
            raise ValueError("Visible features contain NaN or Inf")
        clean.append(x)
    return tuple(clean)


def mask_codes(mask):
    weights = 2 ** torch.arange(mask.shape[1] - 1, -1, -1, device=mask.device)
    return (mask.long() * weights).sum(1)


def pooled(features):
    return tuple(x.mean(1) if x.ndim == 3 else x for x in features)
