"""Visibility-constrained Prediction Aggregation (VPA)."""

import torch
from torch import nn
import torch.nn.functional as F


VPA_FEATURES = {
    "full": (0, 1, 2, 3, 4),
    "mask": (3,),
    "confidence": (0, 2),
    "agreement": (1,),
}


class VisibilityPredictionAggregator(nn.Module):
    """Aggregate the fused prediction and visible observational anchors."""

    def __init__(self, modalities, kind="full"):
        super().__init__()
        if kind not in VPA_FEATURES:
            raise ValueError(f"Unknown VPA feature set: {kind}")
        self.modalities = modalities
        self.kind = kind
        self.scorer = nn.Linear(5, 1)
        self.modality_bias = nn.Parameter(torch.zeros(modalities))
        self.register_buffer(
            "feature_selection",
            torch.tensor(
                [float(i in VPA_FEATURES[kind]) for i in range(5)]
            ),
        )
        nn.init.zeros_(self.scorer.weight)
        nn.init.constant_(self.scorer.bias, -3.0)

    def forward(self, fused_logits, branch_logits, mask):
        if branch_logits.shape[1] != self.modalities or mask.shape != branch_logits.shape[:2]:
            raise ValueError("VPA input dimensions do not match modalities")
        if mask.dtype != torch.bool or not mask.any(1).all():
            raise ValueError("VPA expects a bool mask with at least one visible modality")

        fused = fused_logits.softmax(-1)[:, 1]
        branch = branch_logits.softmax(-1)[:, :, 1]
        visible_fraction = mask.sum(1, keepdim=True).float() / self.modalities
        descriptors = torch.stack(
            (
                2 * (branch - 0.5).abs(),
                1 - (branch - fused[:, None]).abs(),
                (2 * (fused - 0.5).abs())[:, None].expand_as(branch),
                visible_fraction.expand_as(branch),
                branch,
            ),
            -1,
        )

        # Center branch biases for identifiability relative to the fusion score.
        bias = self.modality_bias - self.modality_bias.mean()
        scores = self.scorer(descriptors * self.feature_selection).squeeze(-1) + bias[None]
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = F.softmax(
            torch.cat((torch.zeros_like(fused[:, None]), scores), 1), 1
        )
        probability = weights[:, 0] * fused + (weights[:, 1:] * branch).sum(1)
        return probability, weights


class OaPA(nn.Module):
    """Inference adapter: one frozen backbone call followed by VPA."""

    def __init__(self, backbone, vpa):
        super().__init__()
        self.backbone = backbone
        self.vpa = vpa

    def forward(self, features, mask):
        output = self.backbone(features, mask)
        probability, weights = self.vpa(
            output["logits"], output["branch_logits"], mask
        )
        output["backbone_logits"] = output["logits"]
        output["backbone_probabilities"] = output["probabilities"]
        output["vpa_weights"] = weights
        output["probabilities"] = torch.stack((1 - probability, probability), -1)
        output["logits"] = output["probabilities"].clamp_min(1e-8).log()
        return output
