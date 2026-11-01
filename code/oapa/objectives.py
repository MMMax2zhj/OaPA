"""Training objectives used by OaPA."""

import torch
from torch import nn
import torch.nn.functional as F

from .upstream import distillation_losses


class ModalityPrioritizedDistillation(nn.Module):
    """Distill visible teacher evidence from each modality into the fusion."""

    def __init__(self, temperature=2.0):
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = temperature

    def forward(
        self,
        student_logits,
        student_features,
        teacher_logits,
        teacher_features,
        visible_mask,
    ):
        if student_features.shape[:2] != teacher_features.shape[:2]:
            raise ValueError("Student and teacher features must match [batch, modality, dim]")
        if student_logits.shape[0] != len(visible_mask):
            raise ValueError("Visible mask batch size mismatch")
        losses = distillation_losses()
        kd = student_logits.sum() * 0
        feature = student_features.sum() * 0
        for k in range(visible_mask.shape[1]):
            visible = visible_mask[:, k]
            if visible.any():
                kd = kd + visible.sum() * losses.distillation_loss(
                    student_logits[visible],
                    teacher_logits[visible, k],
                    self.temperature,
                )
                feature = feature + visible.sum() * losses.feature_loss(
                    student_features[visible, k],
                    teacher_features[visible, k],
                )
        count = visible_mask.sum()
        return kd / count, feature / count


class DiscriminativeContrastiveRegularization(nn.Module):
    """Hard-reference contrast for class-separable completed representations."""

    def __init__(self, negative_k=1, temperature=1.0, reference="semantic"):
        super().__init__()
        if not isinstance(negative_k, int) or isinstance(negative_k, bool) or negative_k < 1:
            raise ValueError("negative_k must be a positive integer")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if reference not in ("semantic", "random"):
            raise ValueError("reference must be semantic or random")
        self.negative_k = negative_k
        self.temperature = temperature
        self.reference = reference

    def forward(self, representation, labels, ids, groups, split="D_fit", rng=None):
        if split != "D_fit":
            raise ValueError("DCR is restricted to the D_fit split")
        n = len(representation)
        if representation.ndim != 2 or labels.shape != (n,) or len(ids) != n or len(groups) != n:
            raise ValueError("DCR expects [batch, dim] and aligned labels/metadata")
        if labels.dtype != torch.long or not ((labels == 0) | (labels == 1)).all():
            raise ValueError("DCR expects binary int64 labels")
        if not all(ids) or not all(groups) or not torch.isfinite(representation).all():
            raise ValueError("DCR requires finite representations and explicit IDs")

        unit = F.normalize(representation, dim=-1)
        similarity = unit @ unit.T
        eligible = torch.tensor(
            [
                [
                    i != j and ids[i] != ids[j] and groups[i] != groups[j]
                    for j in range(n)
                ]
                for i in range(n)
            ],
            dtype=torch.bool,
            device=representation.device,
        )
        same = labels[:, None] == labels[None, :]
        positive_mask = eligible & same
        negative_mask = eligible & ~same
        valid = positive_mask.any(1) & negative_mask.any(1)

        terms, pairs = [], []
        for anchor in valid.nonzero(as_tuple=True)[0].tolist():
            candidates = negative_mask[anchor].nonzero(as_tuple=True)[0]
            positives = positive_mask[anchor].nonzero(as_tuple=True)[0]
            if self.reference == "random":
                if rng is None:
                    raise ValueError("Random-reference DCR requires an RNG")
                positive = positives[int(rng.integers(len(positives)))]
                chosen = rng.choice(
                    len(candidates),
                    size=min(self.negative_k, len(candidates)),
                    replace=False,
                )
                negatives = candidates[torch.as_tensor(chosen, device=candidates.device)]
            else:
                positive = similarity[anchor].detach().masked_fill(
                    ~positive_mask[anchor], -torch.inf
                ).argmax()
                order = torch.argsort(
                    similarity[anchor, candidates].detach(),
                    descending=True,
                    stable=True,
                )
                negatives = candidates[order[: self.negative_k]]
            logits = torch.cat(
                [
                    similarity[anchor, positive].reshape(1),
                    similarity[anchor, negatives],
                ]
            ) / self.temperature
            terms.append(torch.logsumexp(logits, 0) - logits[0])
            pairs.append(
                {
                    "anchor": anchor,
                    "positive": int(positive),
                    "negatives": negatives.tolist(),
                }
            )

        loss = torch.stack(terms).mean() if terms else representation.sum() * 0
        coverage = {
            "attempted": n,
            "used": len(terms),
            "skipped": n - len(terms),
            "no_positive": int((~positive_mask.any(1)).sum()),
            "no_negative": int((~negative_mask.any(1)).sum()),
            "pairs": pairs,
        }
        return loss, coverage
