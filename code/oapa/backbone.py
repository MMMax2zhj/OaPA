"""OaPA backbone for text-video-audio and text-image inputs."""

from dataclasses import asdict
import math

import torch
from torch import nn
import torch.nn.functional as F

from .config import BackboneConfig, mask_codes, pooled, sanitize
from .layers import (
    SequenceEncoder,
    VisibilityConditionedCompletion,
    VideoFusion,
)
from .upstream import mplmm_source


class TextImageFusion(nn.Module):
    """Two directed cross-modal stacks and two memory stacks."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        source = mplmm_source()

        def stack(depth):
            return source.TransformerEncoder(
                cfg.hidden,
                cfg.heads,
                depth,
                attn_dropout=cfg.dropout,
                relu_dropout=cfg.dropout,
                res_dropout=cfg.dropout,
                embed_dropout=cfg.dropout,
                attn_mask=False,
            )

        self.cross = nn.ModuleList([stack(cfg.layers) for _ in range(2)])
        self.memory = nn.ModuleList([stack(cfg.memory_layers) for _ in range(2)])
        self.proj1 = nn.Linear(2 * cfg.hidden, 2 * cfg.hidden)
        self.proj2 = nn.Linear(2 * cfg.hidden, 2 * cfg.hidden)
        self.head = nn.Linear(2 * cfg.hidden, 2)
        for name, buffer in self.named_buffers():
            if name.endswith("embed_positions._float_tensor"):
                buffer.zero_()

    def fuse(self, sequences, prompts=None):
        values = [x.transpose(0, 1) for x in sequences]
        summaries = []
        for k in range(2):
            x = self.cross[k](values[k], values[1 - k], values[1 - k])
            if prompts is not None:
                x = torch.cat([x, prompts[:, k].transpose(0, 1)], 0)
            summaries.append(self.memory[k](x)[-1])
        h = torch.cat(summaries, -1)
        h = h + self.proj2(
            F.dropout(
                F.relu(self.proj1(h)),
                p=self.cfg.dropout,
                training=self.training,
            )
        )
        return self.head(h)


class TextImageCompletion(nn.Module):
    """VPC for a two-modality text-image backbone."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        hidden = cfg.hidden
        prompt_length = cfg.prompt_length
        source = mplmm_source()

        if cfg.use_completion:
            self.generative = nn.Parameter(torch.zeros(2, hidden, prompt_length))
            self.translate = nn.ModuleList(
                [source.MLPLayer(hidden, hidden) for _ in range(2)]
            )
            self.to_length = nn.ModuleList(
                [
                    source.MLPLayer(
                        prompt_length + cfg.sequence_lengths[1 - k],
                        cfg.sequence_lengths[k],
                        True,
                    )
                    for k in range(2)
                ]
            )
        self.signal_missing = nn.ParameterList(
            [nn.Parameter(torch.zeros(length, hidden)) for length in cfg.sequence_lengths]
        )
        self.signal_visible = nn.ParameterList(
            [nn.Parameter(torch.zeros(length, hidden)) for length in cfg.sequence_lengths]
        )
        self.missing_type = nn.Parameter(
            torch.randn(2, prompt_length, hidden) * 0.02
        )
        self.type_projection = nn.ParameterList(
            [nn.Parameter(torch.randn(length, hidden) * 0.02) for length in cfg.sequence_lengths]
        )
        if cfg.conditional and cfg.use_completion:
            self.topic_prototypes = nn.Parameter(
                torch.randn(cfg.topics, hidden) * 0.02
            )
            self.topic_prompts = nn.Parameter(torch.randn(cfg.topics, hidden) * 0.02)
            self.queries = nn.ParameterList(
                [nn.Parameter(torch.randn(length, hidden) * 0.02) for length in cfg.sequence_lengths]
            )
            self.missing_prompt = nn.Embedding(4, hidden)
            self.cross_attention = nn.MultiheadAttention(
                hidden, cfg.heads, cfg.dropout, batch_first=True
            )
            self.condition_norm = nn.LayerNorm(hidden)
        self.decoders = (
            nn.ModuleList([nn.Linear(hidden, dim) for dim in cfg.input_dims])
            if cfg.use_completion
            else nn.ModuleList()
        )

    def missing_prompts(self, mask):
        signals = [
            torch.where(
                mask[:, k, None, None],
                self.signal_visible[k][None],
                self.signal_missing[k][None],
            )
            for k in range(mask.shape[1])
        ]
        projection = sum(
            signal.transpose(1, 2) @ parameter
            for signal, parameter in zip(signals, self.type_projection)
        )
        return self.missing_type[None] @ projection[:, None]

    def prepare(self, sequences, mask):
        prompts = self.missing_prompts(mask)
        if not self.cfg.use_completion:
            return sequences, prompts
        values, signals = [], []
        for k, x in enumerate(sequences):
            other = sequences[1 - k].transpose(1, 2)
            translated = self.translate[k](other)
            prompt = self.generative[k][None].expand(len(mask), -1, -1)
            generated = self.to_length[k](
                torch.cat([prompt, translated], -1).transpose(1, 2)
            )
            signal = torch.where(
                mask[:, k, None, None],
                self.signal_visible[k][None],
                self.signal_missing[k][None],
            )
            values.append(
                torch.where(mask[:, k, None, None], x, generated) + signal
            )
            signals.append(signal)

        if self.cfg.conditional:
            z = torch.stack(pooled(sequences), 1)
            mean = z.sum(1) / mask.sum(1, keepdim=True)
            topics = (
                F.normalize(mean, dim=-1)
                @ F.normalize(self.topic_prototypes, dim=-1).T
                / 0.2
            ).softmax(-1)
            entropy = -(topics * topics.clamp_min(1e-8).log()).sum(
                -1, keepdim=True
            )
            confidence = (1 - entropy / math.log(self.cfg.topics)).clamp(0, 1)
            condition = confidence * (topics @ self.topic_prompts) + self.missing_prompt(
                mask_codes(mask)
            )
            memory = torch.cat(sequences, 1)
            padding = torch.cat(
                [
                    ~mask[:, k, None].expand(-1, x.shape[1])
                    for k, x in enumerate(sequences)
                ],
                1,
            )
            for k in range(2):
                query = self.queries[k][None] + condition[:, None]
                residual, _ = self.cross_attention(
                    query,
                    memory,
                    memory,
                    key_padding_mask=padding,
                    need_weights=False,
                )
                conditioned = self.condition_norm(values[k] + residual + query)
                values[k] = torch.where(
                    mask[:, k, None, None], values[k], conditioned
                )
        return tuple(values), prompts


class OaPABackbone(nn.Module):
    """Backbone producing a fused prediction and observational anchors."""

    def __init__(self, cfg, variant="F111"):
        super().__init__()
        self.cfg = cfg
        self.variant = variant
        self.modalities = len(cfg.input_dims)
        self.projectors = nn.ModuleList(
            [SequenceEncoder(dim, cfg) for dim in cfg.input_dims]
        )
        self.branch_heads = nn.ModuleList(
            [nn.Linear(cfg.hidden, 2) for _ in range(self.modalities)]
        )

        if self.modalities == 3:
            self.completion = VisibilityConditionedCompletion(cfg)
            self.fusion = self.completion.fusion
        else:
            self.fusion = TextImageFusion(cfg)
            self.completion = TextImageCompletion(cfg)

        self.source_prompt = nn.Embedding(2, cfg.hidden)
        nn.init.normal_(self.source_prompt.weight, std=0.02)
        self.stage = "A"
        self.set_stage("A")

    def backbone_modules(self):
        modules = [self.projectors, self.branch_heads]
        if self.modalities == 3:
            core = self.completion.fusion.core
            modules.extend(
                module
                for name, module in core.named_children()
                if name.startswith("trans_") or name in ("proj1", "proj2", "out_layer")
            )
        else:
            modules.append(self.fusion)
        return modules

    def backbone_parameters(self):
        return [p for module in self.backbone_modules() for p in module.parameters()]

    def set_stage(self, stage):
        if stage not in ("A", "B", "J", "eval"):
            raise ValueError("OaPA stages are A, B, J, and eval")
        self.stage = stage
        self.requires_grad_(stage in ("B", "J"))
        for module in self.backbone_modules():
            module.requires_grad_(stage in ("A", "J"))
        for parameter in self.parameters():
            parameter.grad = None
        self.train(stage != "eval")
        return self

    def train(self, mode=True):
        super().train(mode)
        if mode and self.stage == "B":
            for module in self.backbone_modules():
                module.eval()
        if mode and self.stage == "A":
            self.completion.eval()
            for module in self.backbone_modules():
                module.train()
        if self.modalities == 3:
            self.completion.fusion.core.training = mode and self.stage in ("A", "J")
        return self

    def _encode(self, features, visible_mask):
        clean = sanitize(features, visible_mask, self.cfg.input_dims)
        sequences = []
        for k, (x, encoder, length) in enumerate(
            zip(clean, self.projectors, self.cfg.sequence_lengths)
        ):
            x = x[:, None] if x.ndim == 2 else x
            if x.shape[1] != length:
                raise ValueError(f"Sequence length mismatch for modality {k}")
            encoded = encoder.sequence(x)
            sequences.append(
                torch.where(
                    visible_mask[:, k, None, None],
                    encoded,
                    torch.zeros_like(encoded),
                )
            )
        return clean, tuple(sequences)

    def forward(self, features, visible_mask):
        clean, sequences = self._encode(features, visible_mask)
        projected = torch.stack(pooled(sequences), 1)
        branch_logits = torch.stack(
            [
                head(projected[:, k])
                for k, head in enumerate(self.branch_heads)
            ],
            1,
        )

        if self.stage == "A":
            completed, prompts = sequences, None
        else:
            generated, prompts = self.completion.prepare(
                sequences, visible_mask
            )
            completed = tuple(
                torch.where(visible_mask[:, k, None, None], sequences[k], x)
                for k, x in enumerate(generated)
            )

        fusion_inputs = tuple(
            x
            + self.source_prompt((~visible_mask[:, k]).long())[:, None]
            for k, x in enumerate(completed)
        )
        logits = self.fusion.fuse(fusion_inputs, prompts)
        return {
            "logits": logits,
            "probabilities": logits.softmax(-1),
            "branch_logits": branch_logits,
            "projected": projected,
            "sequences": sequences,
            "completed": completed,
            "representation": torch.cat(pooled(completed), -1),
            "reconstructed": tuple(
                decoder(x.mean(1))
                for decoder, x in zip(self.completion.decoders, completed)
            ),
        }

    def save(self, path, metadata=None):
        payload = {
            "family": "oapa-backbone",
            "config": asdict(self.cfg),
            "variant": self.variant,
            "stage": self.stage,
            "model": self.state_dict(),
        }
        if metadata is not None:
            payload["metadata"] = metadata
        torch.save(payload, path)

    @classmethod
    def load(cls, path, device="cpu"):
        payload = torch.load(path, map_location=device, weights_only=True)
        if payload.get("family") != "oapa-backbone":
            raise ValueError("Expected an OaPA backbone checkpoint")
        model = cls(
            BackboneConfig(**payload["config"]),
            variant=payload.get("variant", "F111"),
        ).to(device)
        model.load_state_dict(payload["model"])
        model.set_stage(payload["stage"])
        model.eval()
        return model


def build_oapa_backbone(
    input_dims, sequence_lengths, model_settings=None, variant="F111"
):
    settings = dict(model_settings or {})
    if variant.startswith("F") and len(variant) == 4:
        settings["use_completion"] = variant[2] == "1"
    settings.update(
        input_dims=tuple(input_dims),
        sequence_lengths=tuple(sequence_lengths),
    )
    return OaPABackbone(BackboneConfig(**settings), variant=variant)
