import sys
import unittest
import uuid
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

from oapa.aggregation import OaPA, VisibilityPredictionAggregator
from oapa.backbone import OaPABackbone, build_oapa_backbone
from oapa.protocol import views


class CountingBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, features, mask):
        self.calls += 1
        logits, branches = features
        return {
            "logits": logits,
            "branch_logits": branches,
            "probabilities": logits.softmax(-1),
        }


class OaPATests(unittest.TestCase):
    def test_backbone_forward_and_hidden_isolation(self):
        torch.set_num_threads(2)
        settings = dict(
            hidden=8,
            heads=2,
            layers=1,
            memory_layers=1,
            encoder_layers=1,
            dropout=0,
            prompt_length=2,
            topics=2,
        )
        for dims, lengths, modalities in (
            ((8, 6), (5, 3), ("T", "I")),
            ((8, 6, 4), (5, 3, 1), ("T", "V", "A")),
        ):
            with self.subTest(modalities=modalities):
                torch.manual_seed(20261101)
                backbone = build_oapa_backbone(dims, lengths, settings).set_stage("J")
                mask = torch.tensor(list(views(modalities).values()), dtype=torch.bool)
                features = tuple(
                    torch.randn(len(mask), length, dim)
                    for length, dim in zip(lengths, dims)
                )
                hidden_nan = tuple(
                    torch.where(
                        mask[:, k, None, None],
                        feature,
                        torch.full_like(feature, float("nan")),
                    )
                    for k, feature in enumerate(features)
                )
                with torch.inference_mode():
                    expected = backbone(features, mask)
                    actual = backbone(hidden_nan, mask)
                torch.testing.assert_close(
                    actual["probabilities"],
                    expected["probabilities"],
                    rtol=0,
                    atol=0,
                )
                self.assertEqual(expected["probabilities"].shape, (len(mask), 2))
                self.assertEqual(expected["branch_logits"].shape[:2], (len(mask), len(dims)))
                self.assertTrue(torch.isfinite(expected["probabilities"]).all())

    def test_vpa_single_forward_and_missing_branch_invariance(self):
        fused = torch.tensor([[0.3, -0.1], [0.1, 0.8], [-0.5, 0.2]])
        branches = torch.tensor(
            [
                [[0.2, 0.7], [-0.1, 0.1]],
                [[0.6, -0.2], [0.2, 0.9]],
                [[0.1, 0.8], [0.7, -0.1]],
            ]
        )
        mask = torch.tensor([[True, False], [True, True], [False, True]])
        vpa = VisibilityPredictionAggregator(2)
        with torch.no_grad():
            vpa.scorer.weight.fill_(0.2)
        backbone = CountingBackbone()
        model = OaPA(backbone, vpa)
        output = model((fused, branches), mask)
        self.assertEqual(backbone.calls, 1)
        self.assertTrue(torch.allclose(output["vpa_weights"].sum(1), torch.ones(3)))
        self.assertEqual(output["vpa_weights"][0, 2].item(), 0)
        self.assertEqual(output["vpa_weights"][2, 1].item(), 0)

        changed = branches.clone()
        changed[~mask] = torch.tensor([[-1000.0, 1000.0]])
        other = model((fused, changed), mask)
        torch.testing.assert_close(
            other["probabilities"], output["probabilities"], rtol=0, atol=0
        )
        self.assertGreater(output["vpa_weights"][:, 1:].sum(1).std().item(), 0)

    def test_checkpoint_round_trip(self):
        model = build_oapa_backbone(
            (8, 6, 4),
            (5, 3, 1),
            dict(hidden=8, heads=2, layers=1, memory_layers=1, encoder_layers=1,
                 dropout=0, prompt_length=2, topics=2),
        ).set_stage("eval")
        path = ROOT / f".oapa-test-{uuid.uuid4().hex}.pt"
        try:
            model.save(path)
            restored = OaPABackbone.load(path)
        finally:
            path.unlink(missing_ok=True)
        self.assertEqual(restored.cfg, model.cfg)
        self.assertEqual(restored.stage, "eval")


if __name__ == "__main__":
    unittest.main()
