# OaPA

Reference implementation of **How Much to Trust Completion?
Observation-Anchored Prediction Aggregation for Incomplete Multimodal Fake
News Detection**.

OaPA detects fake news when one or more modalities are missing. It combines
source-tagged visibility-conditioned prompt completion (VPC),
modality-prioritized distillation (MPD), discriminative contrastive
regularization (DCR), and visibility-constrained prediction aggregation (VPA).

## Setup

Python 3.10+ is required.

```bash
pip install -r requirements.txt
python -m unittest discover -s code/tests -v
```

## Data

Supply audited feature caches separately:

```text
data/cache/sequence_protocol_v2/FakeSV.pt
data/cache/sequence_protocol_v2/FakeTT.pt
data/cache/full_protocol_v1/PolitiFact.pt
```

FakeSV and FakeTT use text, video, and audio features. PolitiFact uses text and
image features. Datasets, feature caches, and trained weights are not included.

## Train

Run one complete configured main-model cell:

```bash
python code/scripts/run_oapa.py \
  --phase train-cell \
  --run-id MAIN_P0_SV_OaPA_F111_s20261011 \
  --device cuda \
  --execute
```

The run configuration, data split, seed, and output location are defined in
`experiments/oapa/`. Final and ablation configurations are in
`experiments/oapa/configs/`. Each invocation trains only the selected run.

Outputs are written under `outputs/oapa/`. The command trains the selected
backbone, prepares its paper-model checkpoints and VPA artifacts, and does not
evaluate `D_test`.

## License

This project is released under the terms in [LICENSE](LICENSE).
