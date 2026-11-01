# OaPA

OaPA (Observation-anchored Prediction Aggregation) detects fake news when one or more modalities are missing. This repository contains the final model, backbone training, VPA fitting, and test interfaces.

## Setup

Python 3.10+ is required.

```bash
pip install -r requirements.txt
python -m unittest discover -s code/tests -v
```

## Data

Supply audited feature caches separately. The default cache layout is:

```text
data/cache/full_protocol_v1/PolitiFact.pt
data/cache/sequence_protocol_v2/FakeSV.pt
data/cache/sequence_protocol_v2/FakeTT.pt
```

Datasets, feature caches, and trained weights are not included.

## Train

```bash
python code/scripts/train_oapa.py \
  --cache /path/to/FakeSV.pt \
  --config code/configs/fakesv.json \
  --output outputs/runs/OaPA_Backbone_SV_s20261011 \
  --seed 20261011 \
  --device cuda
```

Configs for `fakesv`, `fakett`, and `politifact` are in `code/configs/`.

## Fit and test

```bash
python code/scripts/test_oapa.py --phase fit --conditions SV --seeds 20261011 --runs-dir outputs/runs --cache-root /path/to/cache --output outputs/oapa --device cuda
python code/scripts/test_oapa.py --phase test --conditions SV --seeds 20261011 --runs-dir outputs/runs --cache-root /path/to/cache --output outputs/oapa --device cuda
```
