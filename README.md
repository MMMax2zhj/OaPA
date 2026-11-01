# OaPA

OaPA (Observation-anchored Prediction Aggregation) detects fake news under
missing modalities. This repository contains the complete CoMPA4R four-route
experiment: staged MPD learning, VPC, DCR, singleton experts, deployed Stage-J
selection, VPA calibration, frozen evaluation, audit, and table export.

## Setup

Python 3.10+ is required.

```bash
pip install -r requirements.txt
python -m unittest discover -s code/tests -v
```

## Data

Supply audited feature caches separately:

```text
data/cache/full_protocol_v1/PolitiFact.pt
data/cache/sequence_protocol_v2/FakeSV.pt
data/cache/sequence_protocol_v2/FakeTT.pt
```

Datasets, feature caches, experts, and trained weights are not included.

## Model Training

The default configs train the final `F111` backbone. Component configs
`F000`–`F110` and the random-reference `S-random` control are under
`experiments/four_route/configs/`.

```bash
python code/scripts/train_oapa.py \
  --cache /path/to/FakeSV.pt \
  --config experiments/four_route/configs/SV_OaPA_F111.json \
  --output outputs/oapa4r/runs/example \
  --seed 20261011 \
  --device cuda \
  --epoch-checkpoints \
  --save-j-epochs
```

## Four-Route Pipeline

The full 120-job matrix and the paper-scoped 72-job subset are defined in
`experiments/four_route/`. The pipeline fits experts only on `D_fit`, selects
the deployed J checkpoint on `D_val`, calibrates VPA on the configured
calibration split, freezes all artifacts, and only then evaluates `D_test`.

```bash
python code/scripts/oapa4r.py --phase preflight --execute
python code/scripts/oapa4r.py --phase campaign --execute
```

For only `MAIN_P0` and `MAIN_P1`:

```bash
python code/scripts/oapa4r_main.py --execute
```

All execution phases require `--execute`; omitting it performs a preview.
See [experiments/four_route/README.md](experiments/four_route/README.md) for
the retained routes and protocol.
