# OaPA Paper Experiments

This directory contains the experiment specifications for the OaPA paper:
**How Much to Trust Completion? Observation-Anchored Prediction Aggregation
for Incomplete Multimodal Fake News Detection**.

The paper scope covers:

- fixed-view detection (P0);
- matched modality missingness from `0.1` through `0.7` (P1);
- MPD, VPC, and DCR component ablations;
- VPA prediction-aggregation controls;
- semantic versus random DCR reference selection.

## Contents

```text
configs/                  Final and ablation configurations
protocol.json             Dataset splits and P0/P1 protocols
paper_output_schema.json  Paper table schema
input_manifest.json       Feature-cache paths and hashes
```

The three bits in `Fxyz` correspond to MPD, VPC, and DCR in that order.
`F111` enables all three components:

```text
F000  none
F100  MPD
F010  VPC
F001  DCR
F110  MPD + VPC
F101  MPD + DCR
F011  VPC + DCR
F111  MPD + VPC + DCR
```

`S-random` replaces semantic DCR references with random references for the
paper's reference-selection comparison.

Training is started with `code/scripts/train_oapa.py`. Dataset-specific
configurations are in `configs/`.
