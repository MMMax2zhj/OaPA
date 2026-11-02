# OaPA

Reference implementation of **How Much to Trust Completion?
Observation-Anchored Prediction Aggregation for Incomplete Multimodal Fake
News Detection**.

OaPA is designed for fake news detection when some modalities are missing.
Its central idea is to keep predictions obtained directly from pre-completion
visible encodings as **observational anchors**, while a source-tagged
completion backbone produces a multimodal fused prediction. After the backbone
is frozen, **Visibility-constrained Prediction Aggregation (VPA)** decides how
far the final prediction should move from the fused result toward the visible
anchors.

## Method

### Observational Anchors

For every visible modality, a unimodal head maps the pooled encoder output to a
pre-completion probability. These probabilities are observational anchors. They
are produced alongside the fused prediction in one forward pass and represent
direct evidence that has not passed through missing-modality completion.

### Visibility-conditioned Prompt Completion

VPC reconstructs missing streams from the observed encodings while preserving
visible representations. It uses:

- generative prompts to form missing-stream tokens;
- prototype-guided queries with entropy-based confidence;
- missing-signal and missing-type prompts to encode modality availability;
- cross-modal attention over visible tokens;
- learnable observed/generated source tags for fusion.

The cross-modal Transformer fuses the source-tagged completed streams and
visibility prompts into the fused probability.

### Representation Regularization

**Modality-prioritized Distillation (MPD)** aligns the fusion student with
frozen unimodal teachers using logit and pooled-feature distillation. During
the first five epochs of Stage A, encoder updates are restricted to visible
modalities with the lowest batch-mean ground-truth teacher probability,
including ties. The restriction is removed for the rest of Stage A and for
Stage J.

**Discriminative Contrastive Regularization (DCR)** promotes class separation
in concatenated completed representations. Each anchor uses the most similar
same-label reference and the hardest opposite-label reference, excluding
self-pairs, identical sample IDs, and duplicate groups.

### Staged Backbone Training

The backbone follows three stages:

1. **Stage A:** warm up visible encoders and unimodal anchor heads.
2. **Stage B:** freeze the visible backbone and train completion modules on
   masked inputs.
3. **Stage J:** jointly fine-tune the backbone with task, distillation,
   reconstruction, and contrastive objectives.

Every stage restores its earliest best validation checkpoint. The selected
Stage J backbone is then frozen for VPA fitting.

### Visibility-constrained Prediction Aggregation

VPA forms a five-dimensional descriptor for each visible anchor from:

- anchor certainty margin;
- branch-fusion agreement, the complement of decision shift;
- fusion certainty margin;
- visible-modality fraction;
- anchor probability.

A shared scorer produces branch scores, while a fixed zero score represents
the fused prediction. Missing branches receive a weight of zero, and the
remaining weights form a convex combination of the fused probability and
visible anchors.

VPA is fitted on the frozen backbone using class-balanced binary cross-entropy
and a visible-anchor-weight penalty. The paper uses `lambda_reg = 0.015`,
AdamW with learning rate `0.03`, weight decay `0.02`, and 200 updates.
Inference requires a single forward pass without teachers or contrastive
mining.

## Evaluation

The paper evaluates OaPA on three datasets:

- **FakeSV:** text, video, and audio;
- **FakeTT:** text, video, and audio;
- **PolitiFact:** text and image.

Two protocols are reported:

- **Fixed-view detection (P0):** evaluate every legal nonempty view and report
  Macro-F1 per view and `F_all`, the mean over all legal views.
- **Matched modality missingness (P1):** train and test with independently
  missing modalities at rates `0.1` to `0.7`, retaining at least one visible
  modality and averaging over ten fixed test-mask sets.

All results use a decision threshold of `0.5`. The paper reports first place
among the compared methods in 14 of 17 fixed-view settings and all 21
matched-missingness settings. VPA improves over the uncalibrated backbone by
`+2.92` and `+3.28` percentage points on FakeSV and FakeTT, respectively.

## Repository Layout

```text
.
|-- code/oapa/                 Model, objectives, aggregation, and protocol code
|-- code/scripts/              Backbone training interface
|-- code/tests/                Unit and experiment-contract tests
|-- experiments/oapa/configs/  Final, component-ablation, and DCR-reference configs
`-- requirements.txt
```

## Setup

Python 3.10+ and PyTorch are required. A CUDA-capable GPU is recommended for
training and paper-scale evaluation.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m unittest discover -s code/tests -v
```

For PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

## Data

Feature caches and trained checkpoints are not included. Place the caches at:

```text
data/cache/sequence_protocol_v2/FakeSV.pt
data/cache/sequence_protocol_v2/FakeTT.pt
data/cache/full_protocol_v1/PolitiFact.pt
```

FakeSV and FakeTT use `32 x 384` text, `16 x 512` video, and `1 x 768` audio
features. PolitiFact uses `32 x 384` text and `49 x 768` image features. Each
cache must follow the split, visibility-mask, and metadata interface expected
by `code/oapa/protocol.py`.

## Train One Model

`F111` enables all three representation components. The bits in `Fxyz`
correspond to MPD, VPC, and DCR, respectively:

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

Train the full configuration on FakeSV with:

```bash
python code/scripts/train_oapa.py \
  --cache data/cache/sequence_protocol_v2/FakeSV.pt \
  --config experiments/oapa/configs/SV_OaPA_F111.json \
  --output outputs/oapa/runs/fakesv_f111 \
  --seed 20261011 \
  --device cuda
```

The final configuration uses a four-head Transformer with hidden width `128`,
AdamW with batch size `16`, and weight decay `0.01`. Teacher, A, B, and J
budgets are `10/20/10/10` epochs, with a `0.1x` learning-rate multiplier for
pretrained components in Stage J.

`S-random` uses random DCR references and is the control for the paper's
semantic-reference comparison. Dataset-specific configuration files are
provided for FakeSV, FakeTT, and PolitiFact.

## Evaluate OaPA

For inference, provide the feature sequences and a nonempty visibility mask to
the frozen OaPA backbone. The backbone returns fused logits, branch logits, and
the modality sequences. VPA then combines the fused probability with the
visible observational anchors; branches absent from the mask receive zero
weight.

For P0, evaluate every legal nonempty modality view and report per-view
Macro-F1 plus their arithmetic mean. For P1, use the same nominal missing rate
for training and testing, retain at least one modality per sample, and average
Macro-F1 over the ten fixed test-mask sets.

## License

This project is released under the terms in [LICENSE](LICENSE).
