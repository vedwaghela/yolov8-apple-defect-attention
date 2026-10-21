# Controlled attention comparison

## Final study status, verified 19 October 2026

The three-seed comparison and final evaluation are complete. See
`runs/final_evaluation/RESULTS.md` for validation/test means, sample standard
deviations, class AP, paired seed differences and controlled forward-only
timing. `FINAL_PROTOCOL.md` was fixed before additional training and test use.
The test set has now been evaluated; references below to an unused test set
describe the earlier pilot and seed-42 stages. No test-driven tuning is planned.

## Fixed design

All variants use the same pretrained YOLOv8n detector, two defect classes,
grouped_v1 split and training settings. Each inserts exactly one attention
module after the output of C2f backbone layer 6, before layer 7. This is the
P4/stride-16 stage: 128 channels and 40x40 spatial resolution at 640px input.
The neck also receives this attended layer through its existing skip route.
The baseline remains unmodified. No detection-head or loss changes are made.

This location retains more spatial detail than stride 32 while keeping overhead
small. It is a preselected engineering choice, not an optimized insertion site.
Conclusions concern these configurations at this site, not universal rankings
of attention mechanisms. The study does not claim a novel attention mechanism.

| Variant | Configuration | Added parameters | Total parameters |
| --- | --- | ---: | ---: |
| Baseline | No added attention | 0 | 3,011,238 |
| SE | Average pooling, reduction 16, bias-free channel MLP | 2,048 | 3,013,286 |
| CBAM | Avg/max shared channel MLP, reduction 16; 7x7 spatial gate | 2,284 | 3,013,522 |
| Coordinate Attention | Axis pooling, reduction 32, minimum width 8 | 3,352 | 3,014,590 |

Counts are for the unfused two-class detector. Fusion changes parameter counts;
use one consistent convention in comparisons. Parameter counts are not latency
measurements. The prescribed reduction ratios differ, so this compares common
module configurations, not equal-parameter-budget variants.

## Implementation and sources

`attention.py` implements the modules using PyTorch primitives. Channel MLPs
are implemented as 1x1 convolutions on pooled features. SE omits biases. CBAM
uses biases in the shared channel MLP and includes spatial batch normalization
(eps=1e-5, momentum=0.01), matching the authors' reference design. Coordinate
Attention uses a shared bottleneck with batch normalization and Hardswish,
followed by separate sigmoid gates along height and width. Attention-specific
batch norms use their reference defaults; existing YOLO backbone batch norms
are unchanged.

References used to check module definitions:
- SE: Hu, Shen and Sun, Squeeze-and-Excitation Networks, CVPR 2018,
  https://doi.org/10.1109/CVPR.2018.00745.
  Implementation cross-check (third party):
  https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py.
- CBAM: Woo et al., CBAM: Convolutional Block Attention Module, ECCV 2018.
  Authors' implementation:
  https://github.com/Jongchan/attention-module/blob/master/MODELS/cbam.py.
- Coordinate Attention: Hou, Zhou and Feng, Coordinate Attention for Efficient
  Mobile Network Design, CVPR 2021. Authors' implementation:
  https://github.com/houqb/CoordAttention/blob/main/coordatt.py.

Integration preserves C2f parameter names by changing that block's dispatch to
an importable AttentionC2f subclass and registering an additional attention
submodule. The existing convolutions are not recreated. The trainer constructs
the detector, adds attention, then loads matching pretrained weights. Thus
base weights retain their names; new attention weights are initialized locally.
The same COCO-pretrained checkpoint is used for every run, NOT the trained
50-epoch fruit baseline. Attention is also present when models are reconstructed
from their saved configuration and weights.

Module initialization uses a temporary CPU RNG context so it does not advance
the shared RNG stream for later training operations. This does not guarantee
identical augmentation sequences or perfect hardware-level determinism.

## Validation and execution

`test_attention.py` checks known constant-gate outputs, unchanged common state
keys and weights, non-square shapes, finite attention gradients, whole-model
checkpoint round trips, fused inference agreement, and reconstruction through
each custom trainer. Results are recorded in `attention_checks.json`.

Run from the project root:

```powershell
baseline/.venv/Scripts/python.exe baseline/test_attention.py
baseline/.venv/Scripts/python.exe baseline/train.py --variant se --pilot
baseline/.venv/Scripts/python.exe baseline/train.py --variant cbam --pilot
baseline/.venv/Scripts/python.exe baseline/train.py --variant ca --pilot
```

Each pilot is one epoch at 640px, batch 4, seed 42, on the 500 training and 106
validation images. Pilots verify integration, not which mechanism is best.
The 104 test images are not evaluated. Full comparison commands, after pilots:

```powershell
baseline/.venv/Scripts/python.exe baseline/train.py --variant se --epochs 50 --seed 42
baseline/.venv/Scripts/python.exe baseline/train.py --variant cbam --epochs 50 --seed 42
baseline/.venv/Scripts/python.exe baseline/train.py --variant ca --epochs 50 --seed 42
```

Full runs use AdamW, lr0=0.001, lrf=0.01, weight decay=0.0005, FP32,
no early stopping and mosaic disabled in the last 10 epochs, as in the
completed baseline. Best checkpoints are selected by validation mAP50-95.
Use repeated matched seeds before drawing conclusions. The first full attention
runs are complete (see below); controlled latency has not yet been benchmarked.

Keep `attention.py` available on the Python import path when loading custom
checkpoints. Running the project scripts from this folder provides it. A stock
YOLO CLI without this module available cannot reliably load these checkpoints.
Export formats beyond PyTorch have not been tested.

## Completed pilot checks, 17 October 2026

All three pilots completed training, validation and checkpoint export on the
local RTX 2050. Their metadata settings match the earlier baseline pilot
exactly, excluding output directory/name. The frozen split hash also matches.

- SE: `runs/se_pilot_seed42_20261017T065105371631`
- CBAM: `runs/cbam_pilot_seed42_20261017T065157416521`
- Coordinate Attention: `runs/ca_pilot_seed42_20261017T065247293011`

Each directory contains weights, results.csv, full training arguments, code
snapshots and experiment metadata. Numeric pilot scores must not be compared
against the 50-epoch baseline as evidence for or against attention. All pilots
passed, followed by the equal-budget 50-epoch comparison below.

## Completed 50-epoch comparison, 17 October 2026

Queue: `runs/attention_queue_20261017T065900`. All three models completed
sequentially on the RTX 2050. Training settings, package versions and frozen
split hashes match the baseline. Each row uses the epoch selected by highest
validation mAP50-95, with the other metric taken from that same epoch.

| Model | Best epoch | Validation mAP50 | Validation mAP50-95 |
| --- | ---: | ---: | ---: |
| Baseline | 38 | 0.78393 | 0.41001 |
| SE | 50 | 0.78980 | 0.42519 |
| CBAM | 49 | 0.78138 | 0.43463 |
| Coordinate Attention | 47 | 0.79838 | 0.42377 |

CBAM is 2.462 percentage points above baseline on validation mAP50-95 for
this seed, but slightly below baseline on mAP50. Coordinate Attention has
the highest mAP50. These results do not establish a universal winner or
statistical significance. SE's best epoch is the final epoch, so convergence
under this fixed budget is not established. Any extended-budget study must
apply the same rule to every variant, including baseline.

All checkpoints, CSVs, learning plots and fixed-list prediction previews are
saved in the individual run folders listed in the queue's `comparison.json`.
Independent split-integrity checks passed after completion. The held-out test
set has not been evaluated. Next steps are matched repeated seeds and a
controlled inference-cost benchmark before final test evaluation. Recorded
training durations are not suitable evidence of inference-speed differences.
