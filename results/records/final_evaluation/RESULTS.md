# Completed experimental analysis

Seeds 42, 43 and 44; 50 epochs each. Each checkpoint was selected on validation
mAP50-95 before test evaluation. Values below are mean +/- sample standard deviation
across three seeds, in percentage points. Standard deviation is not a confidence interval.

| Model | Validation mAP50-95 | Test mAP50 | Test mAP50-95 | Parameters |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 41.38 +/- 0.55 | 75.54 +/- 1.92 | 40.70 +/- 1.03 | 3,011,238 |
| SE | 42.08 +/- 1.14 | 76.80 +/- 1.37 | 41.58 +/- 1.66 | 3,013,286 |
| CBAM | 42.27 +/- 1.07 | 76.68 +/- 0.93 | 42.11 +/- 0.58 | 3,013,522 |
| Coordinate Attention | 42.16 +/- 0.28 | 76.83 +/- 1.22 | 41.52 +/- 0.70 | 3,014,590 |

Validation here is re-evaluation of serialized checkpoints, not training CSV maxima.
Small differences from earlier CSV summaries are expected.

## Per-class test AP

| Model | Class | AP50 | AP50-95 |
| --- | --- | ---: | ---: |
| Baseline | scratch | 64.88 +/- 3.42 | 32.10 +/- 1.48 |
| Baseline | pest_damage | 86.20 +/- 1.10 | 49.30 +/- 0.78 |
| SE | scratch | 66.17 +/- 3.18 | 32.43 +/- 2.16 |
| SE | pest_damage | 87.43 +/- 0.96 | 50.74 +/- 1.17 |
| CBAM | scratch | 65.53 +/- 1.82 | 32.94 +/- 2.18 |
| CBAM | pest_damage | 87.82 +/- 0.47 | 51.28 +/- 1.14 |
| Coordinate Attention | scratch | 65.94 +/- 1.86 | 32.02 +/- 2.18 |
| Coordinate Attention | pest_damage | 87.71 +/- 0.72 | 51.02 +/- 1.00 |

## Forward-only GPU timing

RTX 2050, FP32, fused model, batch 1, 640x640, seed-42 checkpoints fixed in advance.
Three rotated-order blocks, 50 warm-ups and 200 CUDA-event samples per block.
Excludes file I/O, preprocessing, host transfers and NMS. Not application FPS.

| Model | Median ms | p95 ms | SD of block medians ms |
| --- | ---: | ---: | ---: |
| Baseline | 6.586 | 9.674 | 0.011 |
| SE | 6.671 | 9.704 | 0.017 |
| CBAM | 7.113 | 9.996 | 0.081 |
| Coordinate Attention | 6.898 | 9.270 | 0.033 |

## Fixed-threshold test diagnostics

Mean counts per seed over all 104 test images. Confidence >=0.25, geometric IoU >=0.5.
One-to-one geometry matching ignores class first; wrong-class pairs are separate.
These counts are not AP or standard class-aware precision/recall.

| Model | Correct | Wrong class | Unmatched annotations | Unmatched predictions |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 175.0 | 5.0 | 53.0 | 76.0 |
| SE | 175.3 | 4.3 | 53.3 | 57.0 |
| CBAM | 174.3 | 5.3 | 53.3 | 57.3 |
| Coordinate Attention | 175.0 | 5.0 | 53.0 | 55.7 |

## Paired seed differences

Test mAP50-95 differences relative to baseline (percentage points); descriptive only.
Three training seeds on one split do not establish statistical significance.

| Model | Seed 42 | Seed 43 | Seed 44 |
| --- | ---: | ---: | ---: |
| SE | +1.49 | +1.39 | -0.23 |
| CBAM | +1.51 | +0.81 | +1.90 |
| Coordinate Attention | +1.12 | -0.81 | +2.15 |

## Interpretation limits

The split is fixed and heuristic groups are isolated, but physical apple identities are unknown.
Small controlled-background data, no healthy negatives, one attention location, three seeds
and one fixed budget limit generalization. No illumination/blur/orchard robustness claim is made.
Unmatched predictions may reflect annotation omissions as well as model errors; inspect images.
Examples are the first six test-list entries for each seed-42 model, not selected successes.
FLOPs are omitted because generic hook-based profilers may miss functional attention operations.
Do not tune models after inspecting test outcomes. Preserve this evaluation as the final study.
