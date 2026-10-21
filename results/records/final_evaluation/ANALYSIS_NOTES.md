# Evidence notes for student interpretation

These are analysis notes, not a submission-ready report. Review the code,
measurements and images yourselves and record actual contributions honestly.

## Main comparisons

- CBAM has the highest mean test mAP50-95: 42.11 +/- 0.58 percentage points,
  versus baseline 40.70 +/- 1.03. Its paired improvements are +1.51, +0.81
  and +1.90 points for seeds 42, 43 and 44 respectively. All three are positive,
  but three seeds on one split are not proof of statistical significance.
- Median forward-only latency is 7.113 ms for CBAM versus 6.586 ms for baseline,
  about 8.0% higher. Added unfused parameters are only 2,284 (about 0.076%),
  illustrating why parameter count alone does not measure execution cost.
- SE has lower measured latency than CBAM (6.671 ms) but greater test AP
  variability (SD 1.66 points). It is not consistently better than baseline:
  seed 44 has a -0.23-point mAP50-95 difference.
- Coordinate Attention has the highest mean mAP50 (76.83%) by a tiny margin,
  but not the highest mAP50-95. Its mean scratch AP50-95 (32.02%) is slightly
  below baseline (32.10%). Avoid saying every attention variant helps every class.
- CBAM improves mean scratch AP50-95 from 32.10% to 32.94% and pest AP50-95
  from 49.30% to 51.28%. Scratch remains the harder class in this test split.
- At the fixed diagnostic threshold, unmatched predictions fall from a mean
  of 76.0 per baseline run to 57.3 for CBAM, while unmatched annotations remain
  near 53. These are geometric-matching diagnostic counts, not standard AP
  false-positive/false-negative totals or evidence about healthy-fruit rejection.

## Inspected examples

Inspected the first test batch's labels and CBAM seed-42 predictions:
`test/cbam_seed42/metrics/val_batch0_labels.jpg` and `val_batch0_pred.jpg`.
The filename prefix is an Ultralytics convention; this is the test directory.
Batch visualization uses the library's plotting threshold, distinct from the
explicit confidence-0.25 diagnostics and first-six individual previews.

- The first three displayed images show detections near the annotated scratch
  or pest-damage region. This is qualitative localization evidence, not a claim
  that every predicted box satisfies all AP IoU thresholds.
- In `23945062_20211027_154503_523.jpg` (fourth image), upper annotated scratches
  are absent from the displayed predictions, while the lower damaged region has
  overlapping scratch predictions. A right-edge prediction is labelled pest
  damage without a corresponding annotation in that displayed region.
- These observations illustrate misses, overlapping predictions and annotation
  disagreements. They do not prove the causal reason for failure, nor establish
  that the annotation or prediction is necessarily correct in ambiguous regions.

## Reporting boundaries

- Training curves still rise near epoch 50. This is an equal-budget comparison,
  not evidence that all architectures have fully converged.
- Error bars show sample SD across seeds, not a confidence interval or a test
  of significance. The dataset split is fixed across all runs.
- Report mean and individual seed results, not only each architecture's best seed.
- Timing is GPU forward-only, batch 1, FP32, fixed 640-square tensor and seed-42
  models. It excludes preprocessing, host transfers and NMS. Do not present
  its inverse as measured end-to-end application FPS.
- No robustness experiment or fully counted FLOPs benchmark was performed.
- Similarity/timestamp grouping cannot establish physical-apple independence.
  Controlled backgrounds and absence of healthy negatives limit generalization.
- Dataset class IDs are inferred from agreement with the source paper's counts;
  preserve that qualification from the dataset notes.

## Next phase

The core experiments are complete. Review these findings together, then develop
the final paper from the existing literature-review draft and prepare the video
presentation afterward. Architecture illustrations and final publication layout
can be prepared during paper authoring; no additional training is implied.
