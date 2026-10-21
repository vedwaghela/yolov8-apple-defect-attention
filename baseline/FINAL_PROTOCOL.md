# Frozen completion protocol, 18 October 2026

Fixed before additional training and before any test-set evaluation.

- Compare baseline, SE, CBAM and Coordinate Attention at the existing layer 6.
- Seeds 42, 43 and 44 for every model. Reuse completed seed-42 experiments.
- Train each new run for 50 epochs using the unchanged train.py settings,
  COCO initialization, grouped_v1 split, 640px inputs and batch 4.
- No hyperparameter, insertion-site or training-budget search.
- Select each checkpoint using highest validation mAP50-95, as implemented
  by the pinned Ultralytics trainer. Do not select seeds by test performance.
- Evaluate all twelve best checkpoints on the 104 held-out images only after
  training is complete. Report each seed, arithmetic mean and sample standard
  deviation. Three seeds describe variability, not proof of significance.
- Evaluation: FP32, GPU 0, 640px, batch 4, confidence 0.001, NMS IoU 0.7,
  max_det 300, no test-time augmentation. Report mAP50, mAP50-95 and class AP.
- Error inspection: fixed confidence 0.25, NMS IoU 0.7, matching IoU 0.5.
  Label qualitative examples explicitly and avoid inferring dataset-wide
  trends from a handful of images. No test-driven model changes afterward.
- Timing: same RTX 2050, FP32, fused evaluation models, batch 1, fixed
  1x3x640x640 GPU tensor. CUDA events, 50 warm-up iterations and 200 timed
  iterations per block, three blocks in rotated model order. Forward-only:
  exclude decoding files, preprocessing, CPU transfers and NMS. Report median
  and p95 plus block variability. This is not end-to-end application latency.
- Report unfused parameter counts. FLOPs are optional estimates and must be
  labelled incomplete if the profiler does not count attention operations.
- Keep the original literature-review files unchanged. Paper and presentation
  authoring start after the experimental evidence is checked.

Limitations remain: small dataset, controlled backgrounds, no healthy negatives,
heuristic grouping rather than verified physical-fruit IDs, one insertion site,
and no real-world robustness claim. Record actual student contributions and
AI assistance accurately.
