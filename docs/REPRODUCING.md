# Reproducing the study

## 1. Saved-result analysis

Follow the environment setup in the root README, then run `python scripts/setup_local.py`. This creates ignored, local working copies under `baseline/runs/`. Published records remain in `results/records/`.

Run `python baseline/evaluate_study.py --self-test` for diagnostic-matching checks and `python baseline/summarize_study.py` to regenerate aggregate tables and plots from saved observations. No GPU or trained checkpoints are needed for aggregation. Saved raw timing observations describe the original GPU, not the current machine.

## 2. Dataset preparation

From the repository root:

```powershell
git clone https://github.com/nbngochan/Surface-Subtle-Defects-Apple.git baseline/data/ssda-source
git -C baseline/data/ssda-source checkout 71fc44fa3358e5cbc0a59d574d38762aaefd676b
python baseline/prepare.py
python scripts/setup_local.py --with-data
python baseline/check_split.py
```

The helper verifies all image and prepared-label hashes and restores the exact recorded memberships and grouping. Do not run `make_split.py` to generate a new partition for this reproduction. `prepare.py` retains the upstream files, clips one out-of-bounds box in a working copy and writes labels at ten decimal places.

The published manifest replaces the original computer's root path with `${PROJECT_ROOT}`. Local setup substitutes the new root and computes a new manifest digest. Original content hashes, groups and memberships are unchanged, but the relocated manifest's byte hash is necessarily different. Historical experiment hashes describe the original run, not regenerated local configuration files.

## 3. Training

The pinned Ultralytics package supplies the `yolov8n.pt` download when the training entry point loads the missing checkpoint. Internet access is required for this first download. Keep `baseline/yolov8n.pt` identical across variants and record its hash. The private original archive retains the original weight file if exact checkpoint provenance is required.

```powershell
python baseline/test_attention.py
python baseline/train.py --variant baseline --epochs 50 --seed 42
python baseline/train.py --variant se --epochs 50 --seed 42
python baseline/train.py --variant cbam --epochs 50 --seed 42
python baseline/train.py --variant ca --epochs 50 --seed 42
```

Repeat all four commands for seeds 43 and 44. Defaults use GPU 0, 640-pixel inputs and physical batch 4. New runs receive timestamped directories. Install `uv` (included in requirements) because the unchanged training script uses it to record the environment. Do not use the historical queue scripts to orchestrate a new study: they refer to original run IDs.

## 4. Evaluation and checkpoint limitations

The original evaluator indexes the twelve historical run directories through the archived queue/status records. Trained `weights/best.pt` files are excluded. To re-evaluate those exact models, restore their matching files from the private archive into the original run directories under `baseline/runs/`, then run:

```powershell
python baseline/evaluate_study.py --split val
python baseline/evaluate_study.py --split test
python baseline/evaluate_study.py --benchmark
python baseline/summarize_study.py
```

For newly trained timestamped runs, create separate evaluation indexes pointing to the new runs; the archived indexes do not automatically select them. Never relabel new results as the archived experiments. Custom checkpoints require `baseline/attention.py` to be importable. Only load checkpoints from trusted sources.

The original training, attention and evaluation code is preserved unchanged. This publication package adds path restoration rather than changing the scientific implementation. Full GPU retraining is not part of package validation.
