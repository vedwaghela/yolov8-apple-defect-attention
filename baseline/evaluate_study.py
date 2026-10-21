"""Fixed-protocol evaluation, diagnostic matching and reproducible GPU timing."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))
VARIANTS = ("baseline", "se", "cbam", "ca")


def write_json(path, value):
    import numpy as np
    def convert(x):
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, np.ndarray):
            return x.tolist()
        raise TypeError(type(x).__name__)
    path.write_text(json.dumps(value, indent=2, default=convert))


def collect_runs():
    old = json.loads((ROOT / "runs/attention_queue_20261017T065900/comparison.json").read_text())
    runs = {(r["model"], 42): Path(r["run"]) for r in old}
    state = json.loads((ROOT / "runs/repeated_seeds_43_44/status.json").read_text())
    if state["status"] != "completed":
        raise RuntimeError("All matched-seed training must finish first")
    for seed in (43, 44):
        for variant in VARIANTS:
            runs[variant, seed] = Path(state["runs"][f"{variant}_full_seed{seed}"]["run"])
    reference = json.loads((runs["baseline", 42] / "experiment.json").read_text())
    ignore = {"seed", "project", "name"}
    expected = {k: v for k, v in reference["settings"].items() if k not in ignore}
    for (variant, seed), run in runs.items():
        meta = json.loads((run / "experiment.json").read_text())
        assert meta["status"] == "completed" and meta["settings"]["seed"] == seed
        assert meta.get("variant", "baseline") == variant
        assert meta["split_manifest_sha256"] == reference["split_manifest_sha256"]
        assert meta["packages"] == reference["packages"]
        assert {k: v for k, v in meta["settings"].items() if k not in ignore} == expected
        with (run / "results.csv").open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 50 and int(rows[-1]["epoch"]) == 50
    return runs


def diagnostic_match(gt, pred, gt_cls, pred_cls):
    """Maximum-cardinality geometric matching, then classify mismatches (not AP)."""
    import numpy as np
    import torch
    from scipy.optimize import linear_sum_assignment
    from ultralytics.utils.metrics import box_iou
    if not len(gt) or not len(pred):
        return dict(correct=0, wrong_class=0, missed=len(gt), extra=len(pred), matched_ious=[])
    iou = box_iou(torch.as_tensor(gt).float(), torch.as_tensor(pred).float()).numpy()
    valid = iou >= 0.5
    score = valid * (min(len(gt), len(pred)) + 1 + iou)
    rows, cols = linear_sum_assignment(-score)
    pairs = [(r, c) for r, c in zip(rows, cols) if valid[r, c]]
    correct = sum(int(gt_cls[r]) == int(pred_cls[c]) for r, c in pairs)
    return dict(correct=correct, wrong_class=len(pairs)-correct,
                missed=len(gt)-len(pairs), extra=len(pred)-len(pairs),
                matched_ious=[float(iou[r, c]) for r, c in pairs])


def self_test():
    import numpy as np
    boxes = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=float)
    a = diagnostic_match(boxes, boxes, [0, 1], [0, 1])
    assert a == dict(correct=2, wrong_class=0, missed=0, extra=0, matched_ious=[1., 1.])
    a = diagnostic_match(boxes, boxes[:1], [0, 1], [1])
    assert a["wrong_class"] == 1 and a["missed"] == 1
    a = diagnostic_match(boxes[:1], boxes, [0], [0, 1])
    assert a["correct"] == 1 and a["extra"] == 1
    assert diagnostic_match(boxes, [], [0, 1], [])["missed"] == 2
    assert diagnostic_match([], boxes, [], [0, 1])["extra"] == 2
    print("PASS: diagnostic matching, class errors, unmatched boxes and empty inputs")


def evaluate(runs, split, out):
    import numpy as np
    import yaml
    from PIL import Image
    from ultralytics import YOLO
    config = yaml.safe_load((ROOT / "data/grouped_v1/ssda.yaml").read_text())
    paths = Path(config[split]).read_text().splitlines()
    result_rows = []
    for (variant, seed), run in runs.items():
        key = f"{variant}_seed{seed}"
        target = out / split / key
        target.mkdir(parents=True, exist_ok=True)
        checkpoint = run / "weights/best.pt"
        sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        record_path = target / "evaluation.json"
        if record_path.exists():
            record = json.loads(record_path.read_text())
            assert record["checkpoint_sha256"] == sha
            result_rows.append(record)
            continue
        model = YOLO(str(checkpoint))
        metrics = model.val(data=str(ROOT / "data/grouped_v1/ssda.yaml"), split=split,
                            imgsz=640, batch=4, device="0", workers=0, half=False,
                            conf=0.001, iou=0.7, max_det=300, augment=False,
                            plots=True, project=str(target), name="metrics", exist_ok=True,
                            verbose=False)
        record = dict(model=variant, seed=seed, run=str(run), split=split,
                      checkpoint_sha256=sha, map50=float(metrics.box.map50),
                      map50_95=float(metrics.box.map), per_class=metrics.summary())
        # Fixed-threshold diagnostics are separate from the AP evaluation above.
        diagnostics = []
        predictions = model.predict(paths, stream=True, batch=4, imgsz=640, device="0",
                                    half=False, conf=0.25, iou=0.7, max_det=300,
                                    augment=False, verbose=False)
        for index, result in enumerate(predictions):
            path = Path(result.path)
            label = path.parents[2] / "labels" / path.parent.name / (path.stem + ".txt")
            labels = np.array([list(map(float, line.split())) for line in label.read_text().splitlines()
                               if line.strip()], dtype=float).reshape(-1, 5)
            h, w = result.orig_shape
            centers = labels[:, 1:3] * [w, h]
            sizes = labels[:, 3:5] * [w, h]
            gt = np.concatenate((centers-sizes/2, centers+sizes/2), axis=1)
            counts = diagnostic_match(gt, result.boxes.xyxy.cpu().numpy(), labels[:, 0],
                                      result.boxes.cls.cpu().numpy())
            diagnostics.append(dict(image=path.name, **counts))
            if seed == 42 and index < 6:
                Image.fromarray(result.plot()[..., ::-1]).save(target / f"example_{index+1}.jpg")
        write_json(target / "diagnostics.json", diagnostics)
        record["diagnostic_totals"] = {k: sum(r[k] for r in diagnostics)
                                       for k in ("correct", "wrong_class", "missed", "extra")}
        record["diagnostic_definition"] = "conf=.25, IoU=.5, class-agnostic geometric assignment; not AP"
        write_json(record_path, record)
        result_rows.append(record)
        print(f"Evaluated {split}: {key}", flush=True)
    write_json(out / (split + "_results.json"), result_rows)


def benchmark(runs, out):
    import time
    import numpy as np
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    x = torch.zeros((1, 3, 640, 640), device="cuda", dtype=torch.float32)
    records = []
    # Seed 42 is fixed for architecture timing, not selected for its score.
    for block in range(3):
        order = VARIANTS[block:] + VARIANTS[:block]
        for variant in order:
            model = YOLO(str(runs[variant, 42] / "weights/best.pt")).model.float().cuda().eval()
            model.fuse(verbose=False)
            with torch.inference_mode():
                for _ in range(50):
                    model(x)
                torch.cuda.synchronize()
                times = []
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                for _ in range(200):
                    start.record()
                    model(x)
                    end.record()
                    end.synchronize()
                    times.append(start.elapsed_time(end))
            records.append(dict(model=variant, block=block, seed=42, milliseconds=times,
                                median_ms=float(np.median(times)), p95_ms=float(np.percentile(times, 95))))
            write_json(out / "timing.json", dict(gpu=torch.cuda.get_device_name(0),
                       protocol="FP32 fused forward-only CUDA events, batch 1, 640 square, 50 warmup, 200 samples/block",
                       blocks=records))
            del model
            torch.cuda.empty_cache()
            time.sleep(2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--split", choices=("val", "test"))
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    runs = collect_runs()
    out = ROOT / "runs/final_evaluation"
    out.mkdir(exist_ok=True)
    snapshot = dict(protocol_sha256=hashlib.sha256((ROOT / "FINAL_PROTOCOL.md").read_bytes()).hexdigest(),
                    evaluator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    checkpoints={f"{v}_{s}": hashlib.sha256((p / "weights/best.pt").read_bytes()).hexdigest()
                                 for (v, s), p in runs.items()})
    snapshot_path = out / "frozen_evaluation.json"
    if snapshot_path.exists():
        assert json.loads(snapshot_path.read_text()) == snapshot, "Evaluation code or checkpoints changed"
    else:
        write_json(snapshot_path, snapshot)
    if args.split:
        evaluate(runs, args.split, out)
    if args.benchmark:
        if (out / "timing.json").exists():
            raise RuntimeError("Timing already exists; inspect rather than silently overwrite")
        benchmark(runs, out)


if __name__ == "__main__":
    main()
