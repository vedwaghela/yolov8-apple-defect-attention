"""Summarize a completed run and preview fixed validation examples, never test data."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))
Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    meta = json.loads((run / "experiment.json").read_text())
    if meta["status"] != "completed":
        raise RuntimeError("Wait for training to finish before summarizing")
    with (run / "results.csv").open(newline="") as stream:
        rows = [{k.strip(): float(v) for k, v in row.items()} for row in csv.DictReader(stream)]
    from PIL import Image, ImageDraw, ImageOps
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import yaml
    from ultralytics import YOLO

    best = max(rows, key=lambda r: (r["metrics/mAP50-95(B)"], r["epoch"]))
    report = dict(purpose=meta["purpose"], epochs=len(rows), best_csv_epoch=int(best["epoch"]),
                  best_epoch_validation=best, test_evaluated=False,
                  split_manifest_sha256=meta.get("split_manifest_sha256"),
                  checkpoint_sha256=hashlib.sha256((run / "weights/best.pt").read_bytes()).hexdigest(),
                  preview_confidence=0.25,
                  preview_selection="First six entries in fixed validation list; no performance selection",
                  limitations=["One seed only; no uncertainty estimate.",
                               "Validation-selected results, not held-out test performance.",
                               "Grouping uses similarity and timestamps, not confirmed physical identities."])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    epochs = [r["epoch"] for r in rows]
    for key, label in [("metrics/mAP50(B)", "mAP50"), ("metrics/mAP50-95(B)", "mAP50-95")]:
        axes[0].plot(epochs, [r[key] for r in rows], label=label)
    for key, label in [("train/box_loss", "Training box loss"), ("val/box_loss", "Validation box loss")]:
        axes[1].plot(epochs, [r[key] for r in rows], label=label)
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Validation AP (0-1)")
    axes[1].set_ylabel("Box loss")
    variant = meta.get("variant", "baseline")
    fig.suptitle(f"YOLOv8n {variant}: validation monitoring, not final test results")
    fig.savefig(run / "learning_summary.png", dpi=160)
    plt.close(fig)
    config = yaml.safe_load(Path(meta["settings"]["data"]).read_text())
    val_file = Path(config["val"])
    if not val_file.is_file():
        raise RuntimeError("Preview requires an explicit validation list")
    examples = [Path(s) for s in val_file.read_text().splitlines()[:6]]
    model = YOLO(str(run / "weights/best.pt"))
    outputs = model.predict([str(p) for p in examples], conf=0.25, iou=0.7,
                            imgsz=meta["settings"]["imgsz"], device=meta["settings"]["device"],
                            verbose=False)
    canvas = Image.new("RGB", (1000, 380 * len(examples)), "white")
    pen = ImageDraw.Draw(canvas)
    for row, (path, result) in enumerate(zip(examples, outputs)):
        with Image.open(path) as source:
            gt = source.convert("RGB")
        painter = ImageDraw.Draw(gt)
        label = path.parents[2] / "labels" / path.parent.name / (path.stem+".txt")
        for line in label.read_text().splitlines():
            if not line.strip():
                continue
            c, x, y, w, h = map(float, line.split())
            width, height = gt.size
            box = ((x-w/2)*width, (y-h/2)*height, (x+w/2)*width, (y+h/2)*height)
            painter.rectangle(box, outline=(0, 255, 0), width=4)
            painter.text((box[0], box[1]), config["names"][int(c)], fill=(255, 255, 0))
        pred = Image.fromarray(result.plot()[..., ::-1])
        for col, im in enumerate((gt, pred)):
            thumb = ImageOps.contain(im, (480, 330))
            canvas.paste(thumb, (col*500 + (480-thumb.width)//2, row*380+25))
        pen.text((8, row*380+5), "ANNOTATIONS", fill="black")
        pen.text((508, row*380+5), "PREDICTIONS | confidence >= 0.25", fill="black")
        pen.text((8, row*380+357), path.name, fill="black")
    canvas.save(run / "validation_examples.jpg")
    (run / "summary.json").write_text(json.dumps(report, indent=2))
    text = f"""# YOLOv8n {variant} result

Purpose: {meta['purpose']}. Completed epochs: {len(rows)}. Seed: {meta['settings']['seed']}.
Best CSV epoch by validation mAP50-95: {int(best['epoch'])}.

| Metric at that epoch | Value |
| --- | ---: |
| Validation mAP50 | {best['metrics/mAP50(B)']:.4f} |
| Validation mAP50-95 | {best['metrics/mAP50-95(B)']:.4f} |
| Validation precision | {best['metrics/precision(B)']:.4f} |
| Validation recall | {best['metrics/recall(B)']:.4f} |

Scores are from the training CSV; final checkpoint revalidation can differ slightly
because checkpoint weights are serialized at reduced precision. The test set has
not been evaluated. These scores do not establish superiority over another model.
One seed is insufficient to estimate training variability.

Validation examples are the first six fixed-list images, not selected successes.
Prediction boxes use confidence 0.25; missing boxes can reflect this display threshold.
This threshold is for visualization, not the AP calculation. The learning plot
shows training progress; training and validation losses involve different
augmentation conditions and are not directly equivalent measurements.
"""
    (run / "SUMMARY.md").write_text(text)
    print(json.dumps({k: v for k, v in report.items() if k != "best_epoch_validation"}, indent=2))


if __name__ == "__main__":
    main()
