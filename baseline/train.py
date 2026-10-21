"""Run either a pipeline smoke test or the development baseline (not a final test)."""
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))
Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--pilot", action="store_true", help="One epoch at research resolution; not a final result")
    parser.add_argument("--data", type=Path, default=ROOT / "data/grouped_v1/ssda.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--device", default="0")
    parser.add_argument("--variant", choices=("baseline", "se", "cbam", "ca"), default="baseline")
    args = parser.parse_args()
    import torch
    from ultralytics import YOLO

    if args.device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use --device cpu explicitly for a small smoke test.")
    if args.smoke and args.pilot:
        parser.error("Choose smoke or pilot, not both")
    data = ROOT / "data/prepared/smoke.yaml" if args.smoke else args.data.resolve()
    if not data.exists():
        raise RuntimeError("Run prepare.py and make_split.py first")
    manifest_hash = None
    if not args.smoke:
        manifest_path = data.parent / "manifest.json"
        manifest_raw = manifest_path.read_bytes()
        manifest_hash = hashlib.sha256(manifest_raw).hexdigest()
        if manifest_hash != (data.parent / "manifest.sha256").read_text().strip():
            raise RuntimeError("Frozen split manifest has changed")
        import yaml
        config = yaml.safe_load(data.read_text())
        records = json.loads(manifest_raw)["records"]
        for split in ("train", "val", "test"):
            listed = Path(config[split]).read_text().splitlines()
            if listed != [r["file"] for r in records if r["split"] == split]:
                raise RuntimeError(f"{split} list no longer matches frozen manifest")
        for record in records:
            path = Path(record["file"])
            label = path.parents[2] / "labels" / path.parent.name / (path.stem + ".txt")
            if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
                raise RuntimeError(f"Image changed: {path}")
            if hashlib.sha256(label.read_bytes()).hexdigest() != record["label_sha256"]:
                raise RuntimeError(f"Annotation changed: {label}")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    phase = 'smoke' if args.smoke else 'pilot' if args.pilot else 'full'
    name = f"{args.variant}_{phase}_seed{args.seed}_{stamp}"
    run = ROOT / "runs" / name
    run.mkdir(parents=True)
    settings = dict(data=str(data), epochs=1 if args.smoke or args.pilot else args.epochs,
                    imgsz=320 if args.smoke else 640, batch=2 if args.smoke else 4,
                    device=args.device, workers=0, seed=args.seed, deterministic=True,
                    optimizer="AdamW", lr0=0.001, lrf=0.01, weight_decay=0.0005,
                    patience=0, amp=False, cache=False, close_mosaic=0 if args.smoke or args.pilot else 10,
                    project=str(ROOT / "runs"), name=name, exist_ok=True, plots=True)
    metadata = {"status": "started", "purpose": "pipeline_check" if args.smoke or args.pilot else "comparison",
                "variant": args.variant, "attention_layer": None if args.variant == "baseline" else 6,
                "split_manifest_sha256": manifest_hash,
                "settings": settings, "python": platform.python_version(),
                "packages": {p: importlib.metadata.version(p) for p in ("torch", "torchvision", "ultralytics")},
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "data_config_sha256": hashlib.sha256(data.read_bytes()).hexdigest()}
    (run / "experiment.json").write_text(json.dumps(metadata, indent=2))
    for source in (Path(__file__), ROOT / "attention.py"):
        shutil.copy2(source, run / source.name)
    freeze = subprocess.check_output(["uv", "--cache-dir", str(ROOT / ".uv-cache"), "pip", "freeze", "--python", __import__("sys").executable], text=True)
    (run / "environment.txt").write_text(freeze)
    model = YOLO(str(ROOT / "yolov8n.pt"))
    if args.variant == "baseline":
        model.train(**settings)
    else:
        from attention import TRAINERS
        model.train(trainer=TRAINERS[args.variant], **settings)
    metadata["status"] = "completed"
    metadata["parameters"] = sum(p.numel() for p in model.model.parameters())
    if args.variant != "baseline":
        metadata["attention_parameters"] = sum(p.numel() for p in model.model.model[6].attention.parameters())
    metadata["note"] = "Validation only. No held-out test evaluation. Smoke scores are not research results."
    (run / "experiment.json").write_text(json.dumps(metadata, indent=2))
    print(f"Completed: {run}")


if __name__ == "__main__":
    main()
