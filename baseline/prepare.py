"""Audit the upstream SSDA split and prepare a small pipeline test."""
import hashlib
import json
import math
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw
import yaml

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data" / "ssda-source"
OUT = ROOT / "data" / "prepared"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"source": "https://github.com/nbngochan/Surface-Subtle-Defects-Apple",
              "splits": {}, "errors": [], "clipped_boxes": [], "cross_split_exact_duplicates": [],
              "limitations": ["No independent test set configured.",
                              "Physical fruit identities and near-duplicate leakage not verified.",
                              "Class names inferred by matching HAFREE Table 5 counts."],
              "licence": {"name": "CC BY 4.0", "release": "https://zenodo.org/records/10957715",
                          "metadata": "https://api.datacite.org/dois/10.5281/zenodo.10957715",
                          "revision": "71fc44fa3358e5cbc0a59d574d38762aaefd676b"}}
    seen, classes, samples = {}, set(), {}
    for split in ("train", "val"):
        images = sorted(p for p in (SOURCE / "images" / split).rglob("*")
                        if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if not images:
            raise RuntimeError(f"No images in {SOURCE / 'images' / split}")
        counts, boxes_by_image = Counter(), {}
        for path in images:
            with Image.open(path) as im:
                im.verify()
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest in seen and seen[digest][0] != split:
                report["cross_split_exact_duplicates"].append([seen[digest][1], str(path)])
            seen[digest] = (split, str(path))
            label = SOURCE / "labels" / split / path.relative_to(SOURCE / "images" / split).with_suffix(".txt")
            boxes = []
            if not label.exists():
                report["errors"].append(f"Missing label: {label}")
                continue
            for line_no, line in enumerate(label.read_text().splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    values = list(map(float, line.split()))
                    if len(values) != 5 or not all(math.isfinite(v) for v in values):
                        raise ValueError("Expected five finite values")
                    c, x, y, w, h = values
                    if c < 0 or not c.is_integer() or not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                        raise ValueError("Invalid class or normalized box")
                    if min(x-w/2, y-h/2) < 0 or max(x+w/2, y+h/2) > 1:
                        left, top = max(0, x-w/2), max(0, y-h/2)
                        right, bottom = min(1, x+w/2), min(1, y+h/2)
                        corrected = [c, (left+right)/2, (top+bottom)/2, right-left, bottom-top]
                        report["clipped_boxes"].append({"file": str(label), "line": line_no,
                                                        "original": values, "corrected": corrected})
                        values = corrected
                    classes.add(int(c))
                    counts[int(c)] += 1
                    boxes.append(values)
                except ValueError as exc:
                    report["errors"].append(f"{label}:{line_no}: {exc}")
            boxes_by_image[path] = boxes
            image_out = OUT / "images" / split / path.name
            label_out = OUT / "labels" / split / label.name
            image_out.parent.mkdir(parents=True, exist_ok=True)
            label_out.parent.mkdir(parents=True, exist_ok=True)
            if not image_out.exists():
                shutil.copy2(path, image_out)
            label_out.write_text("\n".join(f"{int(b[0])} " + " ".join(f"{v:.10f}" for v in b[1:]) for b in boxes)+"\n")
        samples[split] = [OUT / "images" / split / p.name for p in images]
        report["splits"][split] = {"images": len(images), "boxes_by_class": dict(counts),
                                   "empty_labels": sum(not b for b in boxes_by_image.values())}
        # Preserve source annotations; the overlay is for human inspection only.
        canvas = Image.new("RGB", (960, 640), "white")
        draw = ImageDraw.Draw(canvas)
        for i, path in enumerate(images[:6]):
            with Image.open(path) as im:
                im = im.convert("RGB").resize((320, 280))
            pen = ImageDraw.Draw(im)
            for c, x, y, w, h in boxes_by_image.get(path, []):
                pen.rectangle(((x-w/2)*320, (y-h/2)*280, (x+w/2)*320, (y+h/2)*280), outline="lime", width=2)
                pen.text((x*320, y*280), f"class {int(c)}", fill="yellow")
            px, py = (i % 3)*320, (i // 3)*320
            canvas.paste(im, (px, py))
            draw.text((px+3, py+284), path.stem[:35], fill="black")
        canvas.save(OUT / f"{split}_labels.jpg")
    if classes != set(range(max(classes, default=-1)+1)) or not classes:
        report["errors"].append("Class IDs are empty or not contiguous from zero")
    (OUT / "audit.json").write_text(json.dumps(report, indent=2))
    if report["errors"] or report["cross_split_exact_duplicates"]:
        raise RuntimeError("Dataset audit failed; inspect data/prepared/audit.json")
    config = {"path": OUT.as_posix(), "train": "images/train", "val": "images/val",
              "names": {0: "scratch", 1: "pest_damage"}}
    (OUT / "ssda.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    for split, paths in samples.items():
        (OUT / f"smoke_{split}.txt").write_text("\n".join(p.as_posix() for p in paths[:8])+"\n")
    config.update(train=(OUT / "smoke_train.txt").as_posix(), val=(OUT / "smoke_val.txt").as_posix())
    (OUT / "smoke.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
