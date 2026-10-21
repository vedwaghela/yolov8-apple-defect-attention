"""Create a reproducible, heuristic-grouped split without using model scores."""
import hashlib
import json
import random
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy.fft import dctn
import yaml

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data/prepared"
OUT = ROOT / "data/grouped_v1"


def main():
    if (OUT / "manifest.json").exists():
        raise RuntimeError("Split already frozen. Preserve it; use a new version for changes.")
    paths = sorted((DATA / "images").glob("*/*.jpg"))
    if len(paths) != 710:
        raise RuntimeError(f"Expected 710 images, found {len(paths)}")
    records, hashes, pixels = [], [], []
    for p in paths:
        label = DATA / "labels" / p.parent.name / (p.stem + ".txt")
        counts = Counter(int(row.split()[0]) for row in label.read_text().splitlines() if row.strip())
        with Image.open(p) as im:
            rgb = np.asarray(im.convert("RGB").resize((32, 32)), dtype=np.float32) / 255
            gray = np.asarray(im.convert("L").resize((32, 32)), dtype=float)
            coeff = dctn(gray, type=2, norm="ortho")[:8, :8].ravel()[1:]
            hashes.append(coeff > np.median(coeff))
            pixels.append(rgb)
            size = list(im.size)
            decoded = hashlib.sha256(im.convert("RGB").tobytes()).hexdigest()
        camera, date, time, ms = p.stem.split("_")
        stamp = datetime.strptime(date + time + ms, "%Y%m%d%H%M%S%f")
        records.append(dict(file=p.as_posix(), original_split=p.parent.name,
                            sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                            label_sha256=hashlib.sha256(label.read_bytes()).hexdigest(),
                            decoded_sha256=decoded, size=size, camera=camera,
                            timestamp=stamp.isoformat(), counts=dict(counts)))
    parents = list(range(len(paths)))

    def find(x):
        while parents[x] != x:
            parents[x] = parents[parents[x]]
            x = parents[x]
        return x

    def union(a, b):
        parents[find(b)] = find(a)

    edges = []
    for i in range(len(paths)):
        for j in range(i):
            distance = int(np.count_nonzero(hashes[i] != hashes[j]))
            rmse = float(np.sqrt(np.mean((pixels[i]-pixels[j])**2)))
            exact = records[i]["decoded_sha256"] == records[j]["decoded_sha256"] and records[i]["size"] == records[j]["size"]
            visual = distance <= 4 and rmse <= 0.08
            delta = abs((datetime.fromisoformat(records[i]["timestamp"]) - datetime.fromisoformat(records[j]["timestamp"])).total_seconds())
            temporal = records[i]["camera"] == records[j]["camera"] and delta <= 2
            if exact or visual or temporal:
                union(i, j)
                edges.append(dict(a=j, b=i, exact=exact, visual=visual, temporal=temporal,
                                  hash_distance=distance, rmse=rmse))
    groups = defaultdict(list)
    for i in range(len(paths)):
        groups[find(i)].append(i)
    exposed = set()
    for split in ("train", "val"):
        exposed.update((DATA / f"smoke_{split}.txt").read_text().splitlines())
    forced = {g for g, ids in groups.items() if any(paths[i].as_posix() in exposed for i in ids)}
    # Match image and class-box proportions, keeping every connected group intact.
    features = {g: np.array([len(ids), sum(records[i]["counts"].get(0, 0) for i in ids),
                            sum(records[i]["counts"].get(1, 0) for i in ids)], dtype=float)
                for g, ids in groups.items()}
    total = sum(features.values())
    targets = np.array([0.70, 0.15, 0.15])[:, None] * total
    totals = np.zeros((3, 3))
    assigned = {}
    for g in sorted(forced):
        assigned[g] = 0
        totals[0] += features[g]
    remaining = [g for g in groups if g not in forced]
    random.Random(20261016).shuffle(remaining)
    remaining.sort(key=lambda g: -len(groups[g]))
    for g in remaining:
        costs = []
        for s in range(3):
            trial = totals.copy()
            trial[s] += features[g]
            costs.append(float(np.sum(((trial-targets)/total)**2)))
        s = int(np.argmin(costs))
        assigned[g] = s
        totals[s] += features[g]
    names = ["train", "val", "test"]
    OUT.mkdir(parents=True, exist_ok=True)
    for g, ids in groups.items():
        for i in ids:
            records[i].update(group=f"g{g:04}", split=names[assigned[g]])
    assert all(records[e["a"]]["split"] == records[e["b"]]["split"] for e in edges)
    assert all(r["split"] == "train" for r in records if r["file"] in exposed)
    assert len({r["file"] for r in records}) == len(records)
    assert np.all(totals > 0), "Each split must contain both classes"
    summary = {s: {"images": int(totals[i, 0]), "scratch_boxes": int(totals[i, 1]),
                   "pest_damage_boxes": int(totals[i, 2])} for i, s in enumerate(names)}
    for s in names:
        (OUT / f"{s}.txt").write_text("\n".join(r["file"] for r in records if r["split"] == s)+"\n")
    config = dict(path=DATA.as_posix(), **{s: (OUT / f"{s}.txt").as_posix() for s in names},
                  names={0: "scratch", 1: "pest_damage"})
    (OUT / "ssda.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    manifest = dict(seed=20261016, target_ratios=[0.70, 0.15, 0.15], summary=summary,
                    group_count=len(groups), largest_group=max(map(len, groups.values())),
                    grouping="Connected components: identical decoded pixels; OR 63-bit pHash distance <=4 and RGB thumbnail RMSE <=0.08; OR same-camera captures <=2 seconds apart.",
                    class_mapping="Inferred from exact agreement with HAFREE preprint Table 5: train 500/716, validation 124/172.",
                    limitations=["Heuristic groups are not verified physical-fruit identities.",
                                 "Different views or re-encoded near-duplicates may remain unlinked.",
                                 "Split differs from published benchmark; scores not directly comparable.",
                                 "No normal-fruit negatives in this dataset."],
                    prior_smoke_images_forced_into_train=len(exposed),
                    upstream_cross_split_edges=sum(records[e["a"]]["original_split"] != records[e["b"]]["original_split"] for e in edges),
                    edges=edges, records=records)
    raw = json.dumps(manifest, indent=2)
    (OUT / "manifest.json").write_text(raw)
    digest = hashlib.sha256((OUT / "manifest.json").read_bytes()).hexdigest()
    (OUT / "manifest.sha256").write_text(digest+"\n")
    # Review the most similar pairs from the original training/validation boundary.
    candidates = sorted([e for e in edges if records[e["a"]]["original_split"] != records[e["b"]]["original_split"]], key=lambda e: e["rmse"])[:6]
    if candidates:
        canvas = Image.new("RGB", (800, 260*len(candidates)), "white")
        pen = ImageDraw.Draw(canvas)
        for row, edge in enumerate(candidates):
            for col, idx in enumerate((edge["a"], edge["b"])):
                with Image.open(paths[idx]) as im:
                    thumb = ImageOps.contain(im.convert("RGB"), (390, 215))
                canvas.paste(thumb, (col*400, row*260))
                pen.text((col*400, row*260+216), paths[idx].name, fill="black")
                pen.text((col*400, row*260+234), f"original {records[idx]['original_split']}; new {records[idx]['split']}", fill="black")
        canvas.save(OUT / "related_pairs.jpg")
    print(json.dumps({"summary": summary, "groups": len(groups), "largest_group": manifest["largest_group"],
                      "upstream_cross_split_edges": manifest["upstream_cross_split_edges"], "manifest_sha256": digest}, indent=2))


if __name__ == "__main__":
    main()
