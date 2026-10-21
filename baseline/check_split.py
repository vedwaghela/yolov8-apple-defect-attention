"""Independent integrity checks for the frozen experimental split."""
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    folder = ROOT / "data/grouped_v1"
    raw = (folder / "manifest.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == (folder / "manifest.sha256").read_text().strip()
    manifest = json.loads(raw)
    records = manifest["records"]
    assert len(records) == 710
    assert len({r["file"] for r in records}) == 710
    memberships, digest_splits = defaultdict(set), defaultdict(set)
    for r in records:
        memberships[r["group"]].add(r["split"])
        digest_splits[r["decoded_sha256"]].add(r["split"])
    assert all(len(s) == 1 for s in memberships.values()), "Group spans splits"
    assert all(len(s) == 1 for s in digest_splits.values()), "Decoded duplicate spans splits"
    for edge in manifest["edges"]:
        assert records[edge["a"]]["split"] == records[edge["b"]]["split"]
    for split, expected in manifest["summary"].items():
        selected = [r for r in records if r["split"] == split]
        listed = (folder / f"{split}.txt").read_text().splitlines()
        assert listed == [r["file"] for r in selected]
        counts = Counter()
        for r in selected:
            path = Path(r["file"])
            label = path.parents[2] / "labels" / path.parent.name / (path.stem+".txt")
            assert hashlib.sha256(path.read_bytes()).hexdigest() == r["sha256"]
            assert hashlib.sha256(label.read_bytes()).hexdigest() == r["label_sha256"]
            counts.update(int(line.split()[0]) for line in label.read_text().splitlines() if line.strip())
        assert len(selected) == expected["images"]
        assert counts[0] == expected["scratch_boxes"]
        assert counts[1] == expected["pest_damage_boxes"]
    previously_used = set()
    for s in ("train", "val"):
        previously_used.update((ROOT / f"data/prepared/smoke_{s}.txt").read_text().splitlines())
    assert len(previously_used) == 16
    assert all(r["split"] == "train" for r in records if r["file"] in previously_used)
    print("PASS: hashes, image uniqueness, class counts, group isolation, lists, and prior-smoke exclusion from validation/test.")


if __name__ == "__main__":
    main()
