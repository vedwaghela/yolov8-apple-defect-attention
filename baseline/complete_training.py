"""Resumable matched-seed queue; never evaluates the test set."""
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "runs/repeated_seeds_43_44"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(exist_ok=True)
    state_path = OUT / "status.json"
    if "--status" in sys.argv:
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        for key, item in state.get("runs", {}).items():
            candidates = sorted((ROOT / "runs").glob(key + "_*"))
            if candidates and (candidates[-1] / "results.csv").exists():
                with (candidates[-1] / "results.csv").open() as f:
                    rows = list(csv.DictReader(f))
                item["epoch"] = rows[-1]["epoch"] if rows else 0
        print(json.dumps(state, indent=2))
        return
    import msvcrt
    lock = (OUT / "queue.lock").open("a+b")
    lock.seek(0)
    if not lock.read(1):
        lock.write(b"0")
        lock.flush()
    lock.seek(0)
    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    sources = {n: digest(ROOT / n) for n in ("train.py", "attention.py", "FINAL_PROTOCOL.md")}
    reference = json.loads((ROOT / "runs/baseline_seed42_20261016T000839200430/experiment.json").read_text())
    ignore = {"seed", "project", "name"}
    expected = {k: v for k, v in reference["settings"].items() if k not in ignore}
    state = json.loads(state_path.read_text()) if state_path.exists() else {"sources": sources, "runs": {}}
    assert state["sources"] == sources, "Source/protocol changed since queue creation"

    def save():
        temp = OUT / "status.tmp"
        temp.write_text(json.dumps(state, indent=2))
        temp.replace(state_path)

    def verify(run, seed, variant):
        meta = json.loads((run / "experiment.json").read_text())
        assert meta["status"] == "completed"
        assert meta["settings"]["seed"] == seed and meta["variant"] == variant
        assert {k: v for k, v in meta["settings"].items() if k not in ignore} == expected
        assert meta["packages"] == reference["packages"]
        assert meta["split_manifest_sha256"] == reference["split_manifest_sha256"]
        with (run / "results.csv").open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 50 and int(rows[-1]["epoch"]) == 50
        assert (run / "weights/best.pt").is_file()

    state.update(status="running", test_evaluated=False)
    save()
    try:
        for seed in (43, 44):
            for variant in ("baseline", "se", "cbam", "ca"):
                assert all(digest(ROOT / n) == h for n, h in sources.items())
                key = f"{variant}_full_seed{seed}"
                candidates = sorted((ROOT / "runs").glob(key + "_*"))
                completed = [p for p in candidates if (p / "experiment.json").exists()
                             and json.loads((p / "experiment.json").read_text())["status"] == "completed"]
                assert len(completed) <= 1, f"Ambiguous completed runs: {key}"
                if completed:
                    run = completed[0]
                else:
                    if candidates:
                        raise RuntimeError(f"Incomplete run requires inspection before retry: {candidates}")
                    state["runs"][key] = {"status": "training"}
                    save()
                    with (OUT / (key + ".log")).open("w", encoding="utf-8") as log:
                        subprocess.run([sys.executable, str(ROOT / "train.py"), "--variant", variant,
                                        "--seed", str(seed), "--epochs", "50"], cwd=ROOT.parent,
                                       stdout=log, stderr=subprocess.STDOUT, check=True)
                    candidates = sorted((ROOT / "runs").glob(key + "_*"))
                    assert len(candidates) == 1
                    run = candidates[0]
                verify(run, seed, variant)
                state["runs"][key] = {"status": "completed", "run": str(run)}
                save()
        state["status"] = "completed"
        save()
    except BaseException as exc:
        state.update(status="failed", error=repr(exc))
        save()
        raise
    finally:
        lock.close()


if __name__ == "__main__":
    main()
