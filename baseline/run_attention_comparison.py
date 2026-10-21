"""Run the three fixed-budget attention experiments sequentially, with durable logs."""
import argparse
import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASELINE = ROOT / "runs/baseline_seed42_20261016T000839200430"


def write_status(folder, state):
    state["updated_utc"] = datetime.now(timezone.utc).isoformat()
    temporary = folder / "status.tmp"
    temporary.write_text(json.dumps(state, indent=2))
    temporary.replace(folder / "status.json")


def compare(folder, runs):
    reference = json.loads((BASELINE / "experiment.json").read_text())
    ignored = {"project", "name"}
    expected = {k: v for k, v in reference["settings"].items() if k not in ignored}
    results = []
    for name, run in {"baseline": BASELINE, **runs}.items():
        meta = json.loads((run / "experiment.json").read_text())
        assert meta["status"] == "completed"
        assert meta["split_manifest_sha256"] == reference["split_manifest_sha256"]
        assert meta["packages"] == reference["packages"]
        assert {k: v for k, v in meta["settings"].items() if k not in ignored} == expected
        with (run / "results.csv").open(newline="") as f:
            rows = [{k.strip(): float(v) for k, v in row.items()} for row in csv.DictReader(f)]
        assert len(rows) == 50 and rows[-1]["epoch"] == 50
        best = max(rows, key=lambda r: (r["metrics/mAP50-95(B)"], r["epoch"]))
        results.append(dict(model=name, run=str(run), best_epoch=int(best["epoch"]),
                            map50=best["metrics/mAP50(B)"], map50_95=best["metrics/mAP50-95(B)"],
                            precision=best["metrics/precision(B)"], recall=best["metrics/recall(B)"],
                            parameters=meta["parameters"], training_seconds=rows[-1]["time"]))
    (folder / "comparison.json").write_text(json.dumps(results, indent=2))
    with (folder / "comparison.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    lines = ["# Attention comparison: seed 42, validation only", "",
             "All four runs completed 50 epochs with matching settings and split hashes.",
             "Each row uses that model's best validation mAP50-95 epoch from results.csv.", "",
             "| Model | Best epoch | mAP50 | mAP50-95 | Parameters |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for row in results:
        lines.append(f"| {row['model']} | {row['best_epoch']} | {row['map50']:.4f} | {row['map50_95']:.4f} | {row['parameters']:,} |")
    lines.extend(["", "The held-out test set was not evaluated. These are single-seed, validation-selected",
                  "results, not estimates of statistical significance or final generalization.",
                  "Runtime depends on laptop conditions; training duration is not an inference",
                  "latency benchmark. Repeat seeds and controlled timing are still required."])
    (folder / "COMPARISON.md").write_text("\n".join(lines)+"\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    if args.status:
        folders = sorted((ROOT / "runs").glob("attention_queue_*"))
        if not folders:
            print("No attention queue exists.")
            return
        folder = folders[-1]
        state = json.loads((folder / "status.json").read_text())
        for name, item in state["models"].items():
            candidates = sorted((ROOT / "runs").glob(f"{name}_full_seed42_*"))
            if candidates:
                csv_file = candidates[-1] / "results.csv"
                if csv_file.exists():
                    with csv_file.open(newline="") as f:
                        rows = list(csv.DictReader(f))
                    if rows:
                        item["last_recorded_epoch"] = rows[-1]["epoch"]
                        item["latest_validation_map50_95"] = rows[-1]["metrics/mAP50-95(B)"]
        print(json.dumps(state, indent=2))
        return
    folder = ROOT / "runs" / ("attention_queue_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S"))
    folder.mkdir()
    state = dict(status="running", test_evaluated=False, seed=42, epochs=50,
                 models={v: {"status": "pending"} for v in ("se", "cbam", "ca")},
                 source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in (ROOT / "train.py", ROOT / "attention.py")})
    write_status(folder, state)
    runs = {}
    try:
        for variant, item in state["models"].items():
            for name, digest in state["source_sha256"].items():
                assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, "Code changed during comparison"
            before = set((ROOT / "runs").glob(f"{variant}_full_seed42_*"))
            item["status"] = "training"
            write_status(folder, state)
            with (folder / f"{variant}.log").open("w", encoding="utf-8") as log:
                subprocess.run([sys.executable, str(ROOT / "train.py"), "--variant", variant,
                                "--epochs", "50", "--seed", "42"], cwd=ROOT.parent,
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            created = set((ROOT / "runs").glob(f"{variant}_full_seed42_*")) - before
            assert len(created) == 1, "Expected exactly one new training run"
            run = created.pop()
            runs[variant] = run
            item.update(status="completed", run=str(run))
            write_status(folder, state)
            with (folder / f"{variant}_summary.log").open("w", encoding="utf-8") as log:
                result = subprocess.run([sys.executable, str(ROOT / "summarize_run.py"), str(run)],
                                        cwd=ROOT.parent, stdout=log, stderr=subprocess.STDOUT)
            item["summary_status"] = "completed" if result.returncode == 0 else "failed"
            write_status(folder, state)
        compare(folder, runs)
        state["status"] = "completed"
        write_status(folder, state)
        print(f"Completed comparison: {folder}")
    except BaseException as exc:
        state.update(status="failed", error=repr(exc))
        write_status(folder, state)
        raise


if __name__ == "__main__":
    main()
