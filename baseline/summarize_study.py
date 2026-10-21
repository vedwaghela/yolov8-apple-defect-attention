"""Produce analysis artifacts from completed experiments, not a course report."""
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from evaluate_study import ROOT, VARIANTS, collect_runs, write_json


def main():
    runs = collect_runs()
    out = ROOT / "runs/final_evaluation"
    results = {s: json.loads((out / f"{s}_results.json").read_text()) for s in ("val", "test")}
    for split, rows in results.items():
        assert len(rows) == 12
        assert {(r["model"], r["seed"]) for r in rows} == set(runs)
        assert all(r["split"] == split for r in rows)
    timing = json.loads((out / "timing.json").read_text())
    assert len(timing["blocks"]) == 12
    names = ["Baseline", "SE", "CBAM", "Coordinate Attention"]
    colors = ["#4b5563", "#16806a", "#ba405c", "#327ac2"]
    summary = []
    lines = ["# Completed experimental analysis", "",
             "Seeds 42, 43 and 44; 50 epochs each. Each checkpoint was selected on validation",
             "mAP50-95 before test evaluation. Values below are mean +/- sample standard deviation",
             "across three seeds, in percentage points. Standard deviation is not a confidence interval.", "",
             "| Model | Validation mAP50-95 | Test mAP50 | Test mAP50-95 | Parameters |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for variant, name in zip(VARIANTS, names):
        row = dict(model=variant)
        for split, records in results.items():
            selected = [r for r in records if r["model"] == variant]
            for key in ("map50", "map50_95"):
                values = np.array([r[key] for r in selected]) * 100
                row[f"{split}_{key}_mean"] = float(values.mean())
                row[f"{split}_{key}_sd"] = float(values.std(ddof=1))
        meta = json.loads((runs[variant, 42] / "experiment.json").read_text())
        row["parameters"] = meta["parameters"]
        blocks = [b for b in timing["blocks"] if b["model"] == variant]
        samples = np.concatenate([b["milliseconds"] for b in blocks])
        row["forward_median_ms"] = float(np.median(samples))
        row["forward_p95_ms"] = float(np.percentile(samples, 95))
        row["block_median_sd_ms"] = float(np.std([b["median_ms"] for b in blocks], ddof=1))
        summary.append(row)
        lines.append(f"| {name} | {row['val_map50_95_mean']:.2f} +/- {row['val_map50_95_sd']:.2f} | "
                     f"{row['test_map50_mean']:.2f} +/- {row['test_map50_sd']:.2f} | "
                     f"{row['test_map50_95_mean']:.2f} +/- {row['test_map50_95_sd']:.2f} | {row['parameters']:,} |")
    lines += ["", "Validation here is re-evaluation of serialized checkpoints, not training CSV maxima.",
              "Small differences from earlier CSV summaries are expected.", "", "## Per-class test AP", "",
              "| Model | Class | AP50 | AP50-95 |", "| --- | --- | ---: | ---: |"]
    for variant, name in zip(VARIANTS, names):
        selected = [r for r in results["test"] if r["model"] == variant]
        for cls in ("scratch", "pest_damage"):
            cls_rows = [next(p for p in r["per_class"] if p["Class"] == cls) for r in selected]
            ap50 = np.array([p["mAP50"] for p in cls_rows]) * 100
            ap = np.array([p["mAP50-95"] for p in cls_rows]) * 100
            lines.append(f"| {name} | {cls} | {ap50.mean():.2f} +/- {ap50.std(ddof=1):.2f} | "
                         f"{ap.mean():.2f} +/- {ap.std(ddof=1):.2f} |")
    lines += ["", "## Forward-only GPU timing", "",
              "RTX 2050, FP32, fused model, batch 1, 640x640, seed-42 checkpoints fixed in advance.",
              "Three rotated-order blocks, 50 warm-ups and 200 CUDA-event samples per block.",
              "Excludes file I/O, preprocessing, host transfers and NMS. Not application FPS.", "",
              "| Model | Median ms | p95 ms | SD of block medians ms |",
              "| --- | ---: | ---: | ---: |"]
    for row, name in zip(summary, names):
        lines.append(f"| {name} | {row['forward_median_ms']:.3f} | {row['forward_p95_ms']:.3f} | "
                     f"{row['block_median_sd_ms']:.3f} |")
    lines += ["", "## Fixed-threshold test diagnostics", "",
              "Mean counts per seed over all 104 test images. Confidence >=0.25, geometric IoU >=0.5.",
              "One-to-one geometry matching ignores class first; wrong-class pairs are separate.",
              "These counts are not AP or standard class-aware precision/recall.", "",
              "| Model | Correct | Wrong class | Unmatched annotations | Unmatched predictions |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for variant, name in zip(VARIANTS, names):
        selected = [r["diagnostic_totals"] for r in results["test"] if r["model"] == variant]
        values = [np.mean([r[k] for r in selected]) for k in ("correct", "wrong_class", "missed", "extra")]
        lines.append(f"| {name} | " + " | ".join(f"{v:.1f}" for v in values) + " |")
    lines += ["", "## Paired seed differences", "",
              "Test mAP50-95 differences relative to baseline (percentage points); descriptive only.",
              "Three training seeds on one split do not establish statistical significance.", "",
              "| Model | Seed 42 | Seed 43 | Seed 44 |", "| --- | ---: | ---: | ---: |"]
    keyed = {(r["model"], r["seed"]): r for r in results["test"]}
    for variant, name in zip(VARIANTS[1:], names[1:]):
        diffs = [(keyed[variant, s]["map50_95"]-keyed["baseline", s]["map50_95"])*100 for s in (42, 43, 44)]
        lines.append(f"| {name} | " + " | ".join(f"{v:+.2f}" for v in diffs) + " |")
    lines += ["", "## Interpretation limits", "",
              "The split is fixed and heuristic groups are isolated, but physical apple identities are unknown.",
              "Small controlled-background data, no healthy negatives, one attention location, three seeds",
              "and one fixed budget limit generalization. No illumination/blur/orchard robustness claim is made.",
              "Unmatched predictions may reflect annotation omissions as well as model errors; inspect images.",
              "Examples are the first six test-list entries for each seed-42 model, not selected successes.",
              "FLOPs are omitted because generic hook-based profilers may miss functional attention operations.",
              "Do not tune models after inspecting test outcomes. Preserve this evaluation as the final study."]
    write_json(out / "aggregate.json", summary)
    with (out / "aggregate.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    for ax, metric, title in zip(axes, ("map50", "map50_95"), ("Test mAP50", "Test mAP50-95")):
        for index, variant in enumerate(VARIANTS):
            values = [r[metric]*100 for r in results["test"] if r["model"] == variant]
            ax.errorbar(index, np.mean(values), yerr=np.std(values, ddof=1), fmt="o", color=colors[index], capsize=5)
            ax.scatter(index+np.array([-.08, 0, .08]), values, color=colors[index], alpha=.6, s=18)
        ax.set_xticks(range(4), ["Baseline", "SE", "CBAM", "Coordinate\nAttention"])
        ax.set_ylabel("AP (%)")
        ax.set_title(title)
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("Three matched seeds: mean +/- sample SD; dots show individual runs")
    fig.savefig(out / "test_accuracy.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 5), layout="constrained")
    for row, name, color in zip(summary, names, colors):
        ax.errorbar(row["forward_median_ms"], row["test_map50_95_mean"], yerr=row["test_map50_95_sd"],
                    fmt="o", color=color, capsize=4, label=name)
    ax.set_xlabel("Forward-only GPU latency (median ms, seed 42)")
    ax.set_ylabel("Test mAP50-95 (%, mean +/- SD across seeds)")
    ax.set_title("Accuracy and measured inference cost")
    ax.grid(alpha=.2)
    ax.legend()
    fig.savefig(out / "accuracy_cost.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    for variant, name, color in zip(VARIANTS, names, colors):
        curves = []
        for seed in (42, 43, 44):
            with (runs[variant, seed] / "results.csv").open() as f:
                curves.append([float(r["metrics/mAP50-95(B)"])*100 for r in csv.DictReader(f)])
        mean, sd = np.mean(curves, axis=0), np.std(curves, axis=0, ddof=1)
        ax.plot(range(1, 51), mean, label=name, color=color)
        ax.fill_between(range(1, 51), mean-sd, mean+sd, color=color, alpha=.12)
    ax.set(xlabel="Epoch", ylabel="Validation mAP50-95 (%)", title="Training progress: mean +/- SD, three seeds")
    ax.legend()
    ax.grid(alpha=.2)
    fig.savefig(out / "learning_comparison.png", dpi=180)
    plt.close(fig)
    print(out / "RESULTS.md")


if __name__ == "__main__":
    main()
