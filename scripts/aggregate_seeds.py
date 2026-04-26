#!/usr/bin/env python3
"""
Aggregate training results across multiple seed runs.

Reads `runs/seed_*/training_scores.json` (or whatever pattern matches),
computes mean ± std for baseline, trained, and lift across runs.

Layout convention:

    runs/
      seed_42/training_scores.json
      seed_100/training_scores.json
      seed_200/training_scores.json
      seed_300/training_scores.json

After running training with multiple seeds, place each run's
`training_scores.json` under its own directory before invoking this script.

Output:
  - aggregated_results.json  (machine-readable)
  - prints a markdown table to stdout for direct paste into the README

Usage:
    python scripts/aggregate_seeds.py
    python scripts/aggregate_seeds.py --pattern 'runs/seed_*/training_scores.json'
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np


TASKS = ["task_easy", "task_medium", "task_hard", "task_expert"]


def aggregate(run_files: List[str]) -> dict:
    """Compute per-task mean ± std (sample std, ddof=1) across runs.

    Single-run case: report n_runs=1 and skip std (would be NaN).
    """
    if not run_files:
        return {
            "n_runs": 0,
            "seeds": [],
            "note": "No run files found. Place per-seed results under runs/seed_*/training_scores.json.",
        }

    runs = []
    seeds = []
    for path in run_files:
        try:
            data = json.load(open(path))
            runs.append(data)
            # Try to get seed from data; fall back to parsing dir name
            seed = data.get("seed")
            if seed is None:
                # parse "runs/seed_42/training_scores.json" → 42
                p = Path(path)
                parent = p.parent.name
                if parent.startswith("seed_"):
                    try:
                        seed = int(parent.split("_", 1)[1])
                    except ValueError:
                        seed = None
            seeds.append(seed)
        except Exception as e:
            print(f"  ⚠ Skipped {path}: {type(e).__name__}: {e}", file=sys.stderr)

    n_runs = len(runs)
    if n_runs == 0:
        return {"n_runs": 0, "seeds": [], "note": "Failed to load any run files."}

    out = {
        "n_runs": n_runs,
        "seeds": seeds,
        "source_files": list(run_files),
        "per_task": {},
    }

    for task in TASKS:
        baselines = [float(r["baseline"][task]) for r in runs if task in r.get("baseline", {})]
        traineds  = [float(r["trained"][task])  for r in runs if task in r.get("trained", {})]
        if len(baselines) != n_runs or len(traineds) != n_runs:
            out["per_task"][task] = {
                "note": f"missing data — baseline n={len(baselines)}, trained n={len(traineds)}"
            }
            continue
        lifts = [t - b for b, t in zip(baselines, traineds)]

        entry = {
            "baseline_mean": float(np.mean(baselines)),
            "trained_mean":  float(np.mean(traineds)),
            "lift_mean":     float(np.mean(lifts)),
            "n_runs":        n_runs,
        }
        # Sample std (ddof=1) only meaningful with ≥2 runs
        if n_runs >= 2:
            entry["baseline_std"] = float(np.std(baselines, ddof=1))
            entry["trained_std"]  = float(np.std(traineds,  ddof=1))
            entry["lift_std"]     = float(np.std(lifts,     ddof=1))
        else:
            entry["baseline_std"] = None
            entry["trained_std"]  = None
            entry["lift_std"]     = None
        out["per_task"][task] = entry

    # Average lift across tasks per run, then mean/std across runs
    avg_lifts_per_run = []
    for r in runs:
        try:
            lifts = [r["trained"][t] - r["baseline"][t] for t in TASKS]
            avg_lifts_per_run.append(float(np.mean(lifts)))
        except KeyError:
            pass

    if avg_lifts_per_run:
        out["average_lift_mean"] = float(np.mean(avg_lifts_per_run))
        out["average_lift_std"]  = (
            float(np.std(avg_lifts_per_run, ddof=1)) if n_runs >= 2 else None
        )
        out["average_lifts_per_run"] = avg_lifts_per_run
    return out


def print_markdown_table(agg: dict) -> None:
    if agg["n_runs"] == 0:
        print(agg.get("note", "(no data)"))
        return

    n = agg["n_runs"]
    seeds = ", ".join(str(s) if s is not None else "?" for s in agg.get("seeds", []))
    print(f"\n## Reproducibility across seeds (n={n}, seeds: {seeds})\n")

    if n == 1:
        print("⚠ Only 1 run completed. Reproducibility analysis pending — "
              "run training with seeds 100, 200, 300 to compute cross-seed std.\n")
        print("| Task | Baseline | Trained | Lift |")
        print("|------|---------:|--------:|-----:|")
        for task in TASKS:
            e = agg["per_task"].get(task, {})
            if "baseline_mean" not in e:
                continue
            print(f"| {task} | {e['baseline_mean']:.3f} | "
                  f"{e['trained_mean']:.3f} | {e['lift_mean']:+.3f} |")
        return

    print("| Task | Baseline (μ ± σ) | Trained (μ ± σ) | Lift (μ ± σ) |")
    print("|------|-----------------:|----------------:|-------------:|")
    for task in TASKS:
        e = agg["per_task"].get(task, {})
        if "baseline_mean" not in e:
            print(f"| {task} | (missing) | (missing) | (missing) |")
            continue
        print(f"| {task} | "
              f"{e['baseline_mean']:.3f} ± {e['baseline_std']:.3f} | "
              f"{e['trained_mean']:.3f} ± {e['trained_std']:.3f} | "
              f"{e['lift_mean']:+.3f} ± {e['lift_std']:.3f} |")

    if "average_lift_mean" in agg:
        m = agg["average_lift_mean"]
        s = agg["average_lift_std"]
        rel_pct = (m / np.mean([
            agg["per_task"][t]["baseline_mean"] for t in TASKS
            if "baseline_mean" in agg["per_task"].get(t, {})
        ])) * 100 if n >= 1 else 0
        if s is not None:
            print(f"\n**Average lift across {n} runs: μ={m:+.3f} (≈{rel_pct:+.1f}%), σ={s:.3f}**")
        else:
            print(f"\n**Average lift: {m:+.3f} (≈{rel_pct:+.1f}%)**")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pattern", default="runs/seed_*/training_scores.json",
                        help="glob pattern for per-seed score files")
    parser.add_argument("--out", default="aggregated_results.json",
                        help="output path (default: ./aggregated_results.json)")
    parser.add_argument("--include-default", action="store_true",
                        help="also include ./training_scores.json if present "
                             "(treats it as the seed_42 run)")
    args = parser.parse_args(argv)

    files = sorted(glob.glob(args.pattern))
    if args.include_default and Path("training_scores.json").exists():
        # Avoid duplicating if it also matches the pattern
        if "training_scores.json" not in files:
            files.append("training_scores.json")

    print(f"Aggregating {len(files)} run file(s):")
    for f in files:
        print(f"  - {f}")

    agg = aggregate(files)
    with open(args.out, "w") as f:
        json.dump(agg, f, indent=2)

    print_markdown_table(agg)
    print(f"\n✓ Saved to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
