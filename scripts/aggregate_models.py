#!/usr/bin/env python3
"""
aggregate_models.py — aggregate per-model training_scores files into one JSON
that downstream chart + populate scripts can consume.

Reads (from repo root):
    training_scores_qwen3b.json
    training_scores_gemma2b.json
    training_scores_llama3b.json
    (any combination — at least 1 required)

    If training_scores_qwen3b.json is missing but training_scores.json exists,
    the untagged file is treated as the qwen3b artifact (single-model workflow
    backwards compatibility).

Writes:
    aggregated_models.json — {n_models, models: {tag: {label, baseline, trained, lift}}}

Usage:
    python scripts/aggregate_models.py
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent

MODEL_LABELS = {
    "qwen3b": "Qwen2.5-3B",
    "gemma2b": "Gemma-2-2B",
    "llama3b": "Llama-3.2-3B",
}

TASKS = ["task_easy", "task_medium", "task_hard", "task_expert"]


def _load_scores(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    # sanity check
    if "baseline" not in data or "trained" not in data:
        print(f"[aggregate_models] {path.name}: missing baseline/trained keys, skipping")
        return None
    for t in TASKS:
        if t not in data["baseline"] or t not in data["trained"]:
            print(f"[aggregate_models] {path.name}: missing task {t}, skipping")
            return None
    return data


def _avg(d: Dict[str, float]) -> float:
    return sum(float(d[t]) for t in TASKS) / len(TASKS)


def main() -> int:
    out: Dict[str, Any] = {"n_models": 0, "models": {}}

    # 1. Pick up all tagged files
    tagged_paths = sorted((REPO_ROOT).glob("training_scores_*.json"))
    seen_tags = set()
    for path in tagged_paths:
        tag = path.stem.replace("training_scores_", "")
        if tag not in MODEL_LABELS:
            print(f"  Skipping unknown tag: {tag}")
            continue
        data = _load_scores(path)
        if data is None:
            continue
        seen_tags.add(tag)
        out["models"][tag] = {
            "label":    MODEL_LABELS[tag],
            "baseline": {t: float(data["baseline"][t]) for t in TASKS},
            "trained":  {t: float(data["trained"][t])  for t in TASKS},
            "lift":     {t: float(data["trained"][t]) - float(data["baseline"][t]) for t in TASKS},
            "baseline_avg": _avg(data["baseline"]),
            "trained_avg":  _avg(data["trained"]),
            "lift_avg":     _avg(data["trained"]) - _avg(data["baseline"]),
            "source_file": path.name,
        }
        print(f"  Loaded {tag}: baseline_avg={out['models'][tag]['baseline_avg']:.3f}, "
              f"trained_avg={out['models'][tag]['trained_avg']:.3f}, "
              f"lift={out['models'][tag]['lift_avg']*100:+.1f}%")

    # 2. Backwards-compat: if qwen3b not seen but untagged training_scores.json
    # exists, treat it as the qwen3b artifact
    untagged = REPO_ROOT / "training_scores.json"
    if "qwen3b" not in seen_tags and untagged.exists():
        data = _load_scores(untagged)
        if data is not None:
            out["models"]["qwen3b"] = {
                "label":    MODEL_LABELS["qwen3b"],
                "baseline": {t: float(data["baseline"][t]) for t in TASKS},
                "trained":  {t: float(data["trained"][t])  for t in TASKS},
                "lift":     {t: float(data["trained"][t]) - float(data["baseline"][t]) for t in TASKS},
                "baseline_avg": _avg(data["baseline"]),
                "trained_avg":  _avg(data["trained"]),
                "lift_avg":     _avg(data["trained"]) - _avg(data["baseline"]),
                "source_file": untagged.name + " (untagged fallback)",
            }
            print(f"  Loaded qwen3b from {untagged.name} (single-model workflow fallback)")

    out["n_models"] = len(out["models"])

    if out["n_models"] == 0:
        print("[aggregate_models] No training_scores files found. Run training first.")
        return 1

    # Rank by average lift
    ranked = sorted(out["models"].items(), key=lambda kv: -kv[1]["lift_avg"])
    out["ranking"] = [{"tag": tag, "label": d["label"], "lift_avg": d["lift_avg"]}
                       for tag, d in ranked]

    output_path = REPO_ROOT / "aggregated_models.json"
    with open(output_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n✓ Saved {output_path.name} with {out['n_models']} model(s)")
    if out["n_models"] >= 2:
        winner = ranked[0]
        print(f"  Best lift: {winner[1]['label']} ({winner[1]['lift_avg']*100:+.1f}%)")
    else:
        print(f"  (single-model run; multi-model comparison needs 2+ models)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
