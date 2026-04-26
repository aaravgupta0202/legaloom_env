#!/usr/bin/env python3
"""
Statistical analysis of LegaLoom-Env training results.

Computes four diagnostics from a completed training run:

  Item 1. Paired Wilcoxon signed-rank test per task (30 baseline-vs-trained pairs)
  Item 2. Bootstrap 95% confidence intervals on per-task deltas (10,000 iterations)
  Item 3. Useful-gradient-signal fraction from frac_reward_zero_std
  Item 4. KL divergence trajectory (start → end → max)

Inputs:
  - training_scores.json  (from LegaLoom_FullCurriculum.ipynb Cell 8)
  - training_log.json     (from LegaLoom_FullCurriculum.ipynb Cell 6)

Output:
  - statistical_results.json — machine-readable summary
  - prints a human-readable report to stdout

Usage:
    python scripts/statistical_analysis.py
    python scripts/statistical_analysis.py --scores path/to/scores.json --log path/to/log.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy.stats import wilcoxon


TASKS = ["task_easy", "task_medium", "task_hard", "task_expert"]


# ---------------------------------------------------------------------------
# Item 1 — Paired Wilcoxon signed-rank test per task
# ---------------------------------------------------------------------------

def wilcoxon_per_task(scores: Dict, alpha: float = 0.05) -> Dict[str, dict]:
    """Paired Wilcoxon signed-rank test per task.

    H0: trained == baseline
    H1: trained > baseline   (alternative='less' tests baseline < trained)

    Wilcoxon is correct here because per-episode scores are bimodal (mostly
    0.01 or 0.99) and not normally distributed. A t-test would be inappropriate.
    """
    results = {}
    for task in TASKS:
        baseline = scores["baseline_per_episode"][task]
        trained = scores["trained_per_episode"][task]

        diffs = [t - b for b, t in zip(baseline, trained)]
        nonzero_diffs = [d for d in diffs if d != 0]
        n_changed = len(nonzero_diffs)
        n_total = len(diffs)
        delta = float(np.mean(trained) - np.mean(baseline))

        if n_changed < 5:
            # Wilcoxon is uninformative with very few changed pairs.
            # Report honestly rather than producing a misleading p-value.
            results[task] = {
                "delta": delta,
                "n_changed": n_changed,
                "n_total": n_total,
                "p_value": None,
                "statistic": None,
                "significant_at_05": False,
                "note": (
                    f"p-value uninformative due to limited policy movement "
                    f"on this task (only {n_changed}/{n_total} pairs changed)"
                ),
            }
            continue

        # alternative='less' tests whether the FIRST argument (baseline) is
        # stochastically LESS than the SECOND (trained) — i.e., training helped.
        # Verified against scipy.stats.wilcoxon docs (>=1.7).
        try:
            stat, p = wilcoxon(baseline, trained, alternative="less")
            results[task] = {
                "delta": delta,
                "n_changed": n_changed,
                "n_total": n_total,
                "p_value": float(p),
                "statistic": float(stat),
                "significant_at_05": bool(p < alpha),
                "note": "",
            }
        except ValueError as e:
            results[task] = {
                "delta": delta,
                "n_changed": n_changed,
                "n_total": n_total,
                "p_value": None,
                "statistic": None,
                "significant_at_05": False,
                "note": f"wilcoxon failed: {e}",
            }
    return results


# ---------------------------------------------------------------------------
# Item 2 — Bootstrap 95% confidence intervals on deltas (paired)
# ---------------------------------------------------------------------------

def bootstrap_ci(
    baseline: List[float],
    trained: List[float],
    n_iter: int = 10000,
    seed: int = 42,
) -> Dict[str, float]:
    """Paired bootstrap 95% CI on the mean delta.

    Critical: must sample the same INDICES for both arrays on each iteration —
    paired bootstrap preserves within-episode correlation. Independent
    bootstrap of the two arrays would inflate the CI by destroying the
    pairing structure.
    """
    rng = np.random.default_rng(seed)
    b_arr = np.asarray(baseline, dtype=float)
    t_arr = np.asarray(trained, dtype=float)
    n = len(b_arr)
    deltas = np.empty(n_iter, dtype=float)
    for i in range(n_iter):
        idx = rng.integers(0, n, size=n)
        deltas[i] = float(np.mean(t_arr[idx] - b_arr[idx]))
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    point_delta = float(np.mean(t_arr - b_arr))
    return {
        "delta": point_delta,
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "ci_excludes_zero": bool(lo > 0 or hi < 0),
        "n_iter": n_iter,
        "n_episodes": n,
    }


def bootstrap_per_task(scores: Dict) -> Dict[str, dict]:
    out = {}
    for task in TASKS:
        out[task] = bootstrap_ci(
            scores["baseline_per_episode"][task],
            scores["trained_per_episode"][task],
        )
    return out


# ---------------------------------------------------------------------------
# Item 3 — Reward variance diagnostic
# ---------------------------------------------------------------------------

def reward_variance_diagnostic(log: List[dict]) -> dict:
    """Count training steps where reward variance was effectively zero.

    Preferred: TRL's ``frac_reward_zero_std`` field, which equals 1.0 when
    all generations in a group scored identically.

    Fallback: when that field is absent (older TRL versions or certain
    configs), compute a proxy from ``reward_std`` — a step counts as
    zero-variance if ``reward_std < 0.001``. The fallback is approximate
    but informative; the report explicitly notes which signal was used.

    With num_generations=4 (prior runs), zero-variance fraction was ~50%.
    With num_generations=8 (this run), it should be substantially lower.
    """
    has_field = [e for e in log if "frac_reward_zero_std" in e]
    used_fallback = False
    if has_field:
        total = len(has_field)
        zero_var = sum(1 for e in has_field
                       if float(e.get("frac_reward_zero_std", 0.0)) >= 0.99)
        signal_source = "frac_reward_zero_std field"
    else:
        # Fallback: reward_std < 1e-3 approximates "all generations identical"
        has_std = [e for e in log if "reward_std" in e]
        if not has_std:
            return {
                "total_steps": 0,
                "zero_variance_steps": 0,
                "useful_steps": 0,
                "useful_fraction": None,
                "signal_source": None,
                "note": "Neither frac_reward_zero_std nor reward_std field present in log",
            }
        total = len(has_std)
        zero_var = sum(1 for e in has_std if float(e.get("reward_std", 0.0)) < 1e-3)
        signal_source = "reward_std < 1e-3 (proxy)"
        used_fallback = True

    useful = total - zero_var
    return {
        "total_steps": total,
        "zero_variance_steps": zero_var,
        "useful_steps": useful,
        "useful_fraction": float(useful) / total if total > 0 else None,
        "signal_source": signal_source,
        "fallback_used": used_fallback,
        "note": "",
    }


# ---------------------------------------------------------------------------
# Item 4 — KL trajectory (policy movement diagnostic)
# ---------------------------------------------------------------------------

def kl_trajectory(log: List[dict]) -> dict:
    """KL divergence from base model at each training step.

    Establishes that training was real (KL > 0) without being catastrophic
    (KL < 0.05). Earlier broken runs had KL barely moving (< 0.001).
    """
    kls = [float(e["kl"]) for e in log if "kl" in e]
    if not kls:
        return {"start": None, "end": None, "max": None, "n_steps": 0,
                "note": "kl field absent from log"}
    return {
        "start": kls[0],
        "end": kls[-1],
        "max": max(kls),
        "n_steps": len(kls),
        "moved_measurably": bool(max(kls) > 1e-3),
        "stayed_safe": bool(max(kls) < 0.05),
        "note": "",
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(results: Dict) -> None:
    print("=" * 64)
    print("LegaLoom-Env — Statistical Analysis")
    print("=" * 64)

    # Item 1 — Wilcoxon
    print("\n[1] Paired Wilcoxon signed-rank test (n=30 per task)")
    print("    H1: trained > baseline (alternative='less' on baseline,trained)")
    for task, r in results["wilcoxon"].items():
        delta = r["delta"]
        n_changed = r["n_changed"]
        n_total = r["n_total"]
        if r["p_value"] is None:
            print(f"  {task}: Δ={delta:+.3f}  n_changed={n_changed}/{n_total}  "
                  f"({r['note']})")
        else:
            sig = "*" if r["significant_at_05"] else " "
            print(f"  {task}: Δ={delta:+.3f}  p={r['p_value']:.4f}{sig}  "
                  f"n_changed={n_changed}/{n_total}")

    # Item 2 — Bootstrap CIs
    print("\n[2] Bootstrap 95% confidence intervals on Δ (paired, 10,000 iter)")
    for task, r in results["bootstrap"].items():
        marker = "*" if r["ci_excludes_zero"] else " "
        print(f"  {task}: Δ={r['delta']:+.3f}  "
              f"95% CI=[{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]{marker}")

    # Item 3 — Reward variance
    rv = results["reward_variance"]
    print("\n[3] Useful gradient signal")
    if rv["useful_fraction"] is not None:
        print(f"  {rv['useful_steps']}/{rv['total_steps']} steps "
              f"({rv['useful_fraction']:.0%}) had non-zero reward variance")
        print(f"  ({rv['zero_variance_steps']} wasted steps where all generations "
              f"scored identically — GRPO had no advantage signal)")
        print(f"  Signal source: {rv['signal_source']}")
        if rv.get("fallback_used"):
            print(f"  ⚠ frac_reward_zero_std field absent from log; using reward_std<1e-3 proxy.")
    else:
        print(f"  ⚠ {rv['note']}")

    # Item 4 — KL trajectory
    kl = results["kl"]
    print("\n[4] Policy movement (KL divergence vs. base model)")
    if kl["n_steps"] > 0:
        moved = "✓" if kl["moved_measurably"] else "✗"
        safe = "✓" if kl["stayed_safe"] else "✗"
        print(f"  Start: {kl['start']:.5f}   End: {kl['end']:.5f}   Max: {kl['max']:.5f}")
        print(f"  Moved measurably (max > 1e-3): {moved}")
        print(f"  Stayed safe (max < 0.05, no policy collapse): {safe}")
    else:
        print(f"  ⚠ {kl['note']}")

    print("\n* significant at α=0.05 / CI excludes zero")
    print("=" * 64)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scores", default="training_scores.json",
                        help="path to training_scores.json (default: ./training_scores.json)")
    parser.add_argument("--log", default="training_log.json",
                        help="path to training_log.json (default: ./training_log.json)")
    parser.add_argument("--out", default="statistical_results.json",
                        help="output path (default: ./statistical_results.json)")
    args = parser.parse_args(argv)

    scores_path = Path(args.scores)
    log_path = Path(args.log)
    if not scores_path.exists():
        print(f"ERROR: {scores_path} not found. Run training notebook first.", file=sys.stderr)
        return 1
    if not log_path.exists():
        print(f"ERROR: {log_path} not found. Run training notebook first.", file=sys.stderr)
        return 1

    scores = json.load(open(scores_path))
    log = json.load(open(log_path))

    results = {
        "source_files": {
            "scores": str(scores_path),
            "log": str(log_path),
        },
        "wilcoxon":         wilcoxon_per_task(scores),
        "bootstrap":        bootstrap_per_task(scores),
        "reward_variance":  reward_variance_diagnostic(log),
        "kl":               kl_trajectory(log),
    }

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    print_report(results)
    print(f"\n✓ Saved to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
