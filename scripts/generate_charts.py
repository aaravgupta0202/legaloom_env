#!/usr/bin/env python3
"""
Generate all charts for the LegaLoom-Env submission.

Reads three JSON artifacts produced by the notebooks:
  - training_scores.json    (from LegaLoom_FullCurriculum.ipynb Cell 8)
  - training_log.json       (from LegaLoom_FullCurriculum.ipynb Cell 6)
  - adversarial_results.json (from LegaLoom_AdversarialBenchmark.ipynb Cell 9)

Outputs PNGs in the repo root (overwrites if present):
  1. before_after.png         — bar chart: baseline vs trained, plus per-task Δ
  2. reward_curves.png        — annotated GRPO training curves
  3. reward_distribution.png  — bimodal score histograms (baseline vs trained)
  4. episode_scatter.png      — per-episode paired comparison, 4 task panels
  5. adversarial_heatmap.png  — model × failure-mode category scores (only if
                                 adversarial_results.json exists)
  6. model_leaderboard.png    — cross-model comparison (only if
                                 aggregated_models.json exists with 2+ models)

Usage from repo root, after a training run:
    python scripts/generate_charts.py

The functions are importable so the team can regenerate selectively without
re-running the full pipeline.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

# Optional dependency: seaborn is used for histograms and the heatmap. We fall
# back to matplotlib for the histogram if seaborn isn't installed, but the
# heatmap genuinely needs seaborn — we'll error out cleanly there.
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


# ---------------------------------------------------------------------------
# Shared style — keep colors consistent with existing artifact convention
# ---------------------------------------------------------------------------

BASELINE_COLOR = "#E76F51"   # terracotta orange
TRAINED_COLOR = "#2A9D8F"    # teal green
THRESHOLD_COLOR = "#888888"
ANNOTATE_COLOR = "#264653"

TASK_LABELS = {
    "task_easy":   "Easy",
    "task_medium": "Medium",
    "task_hard":   "Hard",
    "task_expert": "Expert",
}
TASK_ORDER = ["task_easy", "task_medium", "task_hard", "task_expert"]


def _load_json(path: str | Path) -> Optional[dict]:
    p = Path(path)
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def _ensure_seaborn() -> None:
    if not HAS_SEABORN:
        raise RuntimeError(
            "seaborn is required for this chart. Install with `pip install seaborn`."
        )


# ===========================================================================
# Chart 1 — Before/After + Per-Task Improvement (replaces existing)
# ===========================================================================

def make_before_after(
    scores_path: str = "training_scores.json",
    out_path: str = "before_after.png",
) -> str:
    """Side-by-side: (left) baseline vs trained bars with std-dev; (right) per-task Δ%."""
    data = _load_json(scores_path)
    if data is None:
        raise FileNotFoundError(f"{scores_path} not found. Run training notebook first.")

    baseline = data["baseline"]
    trained = data["trained"]
    base_per = data["baseline_per_episode"]
    trained_per = data["trained_per_episode"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=120,
                                     gridspec_kw={"width_ratios": [1.3, 1.0]})

    # --- Left: baseline vs trained bars ---
    labels = [TASK_LABELS[t] for t in TASK_ORDER]
    before_vals = [baseline[t] for t in TASK_ORDER]
    after_vals = [trained[t] for t in TASK_ORDER]
    before_std = [float(np.std(base_per[t])) for t in TASK_ORDER]
    after_std = [float(np.std(trained_per[t])) for t in TASK_ORDER]

    x = np.arange(len(TASK_ORDER))
    w = 0.36
    ax1.bar(x - w/2, before_vals, w, yerr=before_std, capsize=4,
            label="Before GRPO (untrained)", color=BASELINE_COLOR,
            error_kw={"alpha": 0.6})
    ax1.bar(x + w/2, after_vals, w, yerr=after_std, capsize=4,
            label="After GRPO (single-phase task_hard)", color=TRAINED_COLOR,
            error_kw={"alpha": 0.6})
    for i in range(len(TASK_ORDER)):
        ax1.text(i - w/2, before_vals[i] + before_std[i] + 0.025,
                 f"{before_vals[i]:.3f}", ha="center", fontsize=9)
        ax1.text(i + w/2, after_vals[i] + after_std[i] + 0.025,
                 f"{after_vals[i]:.3f}", ha="center", fontsize=9)
    ax1.axhline(y=0.5, color=THRESHOLD_COLOR, linestyle="--", alpha=0.5,
                label="Success threshold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("Average Score")
    ax1.set_title("LegaLoom-Env — Before vs After GRPO Training")
    ax1.set_ylim(0, 1.0)
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(axis="y", alpha=0.3)

    # --- Right: per-task improvement bars ---
    deltas = [(trained[t] - baseline[t]) / baseline[t] * 100 if baseline[t] > 0 else 0.0
              for t in TASK_ORDER]
    sorted_pairs = sorted(zip(labels, deltas), key=lambda p: -p[1])
    sorted_labels = [p[0] for p in sorted_pairs]
    sorted_deltas = [p[1] for p in sorted_pairs]
    colors = [TRAINED_COLOR if d > 0 else BASELINE_COLOR for d in sorted_deltas]
    bars = ax2.barh(sorted_labels, sorted_deltas, color=colors, edgecolor="white")
    # Compute axis range with padding for labels on both sides
    xmax = max(sorted_deltas + [0.0])
    xmin = min(sorted_deltas + [0.0])
    span = max(xmax - xmin, 1.0)
    pad = span * 0.18
    for b, d in zip(bars, sorted_deltas):
        # Place label outside bar end, on the side of the bar's tip
        if d >= 0:
            ax2.text(b.get_width() + span * 0.01,
                     b.get_y() + b.get_height()/2,
                     f"{d:+.0f}%", va="center", ha="left",
                     fontsize=11, fontweight="bold")
        else:
            ax2.text(b.get_width() - span * 0.01,
                     b.get_y() + b.get_height()/2,
                     f"{d:+.0f}%", va="center", ha="right",
                     fontsize=11, fontweight="bold")
    ax2.set_xlim(xmin - pad, xmax + pad)
    ax2.set_xlabel("% Improvement vs Baseline")
    ax2.set_title("Per-Task Improvement")
    ax2.axvline(x=0, color="black", linewidth=0.8)
    ax2.grid(axis="x", alpha=0.3)
    ax2.invert_yaxis()

    fig.suptitle("", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    return out_path


# ===========================================================================
# Chart 2 — Annotated GRPO training curves (overwrites existing)
# ===========================================================================

def make_reward_curves(
    log_path: str = "training_log.json",
    out_path: str = "reward_curves.png",
) -> str:
    """Reward + loss curves with annotations for peaks and zero-variance regions."""
    log = _load_json(log_path)
    if log is None:
        raise FileNotFoundError(f"{log_path} not found. Run training notebook first.")

    train_entries = [e for e in log if "reward" in e and "loss" in e]
    rewards = [float(e["reward"]) for e in train_entries]
    losses = [float(e["loss"]) for e in train_entries]
    zero_std = [float(e.get("frac_reward_zero_std", 0.0)) for e in train_entries]
    steps = list(range(1, len(rewards) + 1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5), dpi=120)

    # --- Left: reward curve with annotations ---
    # Shade zero-variance regions (frac_reward_zero_std == 1.0 means GRPO had no signal)
    in_zero = False
    span_start = None
    for i, z in enumerate(zero_std):
        if z >= 0.99 and not in_zero:
            span_start = steps[i]
            in_zero = True
        elif z < 0.99 and in_zero:
            ax1.axvspan(span_start - 0.5, steps[i] - 0.5, color="#cccccc", alpha=0.35)
            in_zero = False
    if in_zero:
        ax1.axvspan(span_start - 0.5, steps[-1] + 0.5, color="#cccccc", alpha=0.35)

    ax1.plot(steps, rewards, "b-", linewidth=1.4, alpha=0.6, label="Episode reward")
    if len(rewards) >= 3:
        w = 3
        ma = [sum(rewards[max(0, i-w+1):i+1]) / min(i+1, w) for i in range(len(rewards))]
        ax1.plot(steps, ma, "r-", linewidth=2.4, label="3-step moving avg")
    ax1.axhline(y=0.5, color=THRESHOLD_COLOR, linestyle="--", alpha=0.5, label="Success threshold")

    # Annotate two highest rewards (data-driven, not hardcoded)
    if rewards:
        peak_idx = int(np.argmax(rewards))
        first_high_idx = next(
            (i for i, r in enumerate(rewards) if r > 0.12 and i < peak_idx), None
        )
        if first_high_idx is not None and first_high_idx != peak_idx:
            ax1.annotate(
                "First major rollout",
                xy=(steps[first_high_idx], rewards[first_high_idx]),
                xytext=(steps[first_high_idx] + 4, rewards[first_high_idx] + 0.15),
                fontsize=9, color=ANNOTATE_COLOR,
                arrowprops=dict(arrowstyle="->", color=ANNOTATE_COLOR, alpha=0.7),
            )
        ax1.annotate(
            "Policy peak",
            xy=(steps[peak_idx], rewards[peak_idx]),
            xytext=(steps[peak_idx] - 8, rewards[peak_idx] + 0.18),
            fontsize=9, color=ANNOTATE_COLOR,
            arrowprops=dict(arrowstyle="->", color=ANNOTATE_COLOR, alpha=0.7),
        )

    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Episode Reward")
    ax1.set_title("GRPO — task_hard (40 steps, num_gen=8)")
    ax1.set_ylim(0, 1.0)
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # --- Right: loss curve with peak annotation ---
    if losses:
        ax2.plot(steps, losses, "g-", linewidth=1.5, alpha=0.85)
        peak_loss_idx = int(np.argmax(losses))
        ax2.annotate(
            "Trajectory absorbed\nby policy update",
            xy=(steps[peak_loss_idx], losses[peak_loss_idx]),
            xytext=(steps[peak_loss_idx] + 5, losses[peak_loss_idx] * 0.7 + 1e-6),
            fontsize=9, color=ANNOTATE_COLOR,
            arrowprops=dict(arrowstyle="->", color=ANNOTATE_COLOR, alpha=0.7),
        )
    ax2.set_xlabel("Training Step")
    ax2.set_ylabel("Loss")
    ax2.set_title("GRPO — Loss")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    return out_path


# ===========================================================================
# Chart 3 — Reward distribution histogram (new)
# ===========================================================================

def make_reward_distribution(
    scores_path: str = "training_scores.json",
    out_path: str = "reward_distribution.png",
) -> str:
    """Overlapping histograms — bimodal score distribution, baseline vs trained."""
    data = _load_json(scores_path)
    if data is None:
        raise FileNotFoundError(f"{scores_path} not found. Run training notebook first.")

    base_per = data["baseline_per_episode"]
    trained_per = data["trained_per_episode"]
    baseline_flat = [s for t in TASK_ORDER for s in base_per[t]]
    trained_flat  = [s for t in TASK_ORDER for s in trained_per[t]]

    fig, ax = plt.subplots(figsize=(12, 5), dpi=120)
    bins = np.arange(0, 1.05, 0.05)

    if HAS_SEABORN:
        sns.histplot(baseline_flat, bins=bins, color=BASELINE_COLOR, alpha=0.6,
                     label=f"Baseline (n={len(baseline_flat)})", ax=ax)
        sns.histplot(trained_flat, bins=bins, color=TRAINED_COLOR, alpha=0.6,
                     label=f"Trained (n={len(trained_flat)})", ax=ax)
    else:
        ax.hist(baseline_flat, bins=bins, color=BASELINE_COLOR, alpha=0.6,
                label=f"Baseline (n={len(baseline_flat)})")
        ax.hist(trained_flat, bins=bins, color=TRAINED_COLOR, alpha=0.6,
                label=f"Trained (n={len(trained_flat)})")

    ax.axvline(x=0.5, color=THRESHOLD_COLOR, linestyle="--", alpha=0.6,
               label="Success threshold")
    ax.set_xlabel("Episode Score")
    ax.set_ylabel("Count")
    ax.set_title("Score Distribution — All Episodes (4 tasks × 30 evals)")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    # Annotation on the bimodal nature — placed in middle of plot below legend
    above_threshold_base = sum(1 for s in baseline_flat if s >= 0.5)
    above_threshold_trained = sum(1 for s in trained_flat if s >= 0.5)
    ax.text(0.50, 0.62,
            f"Bimodal distribution: most scores cluster at 0.01 (failure) or ≥0.99 (correct).\n"
            f"Episodes scoring ≥0.5: baseline {above_threshold_base}/{len(baseline_flat)} "
            f"→ trained {above_threshold_trained}/{len(trained_flat)}.",
            transform=ax.transAxes, fontsize=9, ha="center", va="top",
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#aaa", alpha=0.95))

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    return out_path


# ===========================================================================
# Chart 4 — Per-episode paired scatter (new)
# ===========================================================================

def make_episode_scatter(
    scores_path: str = "training_scores.json",
    out_path: str = "episode_scatter.png",
) -> str:
    """2×2 grid — one task per panel; points above y=x mean training helped."""
    data = _load_json(scores_path)
    if data is None:
        raise FileNotFoundError(f"{scores_path} not found. Run training notebook first.")

    base_per = data["baseline_per_episode"]
    trained_per = data["trained_per_episode"]
    baseline_means = data["baseline"]
    trained_means = data["trained"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 12), dpi=120)
    panels = list(zip(TASK_ORDER, axes.flat))
    for task_id, ax in panels:
        x_vals = base_per[task_id]
        y_vals = trained_per[task_id]
        ax.plot([0, 1], [0, 1], "--", color=THRESHOLD_COLOR, alpha=0.5,
                linewidth=1, label="y = x (no change)")
        ax.scatter(x_vals, y_vals, alpha=0.5, s=70, color=TRAINED_COLOR,
                   edgecolor="white", linewidth=0.5)
        b, t = baseline_means[task_id], trained_means[task_id]
        d = (t - b) / b * 100 if b > 0 else 0.0
        ax.set_title(f"{TASK_LABELS[task_id]}: {b:.3f} → {t:.3f}  ({d:+.0f}%)",
                     fontsize=11)
        ax.set_xlabel("Baseline score (per episode)")
        ax.set_ylabel("Trained score (per episode)")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Per-Episode Comparison — n=30 per task, paired seeds 42–71",
                 fontsize=13, y=1.00)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    return out_path


# ===========================================================================
# Chart 5 — Adversarial benchmark heatmap (new; depends on Cell 9 output)
# ===========================================================================

def make_adversarial_heatmap(
    results_path: str = "adversarial_results.json",
    out_path: str = "adversarial_heatmap.png",
    highlight_model: str = "Qwen2.5-3B (trained)",
) -> Optional[str]:
    """Model × failure-mode heatmap. Only runs if adversarial_results.json exists.

    Returns the output path on success, None if results aren't available yet.
    """
    data = _load_json(results_path)
    if data is None:
        print(f"  (skipped — {results_path} not found; run AdversarialBenchmark notebook first)")
        return None

    _ensure_seaborn()

    by_model = data.get("by_model", {})
    if not by_model:
        print(f"  (skipped — by_model is empty in {results_path})")
        return None

    # Sort rows by overall_mean descending
    ranked = sorted(by_model.items(), key=lambda kv: -kv[1].get("overall_mean", 0.0))
    model_names = [n for n, _ in ranked]

    all_categories = sorted(
        {c for _, agg in ranked for c in agg.get("by_category", {})}
    )

    matrix = np.zeros((len(model_names), len(all_categories)))
    for i, (_, agg) in enumerate(ranked):
        cat_dict = agg.get("by_category", {})
        for j, cat in enumerate(all_categories):
            matrix[i, j] = float(cat_dict.get(cat, np.nan))

    fig_w = max(10.0, len(all_categories) * 1.4)
    fig, ax = plt.subplots(figsize=(fig_w, max(4.5, len(model_names) * 0.7)), dpi=120)

    sns.heatmap(
        matrix, annot=True, fmt=".2f",
        xticklabels=[c.replace("_", "\n") for c in all_categories],
        yticklabels=model_names,
        cmap="RdYlGn", vmin=0.0, vmax=1.0, ax=ax,
        cbar_kws={"label": "Mean Score (20 cases)"},
    )

    # Highlight the trained Qwen-3B row with bold black border
    if highlight_model in model_names:
        row_idx = model_names.index(highlight_model)
        from matplotlib.patches import Rectangle
        ax.add_patch(Rectangle(
            (0, row_idx), len(all_categories), 1,
            fill=False, edgecolor="black", linewidth=2.5,
        ))

    ax.set_title(f"Adversarial Benchmark — Model × Failure-Mode Category "
                 f"(n=20 cases, {len(all_categories)} categories)",
                 fontsize=11, pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.xticks(rotation=0, fontsize=8)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    return out_path


# ===========================================================================
# CLI
# ===========================================================================

def make_model_leaderboard(
    aggregated_path: str | Path = "aggregated_models.json",
    output_path: str | Path = "model_leaderboard.png",
) -> Optional[str]:
    """Cross-model comparison chart: 1×2 layout, 6-bar grouped + per-task Δ.

    Reads aggregated_models.json (output of scripts/aggregate_models.py).
    Produces model_leaderboard.png.

    Returns the path on success, None if there's only 1 model (single-model
    workflow doesn't need this chart — make_before_after covers it).
    """
    data = _load_json(aggregated_path)
    if not data or data.get("n_models", 0) < 2:
        n = (data or {}).get("n_models", 0)
        print(f"  ✗ model_leaderboard.png: need at least 2 models for "
              f"comparison, found {n}. Run more models or skip.")
        return None

    models = data["models"]
    # Stable order: qwen3b, gemma2b, llama3b (always preserve this order
    # whether or not all three are present)
    canonical_order = ["qwen3b", "gemma2b", "llama3b"]
    tags = [t for t in canonical_order if t in models]

    # Color per model (consistent across both subplots)
    MODEL_COLORS = {
        "qwen3b":  "#E76F51",   # terracotta — same as Qwen baseline color
        "gemma2b": "#2A9D8F",   # teal
        "llama3b": "#F4A261",   # sandy orange
    }

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(16, 6.0))

    # ----- LEFT: grouped bars, 4 tasks, 6 bars each (3 models × 2 conditions) -----
    n_tasks = len(TASK_ORDER)
    n_models = len(tags)
    bar_w = 0.13
    group_gap = 0.10
    # Position bars: for each task, place 2*n_models bars
    cluster_width = (2 * n_models) * bar_w
    x_centers = np.arange(n_tasks) * (cluster_width + group_gap)

    for mi, tag in enumerate(tags):
        m = models[tag]
        base_vals = [m["baseline"][t] for t in TASK_ORDER]
        trained_vals = [m["trained"][t] for t in TASK_ORDER]
        color = MODEL_COLORS.get(tag, "#888888")
        # Baseline bar (lighter): first slot in pair
        offset_base = (mi * 2 - n_models) * bar_w + bar_w / 2
        offset_trained = offset_base + bar_w
        ax_left.bar(x_centers + offset_base, base_vals, bar_w,
                    color=color, alpha=0.45,
                    label=f"{m['label']} — baseline",
                    edgecolor="white", linewidth=0.4)
        ax_left.bar(x_centers + offset_trained, trained_vals, bar_w,
                    color=color, alpha=1.0,
                    label=f"{m['label']} — trained",
                    edgecolor="white", linewidth=0.4)

    ax_left.set_xticks(x_centers)
    ax_left.set_xticklabels([TASK_LABELS[t] for t in TASK_ORDER])
    ax_left.set_ylabel("Average Score")
    ax_left.set_ylim(0, 1.0)
    ax_left.axhline(0.5, color=THRESHOLD_COLOR, linestyle="--",
                     linewidth=0.8, alpha=0.6, label="Success threshold")
    ax_left.set_title("Per-task scores — baseline (light) vs trained (solid)")
    ax_left.legend(loc="upper left", fontsize=8, ncol=1, framealpha=0.95)
    ax_left.spines["top"].set_visible(False)
    ax_left.spines["right"].set_visible(False)
    ax_left.grid(axis="y", linestyle=":", alpha=0.4)

    # ----- RIGHT: per-task Δ, grouped by model -----
    bar_w_r = 0.22
    cluster_w = n_models * bar_w_r
    x_r = np.arange(n_tasks)

    for mi, tag in enumerate(tags):
        m = models[tag]
        deltas = [m["lift"][t] for t in TASK_ORDER]
        color = MODEL_COLORS.get(tag, "#888888")
        offset = (mi - (n_models - 1) / 2) * bar_w_r
        bars = ax_right.bar(x_r + offset, deltas, bar_w_r,
                              color=color, alpha=0.95,
                              label=m["label"], edgecolor="white", linewidth=0.4)
        # Annotate
        for bar, d, base in zip(bars, deltas, [m["baseline"][t] for t in TASK_ORDER]):
            pct = (d / base * 100.0) if base > 0 else 0.0
            label = f"{pct:+.0f}%"
            y = bar.get_height()
            va = "bottom" if y >= 0 else "top"
            y_offset = 0.005 if y >= 0 else -0.005
            ax_right.text(bar.get_x() + bar.get_width() / 2, y + y_offset,
                           label, ha="center", va=va, fontsize=8,
                           color=ANNOTATE_COLOR)

    ax_right.axhline(0, color="#000", linewidth=0.6, alpha=0.4)
    ax_right.set_xticks(x_r)
    ax_right.set_xticklabels([TASK_LABELS[t] for t in TASK_ORDER])
    ax_right.set_ylabel("Δ vs baseline (absolute score)")
    ax_right.set_title("Per-task improvement after GRPO post-training")
    ax_right.legend(loc="best", fontsize=9, framealpha=0.95)
    ax_right.spines["top"].set_visible(False)
    ax_right.spines["right"].set_visible(False)
    ax_right.grid(axis="y", linestyle=":", alpha=0.4)

    fig.suptitle("Model Comparison — GRPO Post-Training on TDS Compliance",
                 fontsize=13, y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def main(argv=None) -> int:
    """Generate all available charts. Skips charts whose data isn't on disk yet."""
    out_paths = []

    print("Generating charts...")
    for name, fn in [
        ("before_after.png",        make_before_after),
        ("reward_curves.png",       make_reward_curves),
        ("reward_distribution.png", make_reward_distribution),
        ("episode_scatter.png",     make_episode_scatter),
    ]:
        try:
            p = fn()
            print(f"  ✓ {p}")
            out_paths.append(p)
        except FileNotFoundError as e:
            print(f"  ✗ {name}: {e}")

    # Adversarial heatmap is optional (depends on benchmark having been run)
    print("Adversarial heatmap (optional):")
    try:
        p = make_adversarial_heatmap()
        if p:
            print(f"  ✓ {p}")
            out_paths.append(p)
    except Exception as e:
        print(f"  ✗ adversarial_heatmap.png: {type(e).__name__}: {e}")

    # Model leaderboard is optional (depends on multi-model aggregation having been run)
    print("Model leaderboard (optional, multi-model only):")
    try:
        p = make_model_leaderboard()
        if p:
            print(f"  ✓ {p}")
            out_paths.append(p)
    except Exception as e:
        print(f"  ✗ model_leaderboard.png: {type(e).__name__}: {e}")

    print(f"\nGenerated {len(out_paths)} charts.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
