#!/usr/bin/env python3
"""
generate_storytelling_charts.py — 4 additional charts for narrative depth.

Reads from repo root:
    training_scores.json
    training_log.json

Outputs:
    win_loss_tie.png      — per-task outcome heatmap (wins/ties/losses)
    gradient_signal.png   — per-step GRPO signal vs wasted steps
    episode_waterfall.png — per-episode score deltas, 4-panel
    score_regimes.png     — pie charts showing bimodal distribution

Usage:
    python scripts/generate_storytelling_charts.py
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
TASKS = ['task_easy', 'task_medium', 'task_hard', 'task_expert']
LABELS = ['Easy', 'Medium', 'Hard', 'Expert']


def make_win_loss_tie():
    scores = json.load(open(REPO / 'training_scores.json'))

    wins, losses, ties = [], [], []
    for t in TASKS:
        b = scores['baseline_per_episode'][t]
        tr = scores['trained_per_episode'][t]
        w = sum(1 for bi, ti in zip(b, tr) if ti > bi + 0.005)
        l = sum(1 for bi, ti in zip(b, tr) if ti < bi - 0.005)
        tie = len(b) - w - l
        wins.append(w); losses.append(l); ties.append(tie)

    data = np.array([wins, ties, losses]).T
    colors = ['#2A9D8F', '#E9E9E9', '#E76F51']

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i in range(len(TASKS)):
        for j in range(3):
            val = data[i, j]
            ax.add_patch(plt.Rectangle((j - 0.45, i - 0.45), 0.9, 0.9,
                                        facecolor=colors[j],
                                        alpha=0.7 if val > 0 else 0.15,
                                        edgecolor='white', linewidth=2))
            ax.text(j, i, str(val), ha='center', va='center',
                    fontsize=22, fontweight='bold',
                    color='white' if j != 1 and val > 0 else '#333')

    col_labels = ['Win\n(trained > baseline)', 'Tie\n(no change)', 'Loss\n(trained < baseline)']
    ax.set_xticks(range(3)); ax.set_xticklabels(col_labels, fontsize=11)
    ax.set_yticks(range(len(TASKS))); ax.set_yticklabels(LABELS, fontsize=12)
    ax.set_xlim(-0.5, 2.5); ax.set_ylim(len(TASKS) - 0.5, -0.5)
    ax.set_title("Per-Episode Outcome: How Many Seeds Actually Changed?", fontsize=13, pad=12)
    for s in ax.spines.values(): s.set_visible(False)
    ax.tick_params(length=0)

    n = len(scores['baseline_per_episode']['task_easy'])
    ax.text(1, len(TASKS) + 0.15,
            f"n = {n} episodes per task. Most seeds produce identical scores under both policies.\n"
            f"The task_easy regression is driven by {losses[0]} episode(s) flipping, not systemic collapse.",
            ha='center', va='top', fontsize=9, color='#555', style='italic')
    plt.tight_layout()
    plt.savefig(REPO / 'win_loss_tie.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ win_loss_tie.png")


def make_gradient_signal():
    log = json.load(open(REPO / 'training_log.json'))
    steps_data = [e for e in log if 'step' in e and 'reward_std' in e]

    fig, ax = plt.subplots(figsize=(14, 2.0))
    for i, entry in enumerate(steps_data):
        std = entry.get('reward_std', 0)
        has_signal = std > 0.001
        color = '#2A9D8F' if has_signal else '#E76F51'
        ax.barh(0, 1, left=i, height=0.7, color=color,
                alpha=0.85 if has_signal else 0.5,
                edgecolor='white', linewidth=0.5)

    n_signal = sum(1 for e in steps_data if e.get('reward_std', 0) > 0.001)
    n_total = len(steps_data)
    frac = n_signal / n_total if n_total else 0

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor='#2A9D8F', alpha=0.85, label=f'Gradient signal ({n_signal} steps)'),
        Patch(facecolor='#E76F51', alpha=0.5, label=f'Zero variance — wasted ({n_total - n_signal} steps)'),
    ], loc='upper right', fontsize=9)

    ax.set_xlim(0, n_total); ax.set_ylim(-1.2, 0.8); ax.set_yticks([])
    ax.set_xlabel('Training Step', fontsize=10)
    ax.set_title(f"GRPO Gradient Signal: {n_signal}/{n_total} Steps Had Reward Variance ({frac:.0%})",
                 fontsize=12, pad=8)
    for sp in ['top', 'right', 'left']: ax.spines[sp].set_visible(False)
    plt.tight_layout()
    plt.savefig(REPO / 'gradient_signal.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ gradient_signal.png")


def make_episode_waterfall():
    scores = json.load(open(REPO / 'training_scores.json'))

    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
    for ax, t, label in zip(axes, TASKS, LABELS):
        b = scores['baseline_per_episode'][t]
        tr = scores['trained_per_episode'][t]
        deltas = [ti - bi for bi, ti in zip(b, tr)]
        colors_bar = ['#2A9D8F' if d > 0.005 else '#E76F51' if d < -0.005 else '#CCCCCC' for d in deltas]
        ax.bar(range(len(deltas)), deltas, color=colors_bar, edgecolor='white', linewidth=0.5, alpha=0.85)
        ax.axhline(0, color='#333', linewidth=0.5)
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.set_xlabel('Episode', fontsize=9)
        n_w = sum(1 for d in deltas if d > 0.005)
        n_l = sum(1 for d in deltas if d < -0.005)
        n_t = len(deltas) - n_w - n_l
        ax.text(0.95, 0.95, f"+{n_w} / ={n_t} / −{n_l}",
                transform=ax.transAxes, ha='right', va='top', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    axes[0].set_ylabel('Score Change (trained − baseline)', fontsize=10)
    fig.suptitle("Per-Episode Score Change: Which Seeds Moved?", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(REPO / 'episode_waterfall.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ episode_waterfall.png")


def make_score_regimes():
    scores = json.load(open(REPO / 'training_scores.json'))
    regime_colors = ['#2A9D8F', '#F4A261', '#E76F51']
    regime_labels = ['Success (≥0.5)', 'Partial (0.1–0.5)', 'Floor (<0.1)']

    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    for col, (t, label) in enumerate(zip(TASKS, LABELS)):
        for row, (key, subtitle) in enumerate([('baseline_per_episode', 'Baseline'),
                                                ('trained_per_episode', 'Trained')]):
            eps = scores[key][t]
            sizes = [sum(1 for e in eps if e >= 0.5),
                     sum(1 for e in eps if 0.1 <= e < 0.5),
                     sum(1 for e in eps if e < 0.1)]
            ax = axes[row, col]
            if sum(sizes) == 0: sizes = [0, 0, 1]
            ax.pie(sizes, colors=regime_colors,
                   autopct=lambda p: f'{p:.0f}%' if p > 0 else '',
                   startangle=90, textprops={'fontsize': 9})
            if row == 0: ax.set_title(label, fontsize=12, fontweight='bold', pad=8)
            if col == 0: ax.set_ylabel(subtitle, fontsize=10, rotation=0, labelpad=45)

    fig.legend(regime_labels, loc='lower center', ncol=3, fontsize=10,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Score Regime Distribution: Where Do Episodes Land?", fontsize=13, y=1.01)
    plt.tight_layout(rect=[0, 0.04, 1, 0.98])
    plt.savefig(REPO / 'score_regimes.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ score_regimes.png")


def main():
    print("Generating storytelling charts...")
    make_win_loss_tie()
    make_gradient_signal()
    make_episode_waterfall()
    make_score_regimes()
    print("\n✓ All 4 storytelling charts generated")
    return 0


if __name__ == '__main__':
    sys.exit(main())
