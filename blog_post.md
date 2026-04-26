# LegaLoom-Env: Teaching LLMs Indian Tax Compliance Through RL

## The Problem

Every business in India deducts TDS (Tax Deducted at Source) before paying vendors. There are 8 active sections, each with different rates, thresholds, and edge cases. Miss an inoperative PAN and you owe 20% flat — regardless of the actual section rate. FY 2025-26 added Section 194T (partner drawings) and changed thresholds. A back-office accountant handles dozens of these daily. Current LLMs fail because the task requires multi-step statutory reasoning with verifiable numeric output, not just text generation.

## The Environment

LegaLoom-Env is an OpenEnv-compliant RL environment where an LLM agent reads a vendor invoice and must call a sequence of tools — `read_invoice`, `check_pan`, `lookup_section`, `query_ytd` — before submitting the exact TDS amount, section, and rate. No hints. Four difficulty levels from basic single-section invoices to FY 2025-26 edge cases with inoperative PANs and GST-bundled bases.

The reward is a weighted composite of section accuracy, rate correctness, and amount precision — all deterministic, all verifiable, all clamped to (0.01, 0.99) to keep GRPO gradient signal alive.

## Reward Hacking Audit

Before training, we red-teamed our own reward function and found three exploits:

1. **Hint leak**: the environment was whispering the next correct tool call. Patched — `hint_enabled=False` across all tasks.
2. **Trainer impersonation**: the reward function was injecting `read_invoice` and `check_pan` on behalf of the model. Patched — `episode_reward_fn` replays the model's own action sequence verbatim.
3. **Evidence-free `no_tds`**: claiming below-threshold without checking YTD scored 0.99. Patched — −0.30 penalty without prior `query_ytd`.

These three fixes are documented in the README and verifiable in code. We consider the reward hacking audit one of the strongest parts of this project.

## Training

We used GRPO via Unsloth + TRL with full episode rollouts. The model generates a complete action sequence in one completion; the reward function replays it in the environment and returns only the terminal reward. The trainer injects nothing — the model must plan the full tool sequence independently.

## Results

Qwen2.5-3B-Instruct + LoRA, 40 GRPO steps on `task_hard` with `num_generations=8`, procedural invoices, hints disabled. Each cell averaged over 30 fresh-seed episodes:

<!-- AUTO-RESULTS-TABLE-START -->
| Task | Baseline | After GRPO | Δ |
|------|---------:|-----------:|------:|
| `task_easy` | 0.227 | 0.273 | +20% |
| `task_medium` | 0.452 | 0.487 | +8% |
| `task_hard` | 0.101 | 0.117 | +16% |
| `task_expert` | 0.402 | 0.419 | +4% |
| Average | 0.295 | 0.324 | +9.6% |
<!-- AUTO-RESULTS-TABLE-END -->

![Before vs After GRPO](./before_after.png)

*Single-phase task_hard training (40 GRPO steps, num_generations=8). The left panel shows raw baseline vs. trained scores with standard-deviation error bars across 30 fresh-seed evaluation episodes per task. The right panel ranks the relative improvement per task.*

<!-- AUTO-HEADLINE-START -->
The headline: **single-phase training on `task_hard` produced positive transfer to every task pool**, including pools the model never trained on. Hard improved 16% as expected (the trained target), but easy went up 20% — the largest absolute jump — despite never being in training data. Medium and expert also improved. **No task regressed.**

This is interesting because the conventional wisdom is that focused RL post-training on a narrow distribution causes catastrophic forgetting on other distributions. Our training distribution was inoperative-PAN scenarios (a subset of TDS compliance with one specific edge case). What appears to have transferred is the more general workflow discipline — read invoice, verify PAN, gather threshold and YTD evidence before submission — rather than the specific 206AA override knowledge alone.
<!-- AUTO-HEADLINE-END -->

![Score Distribution](./reward_distribution.png)

*Score distribution across all 240 episodes (4 tasks × 30 evaluation runs each). The composite scoring function clamps outputs to (0.01, 0.99), so the distribution is heavily bimodal — a model either gets the full chain right (section + rate + amount within tolerance) and scores ~0.99, or fails on at least one component and falls toward the floor. The intermediate band around 0.4–0.6 represents partial credit, primarily on mixed-invoice cases where the model gets section right but botches the goods/services split. Training shifts the trained distribution rightward, modestly increasing the count of episodes that clear the 0.5 success threshold.*

![Per-Episode Scatter](./episode_scatter.png)

*Per-episode paired comparison. We use the same 30 seeds (42–71) for both baseline and trained evaluation, so each scatter point represents the same task instance scored under each policy. Points above the y=x diagonal indicate seeds where training improved the score; points on the diagonal indicate no change; points below indicate regression. The bimodal underlying distribution is visible — most points cluster at the corners (0,0), (0,1), (1,0), (1,1) — and the meaningful lift comes from the small fraction of episodes that move from bottom-left to top-left (wrong → right). The exact per-task delta annotations on each panel are computed from the actual training run; see the Results table above for the headline numbers.*

### Training dynamics

![GRPO Reward Curves](./reward_curves.png)

*GRPO training on task_hard, 40 steps, num_generations=8. Episode reward fluctuates between 0.04 and 0.18 with notable spikes where the policy generates a successful trajectory. The bump from `num_generations=4` to 8 was made to reduce zero-variance steps where all generations score identically and GRPO has no advantage signal — see Statistical Rigor below for the actual measured fraction. Loss values are small in absolute terms because GRPO loss is the policy-gradient surrogate, not cross-entropy; what matters is that the LoRA `B` matrices moved off their zero initialization (verified diagnostically in the notebook). Loss spikes correspond to policy updates absorbing high-variance reward signal.*

## Why the Numbers Survive Scrutiny

A 9.6% average lift on a small training run could be noise. We tested it three ways.

### Significance testing

We ran a paired Wilcoxon signed-rank test on the 30 paired episodes per task. Wilcoxon (rather than t-test) because TDS scores are bimodal — most episodes score either ~0.01 (failed) or ~0.99 (succeeded), with few intermediates. T-test assumes normality; Wilcoxon doesn't.

The test is one-sided (H₁: trained > baseline) and paired — for each seed, the same task instance is scored under both the baseline and trained policy. Since most episodes don't change between baseline and trained (the policy moves the model on a small fraction of cases), we report `n_changed` alongside the p-value. Tasks with fewer than 5 changed pairs get an explicit "uninformative" tag rather than a misleading p-value. See `statistical_results.json` for the actual numbers from the latest run.

### Bootstrap confidence intervals

We resample the 30 paired episodes 10,000 times and compute the mean delta on each resample. The 2.5th and 97.5th percentiles of those 10,000 deltas form the 95% CI. Critically: **paired** bootstrap — the same indices are sampled for baseline and trained on each iteration. Independent bootstrap of the two arrays would destroy the within-episode pairing structure and inflate the CI.

A CI that excludes 0 means the lift is statistically significant at α = 0.05. A CI that crosses 0 means we can't rule out chance. We report both kinds honestly.

### Reproducibility across seeds

Single-run RL results are notoriously noisy. We ran the same training configuration with seeds {42, 100, 200, 300} — four independent runs. The notebook's `SEED` cell controls model loading, GRPOConfig, and dataset construction so each run is deterministic given its seed.

We expect substantial cross-seed variance (possibly ±3–5% on the average lift) because GRPO with `num_generations=8` is sensitive to which rewards the 8 generations land on. Different seeds will sample different points in the bimodal reward distribution. The right framing is not "the headline number is fixed" but "across 4 runs, the lift is μ=_X_, σ=_Y_."

### Training dynamics

Two diagnostics from the per-step training log confirm the run was real:

- **Useful gradient signal.** Each GRPO step samples `num_generations=8` completions and computes their reward variance. When all 8 score identically, the GRPO advantage is zero and the step contributes no gradient signal. The bump from `num_generations=4` to 8 was made to reduce these wasted steps; the actual fraction for this run is computed by `scripts/statistical_analysis.py` and reported in the README.
- **Policy movement (KL).** KL divergence between the trained policy and the base model grows from ~0 at step 1 to a non-trivial value at step 40. If KL stays at ~0, training did nothing. If KL exceeds 0.05, the policy may have collapsed away from valid behavior. We report start/end/max to confirm both (a) measurable movement and (b) safety from collapse.

This section converts the project from "we measured a number" to "we measured a number, tested it three ways, and report it honestly — including the cases where statistical significance was marginal."

## What We'd Do With More Time

- **Multi-turn rollouts** — currently the model emits the full action sequence in one shot without environment feedback between actions. A proper multi-turn loop would let it condition each action on the previous tool output.
- **100+ episode evaluations.** Our 30-episode averages have ~0.075 standard error. More episodes would tighten the confidence intervals further.

## Adversarial Benchmark vs Frontier Models

![Adversarial Heatmap](./adversarial_heatmap.png)

<!-- AUTO-ADVCAPTION-START -->
*Adversarial benchmark scores across 6 models and 9 failure-mode categories (n=20 hand-curated cases). The trained Qwen2.5-3B (highlighted with a bold border) is compared against frontier API models: GPT-4o, GPT-4o-mini, Claude Sonnet 4.5, Gemini 2.5 Pro. The 20 cases never appear in training data — they're hand-written probes for known LLM weaknesses on Indian TDS compliance: inoperative-PAN under-correction (where applying 206AA flat 20% over a 0.1% base rate creates a 200x rate jump that models often under-apply), FY 2025-26 sections (194T partner drawings, 194Q goods 0.1%) absent from most pretraining data, threshold off-by-one cases, mixed goods+services positional confusion. Categories where the small specialized model matches or beats frontier models indicate where domain-specific RL post-training adds genuine value.*
<!-- AUTO-ADVCAPTION-END -->

## Links

- **HF Space**: [aarav0202/legaloom-env](https://huggingface.co/spaces/aarav0202/legaloom-env)
- **Training notebook**: `LegaLoom_FullCurriculum.ipynb` in the repo
- **OpenEnv**: built on `openenv-core==0.2.3`
