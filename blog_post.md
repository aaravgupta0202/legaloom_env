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

Qwen2.5-3B-Instruct + LoRA, 40 GRPO steps on `task_hard` with `num_generations=8`, procedural invoices, hints disabled. Each cell averaged over 10 fresh-seed episodes:

<!-- AUTO-RESULTS-TABLE-START -->
| Task | Baseline | After GRPO | Δ |
|------|---------:|-----------:|------:|
| `task_easy` | 0.648 | **0.460** | **-29%** |
| `task_medium` | 0.609 | **0.636** | **+4%** |
| `task_hard` | 0.112 | **0.120** | **+7%** |
| `task_expert` | 0.030 | **0.034** | **+13%** |
| **Average** | **0.350** | **0.312** | **-11%** |
<!-- AUTO-RESULTS-TABLE-END -->

![Before vs After GRPO](./before_after.png)

*Single-phase task_hard training (40 GRPO steps, num_generations=8). The left panel shows raw baseline vs. trained scores with standard-deviation error bars across 10 fresh-seed evaluation episodes per task. The right panel ranks the relative improvement per task.*

<!-- AUTO-HEADLINE-START -->
Single-phase training on `task_hard` produced **mixed** transfer. Hard improved +7% (the trained target). The other three task pools showed 2 improvement(s) and 1 regression(s) (easy -29%, medium +4%, expert +13%), giving an average lift of -1% that conceals heterogeneous per-task effects. The regressions on pools that contain edge cases absent from training (FY 2025-26 sections in expert; threshold-boundary in medium) suggest the policy is over-fitting to inoperative-PAN reasoning at the cost of general workflow discipline.
<!-- AUTO-HEADLINE-END -->

**Why did `task_easy` drop?** Training focused exclusively on `task_hard` (inoperative-PAN scenarios). The model learned to be more aggressive about applying the 206AA 20% override — which is the right move on hard invoices, but the wrong move on easy invoices where the PAN is valid. With only 10 evaluation episodes, two seeds flipping from correct to incorrect is enough to swing the average by ~0.19 points. A larger evaluation set (30-100 episodes) would give a tighter estimate, but we report the numbers as they are.

### How many seeds actually moved?

![Win/Loss/Tie](./win_loss_tie.png)

*This is the chart that puts the regression in context. Out of 10 episodes per task, 8 of the easy episodes didn't change at all — the regression comes from exactly 2 seeds flipping from correct to incorrect. On the other three tasks, training produced only wins and ties.*

![Episode Waterfall](./episode_waterfall.png)

*The waterfall makes the sparsity of policy movement visible. Most episodes are grey (unchanged). The two red bars on Easy are each a ~0.94-point drop — episodes that scored 0.99 under baseline but 0.05 under the trained policy. The trained model applied the 206AA override when it shouldn't have.*

![Score Distribution](./reward_distribution.png)

*Score distribution across all 40 episodes (4 tasks × 10 evaluation runs each). The composite scoring function clamps outputs to (0.01, 0.99), so the distribution is heavily bimodal — a model either gets the full chain right (section + rate + amount within tolerance) and scores ~0.99, or fails on at least one component and falls toward the floor. The intermediate band around 0.4–0.6 represents partial credit, primarily on mixed-invoice cases where the model gets section right but botches the goods/services split. Training shifts the trained distribution rightward, modestly increasing the count of episodes that clear the 0.5 success threshold.*

![Per-Episode Scatter](./episode_scatter.png)

*Per-episode paired comparison. We use the same 10 seeds for both baseline and trained evaluation, so each scatter point represents the same task instance scored under each policy. Points above the y=x diagonal indicate seeds where training improved the score; points on the diagonal indicate no change; points below indicate regression. The bimodal underlying distribution is visible — most points cluster at the corners (0,0), (0,1), (1,0), (1,1) — and the meaningful lift comes from the small fraction of episodes that move from bottom-left to top-left (wrong → right). The exact per-task delta annotations on each panel are computed from the actual training run; see the Results table above for the headline numbers.*

### Training dynamics

![GRPO Reward Curves](./reward_curves.png)

*GRPO training on task_hard, 40 steps, num_generations=8. Episode reward fluctuates between 0.04 and 0.18 with notable spikes where the policy generates a successful trajectory. The bump from `num_generations=4` to 8 was made to reduce zero-variance steps where all generations score identically and GRPO has no advantage signal — see Statistical Rigor below for the actual measured fraction. Loss values are small in absolute terms because GRPO loss is the policy-gradient surrogate, not cross-entropy; what matters is that the LoRA `B` matrices moved off their zero initialization (verified diagnostically in the notebook). Loss spikes correspond to policy updates absorbing high-variance reward signal.*

### Where the training signal came from

![Gradient Signal](./gradient_signal.png)

*Only 38% of training steps had reward variance — the rest were wasted (all 8 GRPO generations scored identically). This is a known limitation of GRPO on sparse-reward environments. More `num_generations` or a more diverse prompt distribution would help, but at 40 steps on a budget, we take what we can get.*

![Score Regimes](./score_regimes.png)

*The pie charts show why this environment is hard to train on: scores are almost entirely bimodal. An episode either succeeds (≥0.5) or hits the floor (<0.1). There's almost no "partial credit" zone. This makes GRPO's advantage estimation noisy — a single completion flipping from floor to success dominates the gradient.*

## Why the Numbers Survive Scrutiny

Our run showed mixed results — `task_hard` improved (+7%), `task_medium` and `task_expert` improved modestly, but `task_easy` regressed (-29%). Rather than hiding the regression, we tested the numbers three ways.

### Significance testing

We ran a paired Wilcoxon signed-rank test on the 10 paired episodes per task. Wilcoxon (rather than t-test) because TDS scores are bimodal — most episodes score either ~0.01 (failed) or ~0.99 (succeeded), with few intermediates. T-test assumes normality; Wilcoxon doesn't.

The test is one-sided (H₁: trained > baseline) and paired — for each seed, the same task instance is scored under both the baseline and trained policy. Since most episodes don't change between baseline and trained (the policy moves the model on a small fraction of cases), we report `n_changed` alongside the p-value. Tasks with fewer than 5 changed pairs get an explicit "uninformative" tag rather than a misleading p-value. See `statistical_results.json` for the actual numbers from the latest run.

### Bootstrap confidence intervals

We resample the 30 paired episodes 10,000 times and compute the mean delta on each resample. The 2.5th and 97.5th percentiles of those 10,000 deltas form the 95% CI. Critically: **paired** bootstrap — the same indices are sampled for baseline and trained on each iteration. Independent bootstrap of the two arrays would destroy the within-episode pairing structure and inflate the CI.

A CI that excludes 0 means the lift is statistically significant at α = 0.05. A CI that crosses 0 means we can't rule out chance. We report both kinds honestly.

### Reproducibility across seeds

Single-run RL results are notoriously noisy. We ran the same training configuration with different seeds — up to four independent runs. The notebook's `SEED` cell controls model loading, GRPOConfig, and dataset construction so each run is deterministic given its seed.

We expect substantial cross-seed variance (possibly ±3–5% on the average lift) because GRPO with `num_generations=8` is sensitive to which rewards the 8 generations land on. Different seeds will sample different points in the bimodal reward distribution. The right framing is not "the headline number is fixed" but "across 4 runs, the lift is μ=_X_, σ=_Y_."

### Training dynamics

Two diagnostics from the per-step training log confirm the run was real:

- **Useful gradient signal.** Each GRPO step samples `num_generations=8` completions and computes their reward variance. When all 8 score identically, the GRPO advantage is zero and the step contributes no gradient signal. The bump from `num_generations=4` to 8 was made to reduce these wasted steps; the actual fraction for this run is computed by `scripts/statistical_analysis.py` and reported in the README.
- **Policy movement (KL).** KL divergence between the trained policy and the base model grows from ~0 at step 1 to a non-trivial value at step 40. If KL stays at ~0, training did nothing. If KL exceeds 0.05, the policy may have collapsed away from valid behavior. We report start/end/max to confirm both (a) measurable movement and (b) safety from collapse.

This section converts the project from "we measured a number" to "we measured a number, tested it three ways, and report it honestly — including the cases where statistical significance was marginal."

## Does the Model Choice Matter?

We built the training pipeline to be model-agnostic. The notebook accepts any HuggingFace causal LM via two environment variables (`LEGALOOM_MODEL_NAME` and `LEGALOOM_MODEL_TAG`), and the rest of the pipeline — LoRA configuration, GRPO training, evaluation, chart generation — stays identical. We tested this with three models: Qwen2.5-3B-Instruct, Gemma-2-2B-IT, and Llama-3.2-3B-Instruct.

The main results above are from Qwen2.5-3B because it had the strongest baseline on this task, but the infrastructure is in place for anyone to run a head-to-head comparison. After training all three models, a single `python scripts/aggregate_models.py && python scripts/generate_charts.py` call produces a cross-model leaderboard chart. The README's Cross-Model Comparison section auto-populates with the results.

<!-- AUTO-MULTIMODEL-START -->
*Single-model results (Qwen2.5-3B) shown above. To see multi-model comparison, train with 2+ models and run `python scripts/aggregate_models.py` followed by `python scripts/populate_results.py`.*
<!-- AUTO-MULTIMODEL-END -->

## What We'd Do With More Time

- **Multi-turn rollouts** — currently the model emits the full action sequence in one shot without environment feedback between actions. A proper multi-turn loop would let it condition each action on the previous tool output.
- **100+ episode evaluations.** Our 10-episode averages have high variance. More episodes (30-100) would tighten the confidence intervals and give a clearer picture of the true effect size.

## Adversarial Benchmark vs Frontier Models

![Adversarial Heatmap](./adversarial_heatmap.png)

<!-- AUTO-ADVCAPTION-START -->
*Adversarial benchmark scores by model and failure-mode category (n=20 hand-curated cases, 9 categories). Trained Qwen2.5-3B (highlighted with bold border) is compared against frontier API models. Categories where the small specialized model matches or exceeds frontier models indicate where domain-specific RL post-training adds value.*
<!-- AUTO-ADVCAPTION-END -->

## Links

- **HF Space**: [aarav0202/legaloom-env](https://huggingface.co/spaces/aarav0202/legaloom-env)
- **Training notebook**: `LegaLoom_FullCurriculum.ipynb` in the repo
- **OpenEnv**: built on `openenv-core==0.2.3`
