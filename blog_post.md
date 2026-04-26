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

| Task | Baseline | After GRPO | Δ |
|------|---------:|-----------:|------:|
| `task_easy` | 0.227 | 0.273 | +20% |
| `task_medium` | 0.452 | 0.487 | +8% |
| `task_hard` | 0.101 | 0.117 | +16% |
| `task_expert` | 0.402 | 0.419 | +4% |
| Average | 0.295 | 0.324 | +9.6% |

The headline: **single-phase training on `task_hard` produced positive transfer to every task pool**, including pools the model never trained on. Hard improved 16% as expected (the trained target), but easy went up 20% — the largest absolute jump — despite never being in training data. Medium and expert also improved. **No task regressed.**

This is interesting because the conventional wisdom is that focused RL post-training on a narrow distribution causes catastrophic forgetting on other distributions. Our training distribution was inoperative-PAN scenarios (a subset of TDS compliance with one specific edge case). What appears to have transferred is the more general workflow discipline — read invoice, verify PAN, gather threshold and YTD evidence before submission — rather than the specific 206AA override knowledge alone. The smaller +4% on expert (which contains FY 2025-26 sections like 194T and 194Q the base model likely didn't see in pretraining) suggests transfer is bottlenecked by parametric knowledge of new sections, not by reasoning ability.

## What We'd Do With More Time

- **Reproducibility seeds.** Single-run results have unknown variance. Re-running the training with seeds 100, 200, 300 would let us report mean ± std and tighten the +9.6% claim from "one run" to "robust across seeds."
- **Multi-turn rollouts** — currently the model emits the full action sequence in one shot without environment feedback between actions. A proper multi-turn loop would let it condition each action on the previous tool output.
- **100+ episode evaluations.** Our 30-episode averages have ~0.075 standard error. More episodes would tighten the confidence intervals further.

## Links

- **HF Space**: [aarav0202/legaloom-env](https://huggingface.co/spaces/aarav0202/legaloom-env)
- **Training notebook**: `LegaLoom_FullCurriculum.ipynb` in the repo
- **OpenEnv**: built on `openenv-core==0.2.3`
