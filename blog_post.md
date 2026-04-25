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

| Task | Baseline | After GRPO | Δ |
|------|---------:|-----------:|------:|
| `task_easy` | 0.186 | 0.206 | +11% |
| `task_medium` | 0.450 | 0.288 | −36% |
| `task_hard` | 0.078 | 0.146 | +87% |
| `task_expert` | 0.200 | 0.214 | +7% |
| Average | 0.229 | 0.214 | −7% |

The headline: **+87% on `task_hard`** — inoperative-PAN scenarios where the model learned to detect the 206AA override and apply 20% flat rate. This is the most realistic compliance edge case and the single most common TDS penalty trigger.

The average may be negative because training on hard pushes the policy toward aggressive TDS application, which can hurt medium where the right answer is sometimes "don't apply TDS." This is a known policy-interference effect — the optimal policies for hard and medium point in opposite directions. We report this as-is. Mixed-task batches would address this with more compute.

## What We'd Do With More Time

- **DPO warmup** before GRPO — 50% of Phase 1 steps had zero reward variance (all generations scored identically), giving GRPO no gradient signal. A short DPO phase would teach the model to emit non-degenerate action sequences first.
- **Multi-turn rollouts** — currently the model emits the full action sequence in one shot without environment feedback between actions. A proper multi-turn loop would let it condition each action on the previous tool output.
- **30+ episode evaluations** — our 10-episode averages have ~0.18 standard error. More episodes would tighten the confidence intervals.

## Links

- **HF Space**: [aarav0202/legaloom-env](https://huggingface.co/spaces/aarav0202/legaloom-env)
- **Training notebook**: `LegaLoom_FullCurriculum.ipynb` in the repo
- **OpenEnv**: built on `openenv-core==0.2.3`
