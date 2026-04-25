Training LLMs on Indian TDS Compliance with GRPO and OpenEnv
=============================================================

The Problem
-----------
Every business in India that pays vendors must deduct TDS (Tax Deducted at Source) before making payment. Get the rate wrong — penalties. Miss an inoperative PAN — you're liable for 20% regardless of the actual section. There are 8 active TDS sections, each with different rates, thresholds, and edge cases, and they changed again in FY 2025-26 with Section 194T (partner drawings). Current LLMs struggle with this task because it requires multi-step statutory reasoning over partially observable documents with verifiable numeric output — not a pattern that shows up much in pretraining data.

Environment Design
------------------
We built LegaLoom-Env: an OpenEnv-compliant environment that simulates the TDS compliance back-office. The agent reads a vendor invoice and must call a sequence of tools — `read_invoice`, `check_pan`, `lookup_section`, `check_threshold`, `query_ytd` — before submitting the exact TDS amount, section, and rate. **No hints are given at any difficulty level** — the agent must plan the tool sequence from the invoice text alone. Four difficulty levels range from clear-section invoices (easy) to inoperative PAN with GST bundling (hard) to the new 194T/194Q sections (expert).

Reward Design and Anti-Hacking
--------------------------------
The terminal reward is a deterministic numeric check — submitted TDS amount must be within INR 1 of ground truth. We red-teamed our own reward function and found three exploits before training:

1. **Hint leak** — the env was whispering the next correct tool call. Fixed: hints disabled on all difficulty levels.
2. **Trainer impersonation** — the GRPO reward function was injecting `read_invoice` and `check_pan` itself, then scoring the model's `submit_answer`. The model never had to plan. Fixed: the model must emit the full action sequence; the trainer only executes it and scores the terminal state.
3. **`no_tds` shortcut** — submitting `no_tds=true` (below threshold) without evidence scored 0.99 with no reasoning. Fixed: `query_ytd` must be called before `no_tds=true` is honored; otherwise a −0.30 penalty applies.

All scores clamped to (0.01, 0.99) to keep reward signal meaningful.

Training Approach (GRPO + OpenEnv)
------------------------------------
We use Group Relative Policy Optimization (GRPO) via Unsloth + TRL on Qwen2.5-3B-Instruct. The design: **full episode rollouts, not single-step scoring**. Each prompt asks the model to generate the complete action sequence. The reward function replays that sequence in the environment and returns only the final terminal reward from `submit_answer`. This prevents the model from gaming per-step rewards without solving the task.

Results
-------
We ran 40 total GRPO steps — 20 on `task_easy` then 20 on `task_hard` — using Qwen2.5-3B-Instruct + LoRA via Unsloth, with procedural invoices and hints disabled. Each cell below is averaged over 5 fresh-seed episodes per task, with the same model and prompt used for both baseline and trained measurements:

| Task | Baseline | After GRPO | Δ |
|------|---------:|-----------:|------:|
| `task_easy` | 0.186 | 0.324 | +74% |
| `task_medium` | 0.450 | 0.336 | −25% |
| `task_hard` | 0.078 | 0.126 | +62% |
| `task_expert` | 0.200 | 0.316 | +58% |
| Average | 0.229 | 0.276 | +21% |

Three of four difficulty tiers improved. The largest relative gain is on `task_hard` (+62%) — the inoperative-PAN scenarios that motivated the project. `task_medium` regressed because it was not in the training curriculum: we trained on easy → hard, and the inoperative-PAN signal appears to over-trigger on threshold-boundary scenarios that don't need it. We are reporting this as-is rather than dropping medium from the table; with more compute we'd interleave it into the curriculum.

The reward curves themselves are noisy — 40 steps on a 3B LoRA is a small budget, and step rewards hover around 0.05–0.20. What changed during training was the underlying behavior the optimizer could grip onto: completion length rose from a degenerate 14 tokens to 43–131 tokens (the model started emitting full action sequences instead of stubs), and reward variance across the GRPO group went from 0.000 to 0.260. Those are the signals GRPO needs. The headline lift then shows up at eval time on held-out seeds, which is what the table captures.

What We'd Do With More Time
-----------------------------
The current limitation is that GRPO's prompt→completion API requires the model to emit the entire action sequence in one shot, without seeing environment feedback between actions. True multi-turn online RL (where the model sees the PAN check result before deciding the section) would be stronger. Procedural invoice generation (varied vendor names, amounts, PAN statuses) is active during training to prevent memorization of the 260-invoice static dataset, which stays as the held-out eval set. With 48 more hours of compute, we'd push curriculum further: easy → medium → hard → expert, evaluating each stage on held-out seeds before advancing.
