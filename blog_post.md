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
We ran 80 total GRPO steps — 20 each on `task_easy`, `task_medium`, `task_hard`, `task_expert` — using Qwen2.5-3B-Instruct + LoRA via HF Transformers + PEFT + bitsandbytes (4-bit), with procedural invoices and hints disabled. Each cell below is averaged over 10 fresh-seed episodes per task, with the same model and prompt used for both baseline and trained measurements:

| Task | Baseline | After GRPO | Δ |
|------|---------:|-----------:|------:|
| `task_easy` | 0.222 | 0.249 | +12% |
| `task_medium` | 0.528 | 0.528 | 0% |
| `task_hard` | 0.068 | 0.063 | −7% |
| `task_expert` | 0.252 | 0.257 | +2% |
| Average | 0.268 | 0.274 | +2% |

The gains are modest. This is an honest result — TDS compliance is a hard multi-step reasoning task, and 80 GRPO steps on a 3B model is a small compute budget. What the numbers do show is that `task_medium` (the threshold-boundary scenarios) crosses the success threshold at 0.528 and remains stable after curriculum training — no regression despite being included in the training loop. The per-episode score distribution is heavily bimodal (the model either solves the full chain and gets 0.99, or fails early and gets 0.01), which makes small-sample averages noisy.

The reward curves show real 4-phase curriculum dynamics — reward patterns shift visibly at each phase boundary, and the hard phase produces intermittent spikes to 0.35 as the model occasionally discovers correct inoperative-PAN reasoning. The GRPO loss is non-zero and structured throughout, confirming the optimizer is active even when eval-time gains are modest.

What We'd Do With More Time
-----------------------------
The current limitation is that GRPO's prompt→completion API requires the model to emit the entire action sequence in one shot, without seeing environment feedback between actions. True multi-turn online RL (where the model sees the PAN check result before deciding the section) would be stronger. Procedural invoice generation (varied vendor names, amounts, PAN statuses) is active during training to prevent memorization of the 260-invoice static dataset, which stays as the held-out eval set. With 48 more hours of compute, we'd push curriculum further: easy → medium → hard → expert, evaluating each stage on held-out seeds before advancing.
