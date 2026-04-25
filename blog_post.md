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
Training runs onsite with HuggingFace compute credits. Baseline: untrained Qwen2.5-3B-Instruct via `rollout_episode` (same model, same prompt as training, no LoRA). After 20 GRPO steps on task_easy then 20 steps on task_hard (procedural invoices, no hints, no worked examples in the prompt):

*[Reward curves and before/after scores to be added after the training run completes.]*

We expect a modest lift on easy and medium tasks, with limited improvement on expert (194T/194Q sections are underrepresented in the base model's pretraining). A real, noisy improvement on hard tasks — particularly inoperative PAN detection — is the core claim. The curve will be committed to this repo as `reward_curves.png`.

What We'd Do With More Time
-----------------------------
The current limitation is that GRPO's prompt→completion API requires the model to emit the entire action sequence in one shot, without seeing environment feedback between actions. True multi-turn online RL (where the model sees the PAN check result before deciding the section) would be stronger. Procedural invoice generation (varied vendor names, amounts, PAN statuses) is active during training to prevent memorization of the 260-invoice static dataset, which stays as the held-out eval set. With 48 more hours of compute, we'd push curriculum further: easy → medium → hard → expert, evaluating each stage on held-out seeds before advancing.
