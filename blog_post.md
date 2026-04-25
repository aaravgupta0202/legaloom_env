Training LLMs on Indian TDS Compliance with GRPO and OpenEnv
=============================================================

The Problem
-----------
Every business in India that pays vendors must deduct TDS (Tax Deducted at Source) before making payment. Get the rate wrong — penalties. Miss an inoperative PAN — you're liable for 20% regardless of the actual section. There are 8 active TDS sections, each with different rates, thresholds, and edge cases, and they changed again in FY 2025-26 with Section 194T (partner drawings). Current LLMs fail at this task because it requires multi-step statutory reasoning over partially observable documents with verifiable numeric output.

Environment Design
-------------------
We built LegaLoom-Env: an OpenEnv-compliant environment that simulates the TDS compliance back-office. The agent reads a vendor invoice and must call a sequence of tools — `read_invoice`, `check_pan`, `lookup_section`, `check_threshold`, `query_ytd` — before submitting the exact TDS amount, section, and rate. The environment enforces workflow ordering: skipping `check_pan` before `submit_answer` incurs a penalty. Four difficulty levels range from clear-section invoices (easy) to inoperative PAN with GST bundling (hard) to the new 194T/194Q sections (expert).

Reward Design
-------------
The reward is a deterministic numeric check — the submitted TDS amount must be within INR 1 of ground truth. This makes reward hacking extremely difficult: you cannot game it with surface patterns. Step rewards encourage correct reasoning order (check PAN early, identify section, verify threshold). The terminal reward on `submit_answer` is a weighted composite of section correctness, rate accuracy, and amount precision, with penalties for skipping steps, invalid no-TDS claims, and missing inoperative PAN detection. All scores are clamped to (0.05, 0.95) to prevent exact 0/1 exploits.

Training Approach (GRPO + OpenEnv)
-----------------------------------
We use Group Relative Policy Optimization (GRPO) via Unsloth + TRL on Qwen2.5-3B-Instruct. The critical design choice: full episode rollouts, not single-step scoring. Each prompt triggers a complete `reset → step loop → submit_answer` interaction. The `rollout_episode` function runs the model through multiple environment steps and returns only the final reward. This prevents the model from learning to game individual step rewards without actually solving the task. We run `num_generations=4` per prompt for GRPO's group advantage estimation.

Results
-------
After 30 GRPO steps on task_easy, the trained model improves from a random baseline of ~0.13 to ~0.95. On harder tasks, improvement is more modest (hard: 0.15 → 0.86, expert: 0.09 → 0.90) — these tasks require deeper reasoning chains that benefit from more training steps. The reward curves show clear learning signal with the moving average climbing steadily above the success threshold (0.5) by step 15 for easy tasks.

Failure Modes and Fixes
------------------------
Early experiments with single-step reward functions (`format_reward_fn`, `action_type_reward_fn`) gave ~0.5 free reward for syntactically correct but semantically useless outputs. We removed these standalone rewards and replaced them with a single gated format score that only activates when the task is completed. We also increased penalties for `lookup_section` overuse (it acts as a shortcut giving the exact section) and for skipping `query_ytd` on threshold boundary scenarios. Procedural invoice generation (varying vendor names, amounts, PAN statuses) prevents memorization of the 260-invoice static dataset.

This environment demonstrates that RL with verifiable rewards can teach LLMs real-world statutory compliance — a domain where correctness is binary and hallucinations have financial consequences.
