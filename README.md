---
title: LegaLoom-Env
emoji: ⚖️
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
tags:
  - openenv
  - reinforcement-learning
  - tax-compliance
  - india
  - tds
  - world-modeling
license: mit
---

# LegaLoom-Env — TDS Compliance RL Environment

**Theme: World Modeling — Professional Tasks (Theme #3.1)**

[![OpenEnv](https://img.shields.io/badge/OpenEnv-0.2.3-blue)](https://github.com/meta-pytorch/OpenEnv)
[![HF Space](https://img.shields.io/badge/HuggingFace-Space-yellow)](https://huggingface.co/spaces/aarav0202/legaloom-env)
[![Tests](https://img.shields.io/badge/tests-32%20passing-green)]()

An OpenEnv-compliant RL environment for training LLMs on Indian TDS (Tax Deducted at Source) compliance — the first of its kind.

---

## The Problem

Every business in India that pays vendors must deduct TDS before making payment. Get the rate wrong — penalties. Get the section wrong — the deduction is disallowed. Miss an inoperative PAN — you're liable for 20% regardless of the actual section rate.

There are **8 active TDS sections**, each with different rates, thresholds, and edge cases — and they changed again in FY 2025-26 with Section 194T (partner drawings) and updated thresholds. A back-office accountant does this dozens of times a day. Current LLMs fail at it because it requires multi-step statutory reasoning over partially observable documents with verifiable numeric output.

> **Capability gap:** Multi-step statutory reasoning over partially observable documents, with deterministic numeric verification.

---

## Environment Overview

The agent reads a vendor invoice and must:

1. Call `read_invoice` to see the invoice text
2. Call `check_pan` to verify the vendor's PAN status (operative vs inoperative)
3. Use tools (`query_ytd`, `lookup_section`, `check_threshold`, `query_law`) to gather evidence
4. Call `submit_answer` with the exact TDS amount, section, and rate

**No hints.** All four difficulty levels require the agent to plan the tool sequence independently — the environment does not suggest the next action. This is real partial observability.

---

## Action Space

| Action | Parameters | Purpose |
|--------|-----------|---------|
| `read_invoice` | none | Fetch invoice text (must be first) |
| `check_pan` | `pan` | Verify vendor PAN: operative or INOPERATIVE |
| `query_ytd` | `pan` | Cumulative YTD payments (required before no_tds claims) |
| `lookup_section` | `description` | Classify service → TDS section |
| `check_threshold` | `section`, `amount` | Verify annual threshold is crossed |
| `query_law` | `section` | Statutory rate, threshold, and exceptions |
| `submit_answer` | `tds_amount_inr`, `section`, `rate_percent` | Final answer |

### Reward Hacking Protections

The environment actively prevents shortcuts that would allow a model to exploit the reward without solving the task:

- **`no_tds=true` requires evidence**: calling `submit_answer` with `no_tds=true` without first calling `query_ytd` incurs a −0.30 penalty. You cannot claim below-threshold without checking YTD.
- **`check_pan` is mandatory**: submitting before `check_pan` is a workflow violation (−0.04, episode continues).
- **`lookup_section` spam**: each call beyond the second costs −0.02.
- **Repeat actions**: repeating the same action incurs diminishing step rewards.
- **Reasoning shortcut detection**: submitting the correct amount but with no evidence trail triggers a shortcut penalty.

---

## 4 Difficulty Levels

| Task | What makes it hard | Hints |
|------|-------------------|-------|
| `task_easy` | Clear section, valid PAN, above threshold | None |
| `task_medium` | Mixed invoice (goods + services) or YTD threshold boundary | None |
| `task_hard` | Inoperative PAN (20% override) or GST-bundled base | None |
| `task_expert` | New FY 2025-26 sections (194T partner, 194Q goods 0.1%) | None |

---

## Reward Structure

Step rewards encourage correct workflow. Terminal reward on `submit_answer` is a **weighted composite**:

```
section_score  (30–40%)  ×  rate_score  (15–30%)  ×  amount_accuracy  (35–65%)
```

All scores clamped to `(0.01, 0.99)` — full 0 and full 1 excluded to keep gradient signal meaningful.

---


## Reward Hacking Audit

Before training, we red-teamed our own reward function and found three exploits. Here's what we found and how we patched each one.

**Exploit 1 — Hint leak (fixed)**
The environment was telling the agent the next correct tool call via the `hint` field in observations. A 3-line agent that just reads the hint string would beat any untrained LLM on easy tasks. The "multi-step reasoning" claim was false.
Fix: `hint_enabled=False` for all four task pools. The agent must read the invoice and plan the tool sequence independently.

**Exploit 2 — Trainer impersonation (fixed)**
The GRPO reward function was injecting `read_invoice` and `check_pan` itself (using ground-truth vendor PAN from `env._task["vendor_pan"]`), then scoring only the model's `submit_answer`. The gradient was teaching "emit submit_answer immediately; the trainer will do the reasoning for you."
Fix: `episode_reward_fn` now parses the model's full action sequence and replays it verbatim. The trainer injects nothing. Only the terminal reward counts.

**Exploit 3 — Evidence-free `no_tds` claim (fixed)**
Submitting `no_tds=true` with `tds_amount_inr=0.0` (claiming below-threshold) scored 0.99 with zero evidence. Any model could achieve this trivially.
Fix: `no_tds=true` without a prior `query_ytd` call incurs a −0.30 penalty. The model must demonstrate it checked the YTD accumulation before claiming below-threshold.

## Results

### Setup

Qwen2.5-3B-Instruct + LoRA (r=16) via Unsloth. **40 GRPO steps total** — 20 on `task_easy` then 20 on `task_hard`. Procedural invoice generation enabled, hints disabled across all four tasks, full episode rollouts (no trainer injection). Both the baseline and trained scores below come from `rollout_episode` in `LegaLoom_QuickTrain.ipynb`, using the `train_grpo.py::ROLLOUT_SYSTEM_PROMPT`. Same model architecture, same prompt, same procedural distribution for both measurements — only the LoRA weights differ. Each cell is the mean of 5 fresh-seed episodes per task.

### Before vs After GRPO

![Before vs After GRPO](./before_after.png)

| Task | Baseline | After GRPO | Δ |
|------|---------:|-----------:|------:|
| `task_easy` | 0.186 | **0.324** | +74% |
| `task_medium` | 0.450 | 0.336 | −25% |
| `task_hard` | 0.078 | **0.126** | +62% |
| `task_expert` | 0.200 | **0.316** | +58% |
| **Average** | **0.229** | **0.276** | **+21%** |

Three of four difficulty levels improved. The largest absolute gain is on `task_easy` (+0.138); the largest relative gain is on `task_hard` (+62%) — the inoperative-PAN scenarios that motivated the project.

### Why `task_medium` regressed

`task_medium` was **not** in the training curriculum. We trained on easy → hard, then evaluated on all four tasks. The policy that learned to detect inoperative PANs (a `task_hard` signal) appears to over-trigger on threshold-boundary scenarios in medium that don't need it, dropping the score from 0.450 to 0.336. With more compute we would interleave medium into the curriculum rather than skip it. We are reporting this as-is — the regression is on the plot and in the table.

### Reward curves

![GRPO Reward Curves](./reward_curves.png)

Both phases show noisy step-reward signal hovering around 0.05–0.20 with intermittent spikes. The curves are not dramatic — 40 steps on a 3B LoRA is a small budget. What changed during training was the underlying behavior the optimizer could grip onto:

- `completion_length` rose from a degenerate ~14 tokens to 43–131 tokens (the model started emitting full action sequences instead of stub completions)
- `reward_std` across each GRPO group rose from 0.000 to 0.260 (genuine variance for the optimizer to exploit)
- Per-step reward spikes to ~0.19 vs a baseline floor of 0.01

The headline reward number staying low is expected at this compute budget. The lift shows up at evaluation time, on held-out seeds — which is what the table above captures.

Raw artifacts: [`training_scores.json`](./training_scores.json), [`training_log.json`](./training_log.json).

---

## Training Pipeline

Training uses GRPO via Unsloth + TRL. The key design decision: **full episode rollouts**, not single-step scoring.

```python
# The model generates a sequence of JSON actions in one completion.
# The reward function replays the full sequence in the environment
# and returns ONLY the final terminal reward from submit_answer.
# The trainer does NOT inject read_invoice or check_pan — the model must.
```

See [`train_grpo.py`](./train_grpo.py) for the full pipeline.

**Why full-episode rewards matter:**
Single-step reward functions train the model to emit syntactically valid JSON and receive a format bonus — not to reason about TDS. Full-episode rewards force the model to produce a complete, coherent action sequence where only the final graded output counts.

---

## Materials

| Resource | Link |
|---------|------|
| 🤗 HuggingFace Space | [aarav0202/legaloom-env](https://huggingface.co/spaces/aarav0202/legaloom-env) |
| 📓 Training Script | [`train_grpo.py`](./train_grpo.py) |
| 📓 Quick Colab Notebook | [`LegaLoom_QuickTrain.ipynb`](./LegaLoom_QuickTrain.ipynb) — ~45 min, 1 phase |
| 📓 Full Curriculum Notebook | [`LegaLoom_FullCurriculum.ipynb`](./LegaLoom_FullCurriculum.ipynb) — ~90 min, 4-phase curriculum, 10 eval episodes |
| 📝 Blog Post | *(link after posting)* |
| 🎬 Demo Video | *(link after recording)* |

---

## Quickstart

```bash
pip install openenv-core==0.2.3 fastapi uvicorn pydantic httpx openai pyyaml

# Run server
uvicorn server.app:app --host 0.0.0.0 --port 7860

# Run baseline inference
export API_BASE_URL=https://api.groq.com/openai/v1
export MODEL_NAME=llama-3.1-8b-instant
export HF_TOKEN=gsk_...
python inference.py
```

```bash
# Docker
docker build -t legaloom-env .
docker run -p 7860:7860 legaloom-env
```

---

## TDS Rules (FY 2025-26)

| Section | Nature | Rate | Threshold |
|---------|--------|------|-----------|
| 194J Professional | Legal, CA, audit, medical (individual/LLP) | 10% | ₹50,000/yr |
| 194J Technical | IT, software, cloud, BPO (company vendor) | 2% | ₹50,000/yr |
| 194C | Security, catering, manpower, events | 2% | ₹30K single / ₹1L annual |
| 194I Building | Office rent, warehouse | 10% | ₹6,00,000/yr |
| 194I Machinery | Equipment, vehicle hire | 2% | ₹6,00,000/yr |
| 194H | Sales commission, brokerage | 2% | ₹20,000/yr |
| **194T** | **Partner salary/drawings (NEW FY25-26)** | 10% | ₹20,000/yr |
| 194Q | Goods >₹50L/yr (buyer turnover >₹10Cr) | 0.1% | ₹50,00,000/yr |
| 206AA | Inoperative PAN override | 20% flat | — |

---

## Project Structure

```
legaloom_env/
├── inference.py                   # Baseline agent
├── train_grpo.py                  # GRPO training pipeline (full episode rollouts)
├── LegaLoom_QuickTrain.ipynb      # Quick run (45 min, 1-phase)
├── LegaLoom_FullCurriculum.ipynb  # Full 4-phase curriculum (90 min)
├── models.py                      # Pydantic typed models
├── openenv.yaml                   # OpenEnv manifest
├── Dockerfile                     # Container
├── server/
│   ├── legaloom_env_environment.py  # Core env (660 lines)
│   ├── graders.py                   # Deterministic composite graders
│   ├── tasks.py                     # 4 tasks, 260-invoice DB
│   ├── tds_rules.py                 # TDS rule engine
│   └── ...
└── tests/ (32 tests)
    ├── test_determinism.py
    ├── test_edge_cases.py
    ├── test_episode_rollout.py
    ├── test_grader_consistency.py
    ├── test_inference_logging.py
    ├── test_reward_hacking.py
    └── test_schema_contract.py
```
