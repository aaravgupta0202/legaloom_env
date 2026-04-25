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

**Round 2 Theme: World Modeling — Professional Tasks (#3.1)**

Real interaction with rule-governed APIs (PAN registry, threshold lookup, YTD accumulation), persistent state across multi-step workflows, no shortcuts to ground truth. The agent must maintain consistent internal state, update beliefs based on tool outputs, and orchestrate a multi-step compliance workflow to arrive at a verifiable numeric answer.

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

**No hints.** All four difficulty levels require the agent to plan the tool sequence independently — the environment does not suggest the next action.

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

---

## Reward Hacking Audit

Before training, we red-teamed our own reward function and found three exploits. Each is documented with the fix applied.

**Exploit 1 — Hint leak (fixed)**
The environment was telling the agent the next correct tool call via the `hint` field. A 3-line agent that reads the hint string would beat any untrained LLM. Fix: `hint_enabled=False` for all four task pools. The agent must plan independently.

**Exploit 2 — Trainer impersonation (fixed)**
The GRPO reward function was injecting `read_invoice` and `check_pan` itself (using ground-truth vendor PAN from `env._task["vendor_pan"]`), then scoring only the model's `submit_answer`. Fix: `episode_reward_fn` now parses the model's full action sequence and replays it verbatim. The trainer injects nothing.

**Exploit 3 — Evidence-free `no_tds` claim (fixed)**
Submitting `no_tds=true` with `tds_amount_inr=0.0` scored 0.99 with zero evidence. Fix: `no_tds=true` without a prior `query_ytd` call incurs a −0.30 penalty. The model must check YTD before claiming below-threshold.

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

All scores clamped to `(0.01, 0.99)` — full 0 and full 1 excluded to keep gradient signal meaningful for GRPO.

Additional penalties: `no_tds` on taxable invoice (−0.25), evidence-free `no_tds` (−0.30), missed inoperative PAN (−0.08), shortcut detection, `lookup_section` spam (−0.02 after 2nd call).

---

## Results

### Setup

Qwen2.5-3B-Instruct + LoRA (r=16) via Unsloth. **40 GRPO steps total** — 20 on `task_easy` then 20 on `task_hard`. Procedural invoice generation enabled, hints disabled across all tasks, full episode rollouts (no trainer injection). Both baseline and trained scores come from `rollout_episode` using the same model, same prompt, same procedural distribution — only the LoRA weights differ. Each cell is the mean of 10 fresh-seed episodes per task.

### Before vs After GRPO

![Before vs After GRPO](./before_after.png)

| Task | Baseline | After GRPO | Δ |
|------|---------:|-----------:|------:|
| `task_easy` | 0.186 | **0.206** | +11% |
| `task_medium` | 0.450 | 0.288 | −36% |
| `task_hard` | 0.078 | **0.146** | +87% |
| `task_expert` | 0.200 | **0.214** | +7% |
| **Average** | **0.229** | **0.214** | **−7%** |

**The headline result is `task_hard`: +87% improvement on inoperative-PAN scenarios** — the most realistic compliance edge case. Phase 2 training taught the model to recognize the Section 206AA override and apply the 20% flat rate when PAN is INOPERATIVE. Missing an inoperative PAN is the single most common TDS penalty trigger in real Indian compliance.

### Why `task_medium` regressed

Training on easy + hard pulled the policy toward inoperative-PAN detection, which over-triggers on threshold-boundary scenarios in medium. The average lift is −7% — honestly negative. With more compute and a broader curriculum, we'd expect medium to stabilize.

### Reward curves

![GRPO Reward Curves](./reward_curves.png)

Phase 1 (easy): mean reward 0.075, mostly flat — 50% of steps had `frac_reward_zero_std = 1.0` (all 4 generations scored identically, giving GRPO zero advantage signal). Phase 2 (hard): mean reward 0.088 with spikes to 0.21. Only 5% zero-std steps — the policy found real gradient signal on inoperative-PAN reasoning, which is why `task_hard` improved.

Raw artifacts: [`training_scores.json`](./training_scores.json), [`training_log.json`](./training_log.json).

---

## Known Limitations

- **50% of Phase 1 GRPO steps had zero reward variance** — all 4 generations scored identically. DPO warmup or higher `num_generations` (8 instead of 4) would address this.
- **Training narrowly on easy + hard caused medium to regress.** A broader curriculum with more compute would help.
- **Single-shot completion API** — the model emits the full action sequence in one generation without environment feedback between actions. A multi-turn rollout loop would improve, but doesn't fit TRL's prompt→completion API cleanly.
- **10-episode evaluations have ~0.13 standard error.** A production evaluation would use 30+ episodes per task for tighter confidence.

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

---

## Materials

| Resource | Link |
|---------|------|
| 🤗 HuggingFace Space | [aarav0202/legaloom-env](https://huggingface.co/spaces/aarav0202/legaloom-env) |
| 📓 Training Notebook | [`LegaLoom_FullCurriculum.ipynb`](./LegaLoom_FullCurriculum.ipynb) — 2-phase GRPO, Colab T4 |
| 📓 Training Script | [`train_grpo.py`](./train_grpo.py) |
| 📝 Blog Post | [`blog_post.md`](./blog_post.md) |

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
├── inference.py                      # Baseline agent (Round 1)
├── train_grpo.py                     # GRPO training pipeline
├── LegaLoom_FullCurriculum.ipynb     # Training notebook (Colab T4)
├── models.py                         # Pydantic typed models
├── openenv.yaml                      # OpenEnv manifest
├── Dockerfile                        # Container
├── blog_post.md                      # Project writeup
├── demo_script.md                    # Video script
├── before_after.png                  # Training evidence
├── reward_curves.png                 # Training evidence
├── training_scores.json              # Raw eval numbers
├── training_log.json                 # TRL log_history
├── server/
│   ├── legaloom_env_environment.py   # Core env
│   ├── graders.py                    # Deterministic composite graders
│   ├── tasks.py                      # 4 tasks, 260-invoice DB
│   ├── tds_rules.py                  # TDS rule engine
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
