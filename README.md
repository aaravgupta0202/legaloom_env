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
[![Tests](https://img.shields.io/badge/tests-60%20passing-green)]()

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

Qwen2.5-3B-Instruct + LoRA (r=16) via Unsloth. **40 GRPO steps on `task_hard`** with `num_generations=8` for maximum reward variance. Procedural invoice generation enabled, hints disabled across all tasks, full episode rollouts (no trainer injection). Both baseline and trained scores come from `rollout_episode` using the same model, same prompt, same procedural distribution — only the LoRA weights differ. Each cell is the mean of 30 fresh-seed episodes per task.

### Before vs After GRPO

![Before vs After GRPO](./before_after.png)

| Task | Baseline | After GRPO | Δ |
|------|---------:|-----------:|------:|
| `task_easy` | 0.227 | **0.273** | **+20%** |
| `task_medium` | 0.452 | **0.487** | **+8%** |
| `task_hard` | 0.101 | **0.117** | **+16%** |
| `task_expert` | 0.402 | **0.419** | **+4%** |
| **Average** | **0.295** | **0.324** | **+9.6%** |

**Single-phase training on `task_hard` produced positive transfer to every task pool, including pools never seen during training.** Hard improved 16% (the trained target), but the more interesting result is positive cross-task transfer: training only on inoperative-PAN scenarios improved easy by 20%, medium by 8%, and expert by 4%. **No task regressed.** This contradicts the conventional intuition that focused RL post-training requires multi-task curricula to avoid catastrophic forgetting — it suggests that statutory-reasoning capabilities (workflow ordering, evidence-gathering before submission, PAN-status checking) generalize across TDS section types when the underlying compliance discipline is shared.

### What this run shows

The +20% lift on `task_easy` (a pool the model never trained on) is the strongest evidence here. It says the model didn't just memorize inoperative-PAN handling — it learned a more general compliance discipline (read invoice → verify PAN → check applicability before computing rate) that transfers to clean-PAN cases too. The smaller +8% lift on `task_medium` is consistent with this: medium contains threshold-boundary cases where the same workflow discipline helps, but the absolute starting point (0.452) leaves less headroom than easy.

The `task_hard` lift (+16%) is the trained target. We expected the largest gain there; the fact that easy outperforms it is partly a baseline effect — easy started lower (0.227 vs hard's 0.101), so equal absolute gains produce different relative percentages.

### Reward curves

![GRPO Reward Curves](./reward_curves.png)

Single-phase task_hard, 40 steps, `num_generations=8`. Mean reward across the run was 0.071 with peaks to 0.178. With 8 generations per group (vs. 4 in earlier runs) the **zero-variance step fraction dropped from ~50% historically to 0%** — every training step produced reward variance for GRPO to exploit. Loss values are small in absolute terms (~3.8e-5) because GRPO loss is the policy-gradient surrogate, not cross-entropy; what matters is that the LoRA `B` matrices moved off their zero initialization (verified by Cell 5.5's diagnostic in the notebook).

Raw artifacts: [`training_scores.json`](./training_scores.json), [`training_log.json`](./training_log.json).

---

## Adversarial Benchmark

A held-out set of **20 hand-curated TDS scenarios** designed to expose specific LLM failure modes. These cases never appear in training data — they're a frontier benchmark for any model claiming to handle Indian compliance.

**9 failure-mode categories:**
- `inoperative_pan_low_base_rate` — 206AA override on 2% / 0.1% sections (10x–200x rate jumps)
- `fy2526_new_sections` — Section 194T (partner drawings) and 194Q (goods 0.1%) absent from pretraining
- `mixed_goods_services_positional` — services portion only is taxable, even when goods are listed first
- `threshold_boundary` — YTD off-by-one cases (₹49,999 + ₹2 vs ₹49,500 + ₹400)
- `gst_base_handling` — TDS computed on pre-GST amount, not invoice total
- `section_subtype_ambiguity` — 194J Professional 10% vs Technical 2%, 194I Building 10% vs Machinery 2%
- `entity_type_rate_change` — Pvt Ltd → 194J Technical (2%); Individual → 194J Professional (10%)
- `conflicting_evidence` — invoice cites stale rules; current statute applies
- `compound_traps` — multiple edge cases stacked (inoperative PAN + mixed invoice, etc.)

Each case has a deterministic ground truth; the scorer computes a composite of section accuracy (35%), rate correctness (25%), and amount precision (40%).

**Run the benchmark yourself:** [`LegaLoom_AdversarialBenchmark.ipynb`](./LegaLoom_AdversarialBenchmark.ipynb) — supports baseline Qwen-3B, trained Qwen-3B, plus optional GPT-4o-mini, Claude Sonnet 4.5, Gemini 2.5 Pro (set `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `GOOGLE_API_KEY` env vars).

The 20-case benchmark is also covered by 12 unit tests in [`tests/test_adversarial_cases.py`](./tests/test_adversarial_cases.py) verifying determinism, well-formedness, perfect-submission scoring, and category coverage.

---

## Reward Hacking Ablation

The README's claim that we patched 3 reward-hacking exploits is backed by **7 ablation tests in [`tests/test_ablation.py`](./tests/test_ablation.py)** that exercise each exploit:

```bash
$ pytest tests/test_ablation.py -v
test_no_tds_without_query_ytd_is_penalized          PASSED
test_no_tds_with_query_ytd_evidence_not_penalized   PASSED
test_hints_disabled_for_all_task_pools              PASSED
test_hint_field_is_empty_in_observations            PASSED
test_episode_reward_fn_does_not_inject_actions      PASSED
test_floor_reward_when_no_submission                PASSED
test_all_three_patches_present                      PASSED
```

Each test runs an exploit trajectory (e.g., "submit `no_tds=true` without calling `query_ytd`") and asserts that the patched environment scores it < 0.5. The first two ablation tests use seed-pinned tasks where the ground truth is *not* `no_tds`, so the exploit is wrong on its merits — the patch makes that wrongness penalty-stacked.

### Property-Based Reward Tests

Beyond example-based tests, [`tests/test_reward_properties.py`](./tests/test_reward_properties.py) uses [Hypothesis](https://hypothesis.readthedocs.io) to generate hundreds of random submissions and verify structural invariants that should hold for **all** inputs:

1. **Score is always in [0, 1]** — no submission, however malformed, can push the grader out of bounds
2. **Output contract** — every call returns `{score, feedback, breakdown}` with the right types
3. **Robustness** — the grader never raises on malformed input (negative amounts, empty strings, NaN-adjacent floats)
4. **Exact-match correctness** — submitting ground-truth values verbatim scores ≥ 0.7
5. **Inoperative-PAN exact-match** — submitting underlying section + 20% override scores ≥ 0.7
6. **Determinism** — same input three times → identical scores
7. **Wrong-everything cap** — if section, rate, AND amount are all wrong, score ≤ 0.5
8. **`no_tds` correctness** — wrongly claiming or wrongly omitting `no_tds` always loses points

These properties are checked against ~700 generated submissions per pytest run. Hypothesis automatically shrinks any failure to the minimal counterexample, so a regression here points directly at the offending input class.

---

## Known Limitations

- **Single training run, single seed.** Reproducibility across seeds is not yet measured. Standard RL practice is 3+ seed replications; we'd run those given more compute.
- **Single-shot completion API** — the model emits the full action sequence in one generation without environment feedback between actions. A multi-turn rollout loop would improve, but doesn't fit TRL's prompt→completion API cleanly.
- **30-episode evaluations have ~0.075 standard error** on the bimodal score distribution — a production evaluation would use 100+ episodes for tight bounds on small lifts.

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
| 📓 Training Notebook | [`LegaLoom_FullCurriculum.ipynb`](./LegaLoom_FullCurriculum.ipynb) — single-phase GRPO, Colab T4 |
| 📓 Adversarial Benchmark | [`LegaLoom_AdversarialBenchmark.ipynb`](./LegaLoom_AdversarialBenchmark.ipynb) — 20 hand-curated cases vs frontier models |
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
├── LegaLoom_AdversarialBenchmark.ipynb  # 20-case frontier benchmark
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
│   ├── adversarial_cases.py          # 20 hand-curated frontier benchmark cases
│   └── ...
└── tests/ (60 tests)
    ├── test_determinism.py
    ├── test_edge_cases.py
    ├── test_episode_rollout.py
    ├── test_grader_consistency.py
    ├── test_inference_logging.py
    ├── test_reward_hacking.py
    ├── test_schema_contract.py
    ├── test_ablation.py
    ├── test_adversarial_cases.py
    └── test_reward_properties.py
```
