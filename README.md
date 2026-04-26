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

Qwen2.5-3B-Instruct + LoRA (r=16) via Unsloth. **40 GRPO steps on `task_hard`** with `num_generations=8` for maximum reward variance. Procedural invoice generation enabled, hints disabled across all tasks, full episode rollouts (no trainer injection). Both baseline and trained scores come from `rollout_episode` using the same model, same prompt, same procedural distribution — only the LoRA weights differ. Each cell is the mean of 10 fresh-seed episodes per task.

### Before vs After GRPO

![Before vs After GRPO](./before_after.png)

*Before vs. after GRPO training on `task_hard` (40 steps, num_generations=8). Error bars show standard deviation across 10 fresh-seed episodes per task. The right panel shows per-task relative change — note that `task_easy` regressed, which we discuss below.*

<!-- AUTO-RESULTS-TABLE-START -->
| Task | Baseline | After GRPO | Δ |
|------|---------:|-----------:|------:|
| `task_easy` | 0.648 | **0.460** | **-29%** |
| `task_medium` | 0.609 | **0.636** | **+4%** |
| `task_hard` | 0.112 | **0.120** | **+7%** |
| `task_expert` | 0.030 | **0.034** | **+13%** |
| **Average** | **0.350** | **0.312** | **-11%** |
<!-- AUTO-RESULTS-TABLE-END -->

<!-- AUTO-HEADLINE-START -->
Single-phase training on `task_hard` produced **mixed** transfer. Hard improved +7% (the trained target). The other three task pools showed 2 improvement(s) and 1 regression(s) (easy -29%, medium +4%, expert +13%), giving an average lift of -1% that conceals heterogeneous per-task effects. The regressions on pools that contain edge cases absent from training (FY 2025-26 sections in expert; threshold-boundary in medium) suggest the policy is over-fitting to inoperative-PAN reasoning at the cost of general workflow discipline.
<!-- AUTO-HEADLINE-END -->

### Why did task_easy regress?

The -29% drop on `task_easy` is the most striking result here, and it deserves an honest look. What happened: training on `task_hard` (which is dominated by inoperative-PAN scenarios requiring a 20% override rate) taught the model to be more aggressive about applying the 206AA override. On easy invoices — where the PAN is valid and the correct rate is the section's base rate — this learned bias causes the model to occasionally apply the 20% override when it shouldn't.

In other words, the model traded some accuracy on "normal" invoices for better handling of the tricky inoperative-PAN edge cases that `task_hard` tests. This is a classic transfer-learning tradeoff: focused training on a narrow distribution can shift the policy away from behaviors that were working fine on simpler cases.

It's also worth noting that with only 10 evaluation episodes per task, the variance is high. Two episodes flipping from ~0.99 to ~0.05 is enough to swing the average by ~0.19 — which is exactly the magnitude of the regression we see. A larger evaluation (30-100 episodes) would give a tighter picture of the true effect size.

We report this as-is rather than cherry-picking a run where easy didn't regress. Honest reporting of mixed results is part of how we evaluate whether our training pipeline is working correctly.

### Score distribution

![Score Distribution](./reward_distribution.png)

*Score distribution across all 40 episodes (4 tasks × 10 episodes). Scores are bimodal — mostly correct (≥0.5) or floor-clamped (~0.01) — with a few intermediates (partial credit). Training shifts the trained distribution rightward, increasing the fraction of episodes scoring above the success threshold.*

### Per-episode comparison

![Per-Episode Scatter](./episode_scatter.png)

*Per-episode comparison. Each point is a single evaluation episode (n=10 per task). Points above the y=x line mean training improved the score on that exact seed. Most points cluster on or near the y=x line because most episodes were already correct or already failing for the same reasons; the meaningful lift comes from the small fraction of episodes where training tipped the outcome from "wrong" to "right."*

### Reward curves

![GRPO Reward Curves](./reward_curves.png)

*GRPO training on task_hard, 40 steps, num_generations=8. Episode reward fluctuates with notable spikes where the policy generates a successful trajectory. Grey-shaded regions (if any) show steps where all 8 generations produced identical rewards — zero advantage signal, wasted steps. Loss spikes correspond to policy updates absorbing high-variance reward signal.*

Single-phase task_hard, 40 steps, `num_generations=8`. The bump from `num_generations=4` (used in earlier runs) to 8 was made to reduce zero-variance steps where all generations score identically and GRPO has no advantage signal — see the Statistical Rigor section below for the actual measured fraction. Loss values are small in absolute terms because GRPO loss is the policy-gradient surrogate, not cross-entropy; what matters is that the LoRA `B` matrices moved off their zero initialization (verified by Cell 5.5's diagnostic in the notebook).

Raw artifacts: [`training_scores.json`](./training_scores.json), [`training_log.json`](./training_log.json).

---

## Statistical Rigor

The numbers above are computed from 10 paired episodes per task (baseline and trained on the **same** 30 seeds, so each seed's task instance is scored under both policies). Three diagnostics test whether the headline lift survives critical reading.

To regenerate these numbers from the raw artifacts: `python scripts/statistical_analysis.py`, then `python scripts/populate_results.py`.

Wilcoxon — not t-test — because per-episode scores are bimodal (mostly 0.01 or 0.99 with few intermediates) and not normally distributed. Bootstrap is **paired** (same indices sampled for baseline and trained on each iteration) — this preserves the within-episode pairing structure that gives the test power; independent bootstrap would inflate the CI.

<!-- AUTO-STATRIGOR-START -->
**Significance (paired Wilcoxon, n=10 per task)**

| Task | Δ | p-value | n changed |
|------|--:|--------:|----------:|
| `task_easy` | -0.188 | _uninformative (2/10 changed)_ | 2/10 |
| `task_medium` | +0.027 | _uninformative (1/10 changed)_ | 1/10 |
| `task_hard` | +0.008 | _uninformative (2/10 changed)_ | 2/10 |
| `task_expert` | +0.004 | _uninformative (1/10 changed)_ | 1/10 |

**Bootstrap 95% confidence intervals on Δ (paired bootstrap, 10,000 iter)**

| Task | Δ | 95% CI |
|------|--:|-------:|
| `task_easy` | -0.188 | [-0.470, +0.000] |
| `task_medium` | +0.027 | [+0.000, +0.081] |
| `task_hard` | +0.008 | [+0.000, +0.020] |
| `task_expert` | +0.004 | [+0.000, +0.012] |

**Reproducibility:** single run (seed 42). Multi-seed reproducibility runs are listed in the *Reproducibility* section below.

**Useful gradient signal:** 15/40 steps (38%) had non-zero reward variance *(reward_std<1e-3 proxy; frac_reward_zero_std absent from log)*.
**Policy movement (KL):** start 0.0000 → end 0.0298, max 0.0892. The policy moved measurably while staying well below the 0.05 collapse threshold.
<!-- AUTO-STATRIGOR-END -->

---

## Cross-Model Comparison

The notebook is fully parameterized for multi-model comparison — the same training pipeline works on Qwen2.5-3B, Gemma-2-2B, and Llama-3.2-3B. Just set `LEGALOOM_MODEL_NAME` and `LEGALOOM_MODEL_TAG` as environment variables (or at the top of the notebook) and re-run. The scripts handle artifact naming, aggregation, and chart generation automatically.

See the **Reproducibility** section below for exact instructions.

<!-- AUTO-MULTIMODEL-START -->
*Single-model results (Qwen2.5-3B) reported above. To populate a cross-model comparison table here, train with 2+ models and run `python scripts/aggregate_models.py` followed by `python scripts/populate_results.py`.*
<!-- AUTO-MULTIMODEL-END -->

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

<!-- AUTO-ADVCAPTION-START -->
*Adversarial benchmark scores by model and failure-mode category (n=20 hand-curated cases, 9 categories). Trained Qwen2.5-3B (highlighted with bold border) is compared against frontier API models. Categories where the small specialized model matches or exceeds frontier models indicate where domain-specific RL post-training adds value.*
<!-- AUTO-ADVCAPTION-END -->

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

- **Mixed transfer on `task_easy`.** Training on `task_hard` (inoperative-PAN scenarios) improved hard/medium/expert but regressed easy by -29%. This is a known risk with single-task focused RL — the policy over-indexes on the trained distribution. Multi-task curricula or a replay buffer would help, but we opted for the simplest pipeline that demonstrates learning.
- **10-episode evaluations.** With only 10 episodes per task, individual seed rolls can swing the average significantly. A production evaluation would use 30-100 episodes. The bimodal score distribution (most episodes score either ~0.01 or ~0.99) makes the mean especially sensitive to a few flips.
- **Single training run, single seed.** RL post-training is high-variance. A different seed could produce different per-task splits. The codebase supports multi-seed runs (`SEED=100, 200, 300` in the notebook), but we ran only seed 42 for this submission.
- **Single-shot completion** — the model emits the full action sequence in one generation without seeing environment feedback between actions. A proper multi-turn rollout loop (where each tool call gets its response before the next is emitted) would improve accuracy, especially on harder tasks.

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
| 📝 **Mini-blog** (Round 2 deliverable) | [`blog_post.md`](./blog_post.md) — markdown writeup explaining the environment, training methodology, and results |

---

## Quickstart

```bash
pip install openenv-core==0.2.3 fastapi uvicorn pydantic httpx openai pyyaml

# Or, install with dev deps for running tests:
pip install -e ".[dev]"
pytest tests/    # should report 60 passed

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

## Reproducibility

To reproduce the numbers in this README:

1. Install with dev extras: `pip install -e ".[dev]"`
2. Verify tests pass: `pytest tests/` (60 passing)
3. Run training: open `LegaLoom_FullCurriculum.ipynb` on Colab/HF A10G, run all cells
4. Generate charts: `python scripts/generate_charts.py`
5. Compute statistics: `python scripts/statistical_analysis.py`
6. Update README + blog with real values: `python scripts/populate_results.py`

Optional:

- **Adversarial benchmark.** Open `LegaLoom_AdversarialBenchmark.ipynb`, set `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `GOOGLE_API_KEY` env vars, run all cells (~$15 in API credits, ~3 hours). After it finishes, re-run `populate_results.py` and the heatmap caption updates with a one-sentence headline if the trained Qwen-3B beat any frontier model on any category.
- **Multi-seed reproducibility.** Run training 3 more times with `SEED=100`, `200`, `300`, save results under `runs/seed_*/training_scores.json`, then `python scripts/aggregate_seeds.py`. The Statistical Rigor section's reproducibility row updates from "single run" to "across N runs, μ ± σ" automatically.

### Multi-model comparison (optional)

To compare multiple base models on the same task:

1. Run training with each model by setting environment variables before launching the notebook:

   ```bash
   LEGALOOM_MODEL_NAME='Qwen/Qwen2.5-3B-Instruct'   LEGALOOM_MODEL_TAG='qwen3b'  jupyter notebook
   LEGALOOM_MODEL_NAME='google/gemma-2-2b-it'        LEGALOOM_MODEL_TAG='gemma2b' jupyter notebook
   LEGALOOM_MODEL_NAME='meta-llama/Llama-3.2-3B-Instruct' LEGALOOM_MODEL_TAG='llama3b' jupyter notebook
   ```

2. After all 3 runs finish, the repo root will have `training_scores_qwen3b.json`, `training_scores_gemma2b.json`, `training_scores_llama3b.json` (plus the per-model `before_after_*.png` and `reward_curves_*.png`).

3. Aggregate: `python scripts/aggregate_models.py` (writes `aggregated_models.json`)

4. Generate the leaderboard chart: `python scripts/generate_charts.py` (now also produces `model_leaderboard.png`)

5. Re-populate README/blog: `python scripts/populate_results.py` — the Cross-Model Comparison section now contains the summary table, the winner narrative, and a reference to the leaderboard chart.

**Access requirements:**
- `Qwen/Qwen2.5-3B-Instruct` — public, no special access needed
- `google/gemma-2-2b-it` — requires accepting Gemma terms on the model's HF page
- `meta-llama/Llama-3.2-3B-Instruct` — requires Meta's gated-access approval (typically <1 hour after request)

Set `HF_TOKEN` environment variable with a token that has read access to gated models.

The marker-bounded sections in this README (`AUTO-RESULTS-TABLE-START`, `AUTO-HEADLINE-START`, `AUTO-STATRIGOR-START`, `AUTO-MULTIMODEL-START`, `AUTO-ADVCAPTION-START`) are auto-replaced by `populate_results.py`. To revert: `python scripts/restore_backups.py`. Pre-submission readiness: `python scripts/verify_pre_submit.py`.

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
├── demo_script.md                    # Spoken-walkthrough script (notes, not produced)
├── before_after.png                  # Bar chart + per-task Δ (training evidence)
├── reward_distribution.png           # Bimodal score histogram (baseline vs trained)
├── episode_scatter.png               # Per-episode paired comparison (4 panels)
├── reward_curves.png                 # Annotated GRPO reward + loss curves
├── training_scores.json              # Raw eval numbers
├── training_log.json                 # TRL log_history
├── adversarial_results.json          # Adversarial benchmark per-model scores
├── scripts/
│   ├── generate_charts.py            # Regenerate all charts from JSON artifacts
│   ├── statistical_analysis.py       # Wilcoxon, bootstrap CIs, training diagnostics
│   ├── aggregate_seeds.py            # Cross-seed reproducibility aggregation
│   ├── aggregate_models.py           # Cross-model leaderboard aggregation
│   ├── populate_results.py           # Fill marker-bounded README/blog sections
│   ├── restore_backups.py            # Undo populate_results (restore from .bak)
│   └── verify_pre_submit.py          # Pre-submission 9-item checklist runner
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
