---
title: LegaLoom-Env
emoji: ⚖️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# LegaLoom-Env — TDS Compliance Environment for AI Agents

An OpenEnv-compatible reinforcement learning environment where AI agents learn
to perform **Tax Deducted at Source (TDS) compliance** — one of the most
critical and error-prone financial back-office tasks in India.

## Why TDS Compliance?

Every Indian company processes hundreds of vendor invoices monthly. Each
invoice requires the accounts team to:

1. Identify the correct TDS section (194C, 194J, 194I, etc.)
2. Verify the vendor's PAN status (operative vs inoperative)
3. Apply the correct deduction rate (2%, 10%, or 20% fallback)
4. Compute the exact INR amount to deduct

A single systemic error — for example applying 2% instead of 10% across all
consulting invoices — results in tax shortfall, 1.5% monthly interest, and
penalties during audit. This environment models that exact real-world task.

## Environment Description

The agent receives realistic Indian vendor invoices and must determine the
correct TDS deduction through a structured action interface. The environment
provides partial rewards at each reasoning step, not just binary end-of-episode
scores.

### Action Space

The agent takes one action per step by sending a JSON object:

| Action Type | Parameters | Description |
|---|---|---|
| `read_invoice` | none | Retrieve the full invoice text |
| `check_pan` | `pan: str` | Verify vendor PAN operative status |
| `check_threshold` | `section: str, amount: float` | Check if TDS threshold is crossed |
| `lookup_section` | `description: str` | Identify applicable TDS section |
| `submit_answer` | `tds_amount_inr: float, section: str, rate_percent: float` | Submit final deduction |

### Observation Space

Each step returns:

| Field | Type | Description |
|---|---|---|
| `invoice_text` | str | Full invoice (populated after read_invoice) |
| `action_result` | str | Environment's response to last action |
| `available_actions` | list | Valid actions for next step |
| `steps_used` | int | Steps taken so far |
| `max_steps` | int | Maximum steps allowed |
| `hint` | str | Guidance (empty on hard difficulty) |

### Reward Function

Rewards are shaped across the full trajectory:

| Event | Reward |
|---|---|
| Checking PAN status | +0.10 to +0.20 |
| Identifying correct TDS section | +0.20 to +0.30 |
| Detecting inoperative PAN | +0.30 (Task 3 only) |
| Excluding goods from TDS | +0.20 (Task 2 only) |
| Submitting exact correct amount (±₹1) | +0.40 |
| Unknown action | −0.05 |

## Tasks

### Task 1 — Easy (`task_easy`)
**Single invoice, unambiguous professional service**

- Vendor: Sharma & Associates LLP (operative PAN)
- Service: Legal Consultation — INR 1,50,000
- Expected: Section 194J @ 10% = **₹15,000 TDS**
- Max steps: 6

### Task 2 — Medium (`task_medium`)
**Mixed invoice: goods + services, threshold check required**

- Vendor: TechServ Solutions Pvt Ltd (operative PAN)
- Line items: Hardware ₹95,000 (no TDS) + IT Support ₹85,000 (TDS applies)
- Agent must exclude goods and apply TDS only on services
- Expected: Section 194J @ 10% on ₹85,000 = **₹8,500 TDS**
- Max steps: 8

### Task 3 — Hard (`task_hard`)
**Inoperative PAN + ambiguous service description**

- Vendor: CloudMatrix Infrastructure Pvt Ltd (**inoperative PAN**)
- Service: "Cloud Infrastructure & Platform Services" — ambiguous between
  194J Technical (2%) and Professional (10%)
- Agent must detect inoperative PAN and apply 20% fallback regardless of section
- Expected: 20% on ₹2,40,000 = **₹48,000 TDS**
- Max steps: 8

## Baseline Scores

Evaluated using `Qwen/Qwen2.5-72B-Instruct` via HuggingFace Inference Router:

| Task | Score | Steps | Notes |
|---|---|---|---|
| task_easy | 1.000 | 4 | Perfect — correct section, rate, amount |
| task_medium | 0.500 | 4 | Partial — section correct, rate wrong (2% vs 10%) |
| task_hard | 1.000 | 3 | Perfect — detected inoperative PAN, applied 20% |
| **Average** | **0.833** | | |

## Setup & Usage

### Install
```bash
pip install openenv-core openai
```

### Run locally
```bash
git clone https://github.com/aaravgupta0202/legaloom_env.git
cd legaloom_env
pip install openenv-core fastapi uvicorn pydantic openai
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### Test endpoints
```bash
# Reset — start a new episode
curl -X POST http://localhost:8000/reset \
     -H "Content-Type: application/json" -d "{}"

# Step — take an action
curl -X POST http://localhost:8000/step \
     -H "Content-Type: application/json" \
     -d '{"action": {"action_type": "read_invoice", "parameters": {}}}'

# Health check
curl http://localhost:8000/health
```

### Run with Docker
```bash
docker build -t legaloom-env .
docker run -p 7860:7860 legaloom-env
```

### Run baseline inference
```bash
export HF_TOKEN=your_token_here
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
python inference.py
```

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/reset` | POST | Start new episode |
| `/step` | POST | Submit action, get observation + reward |
| `/state` | GET | Get current episode metadata |
| `/health` | GET | Health check |
| `/docs` | GET | Auto-generated API documentation |

## Project Structure