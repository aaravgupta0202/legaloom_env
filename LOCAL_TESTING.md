# Local Testing Guide — LegaLoom-Env

Everything you need to test, run, and verify before the finals.

---

## One-time setup

```powershell
cd C:\Users\aarav\Desktop\Work\legaloom_env
python -m venv venv
venv\Scripts\activate
pip install openenv-core==0.2.3 fastapi==0.135.3 uvicorn==0.44.0 pydantic==2.12.5 openai==2.30.0 httpx==0.28.1 pyyaml==6.0.2
```

---

## Method 1: Run inference.py (the real test — what judges do)

This runs all 4 tasks, uses the LLM, and prints scored logs.

```powershell
venv\Scripts\activate

# With Groq (free, fast)
$env:API_BASE_URL = "https://api.groq.com/openai/v1"
$env:MODEL_NAME   = "llama-3.1-8b-instant"
$env:HF_TOKEN     = "gsk_YOUR_GROQ_KEY"
python inference.py
```

Expected output per task:
```
[START] task=task_easy env=legaloom_env model=llama-3.1-8b-instant seed=42
[STEP] step=1 action=read_invoice() reward=0.02 done=false error=null
[STEP] step=2 action=check_pan(pan=AADPA9012F) reward=0.10 done=false error=null
...
[END] success=true steps=4 score=0.85 rewards=0.02,0.10,0.25,0.85
```

Scores appear in `[SUMMARY]` on stderr:
```
[SUMMARY]
  task_easy:  score=0.85 success=True  steps=4
  task_medium: score=0.45 success=False steps=8
  task_hard:  score=0.59 success=True  steps=8
  task_expert: score=0.41 success=False steps=10
  Average score: 0.58
```

---

## Method 2: Run the server + curl

**Terminal 1 — server:**
```powershell
venv\Scripts\activate
$env:PYTHONPATH = "."
uvicorn server.app:app --host 0.0.0.0 --port 7860
# Wait for: "Application startup complete."
```

**Terminal 2 — test calls:**

> ⚠️ **Important:** Step body wraps action inside `{"action": {...}}` — this is the OpenEnv HTTP spec.

```powershell
# Health
curl http://localhost:7860/health

# Reset (must be first — starts a new episode)
curl -X POST http://localhost:7860/reset `
  -H "Content-Type: application/json" `
  -d '{"task_id": "task_easy"}'

# Step: read_invoice — NOTE the "action" wrapper
curl -X POST http://localhost:7860/step `
  -H "Content-Type: application/json" `
  -d '{"action": {"action_type": "read_invoice", "parameters": {}}}'

# Step: check_pan (replace PAN with one from the invoice)
curl -X POST http://localhost:7860/step `
  -H "Content-Type: application/json" `
  -d '{"action": {"action_type": "check_pan", "parameters": {"pan": "AADPA9012F"}}}'

# Step: submit_answer
curl -X POST http://localhost:7860/step `
  -H "Content-Type: application/json" `
  -d '{"action": {"action_type": "submit_answer", "parameters": {"tds_amount_inr": 7500.0, "section": "194J", "rate_percent": 10.0}}}'
```

---

## Method 3: Docker (required — what judges verify)

```powershell
# Build
docker build -t legaloom-env .

# Run
docker run -p 7860:7860 legaloom-env

# From another terminal — verify
curl http://localhost:7860/health
# Expected: {"status":"healthy"}

curl -X POST http://localhost:7860/reset `
  -H "Content-Type: application/json" `
  -d '{"task_id": "task_easy"}'
```

---

## Method 4: Run tests

```powershell
venv\Scripts\activate
$env:PYTHONPATH = "."
pytest tests/ -v
```

Expected: 4 tests pass.

---

## What to check before submitting

| Check | How to verify |
|-------|--------------|
| All 4 tasks reset | `curl /reset` with each task_id |
| `[START]` line has `seed=` | Run inference.py, check output |
| `[END]` line has `score=` and `rewards=` | Run inference.py, check output |
| Scores in (0.05, 0.95) | Check `[END]` line |
| Docker builds | `docker build -t legaloom-env .` |
| Docker runs on port 7860 | `docker run -p 7860:7860 legaloom-env` |
| HF Space responds to /health | `curl https://YOUR_SPACE.hf.space/health` |
| All placeholder URLs updated in README | Search for `YOUR_HF_USERNAME` |
