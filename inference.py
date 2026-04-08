"""
inference.py — LegaLoom-Env Baseline Inference Script

Usage (Windows):
    set API_BASE_URL=https://api.groq.com/openai/v1
    set MODEL_NAME=llama-3.3-70b-versatile
    set HF_TOKEN=gsk_xxxx
    python inference.py

Usage (Linux/Mac):
    export API_BASE_URL=https://api.groq.com/openai/v1
    export MODEL_NAME=llama-3.3-70b-versatile
    export HF_TOKEN=gsk_xxxx
    python inference.py

Stdout format (strictly enforced by hackathon spec):
    [START] task=<task_id> env=legaloom_env model=<model>
    [STEP]  step=<n> action=<str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.00> rewards=<r1,r2,...,rn>

Score contract: every value in the rewards= list and the score= field must be
strictly between 0 and 1 (not 0.0 and not 1.0).
"""

import json
import os
import sys
import textwrap
from typing import List, Optional

from openai import OpenAI

# ── Mandatory environment variables ──────────────────────────────────────────
API_BASE_URL     = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME       = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN         = os.getenv("HF_TOKEN")
API_KEY          = HF_TOKEN or os.getenv("API_KEY")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

BENCHMARK        = "legaloom_env"
MAX_STEPS        = 10
TEMPERATURE      = 0        # deterministic — reproducible baseline scores
MAX_TOKENS       = 300
SUCCESS_THRESHOLD = 0.5

# Score safety constants — values formatted with .2f must stay in (0.00, 1.00)
_SCORE_MIN = 0.05
_SCORE_MAX = 0.95


def _safe(v: float) -> float:
    """Clamp to [_SCORE_MIN, _SCORE_MAX] — always strictly in (0, 1)."""
    return round(min(max(float(v), _SCORE_MIN), _SCORE_MAX), 4)


# ── Mandatory stdout loggers ──────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float,
             done: bool, error: Optional[str]) -> None:
    print(
        f"[STEP] step={step} "
        f"action={action.replace(chr(10),' ').replace(chr(13),'')[:200]} "
        f"reward={_safe(reward):.2f} "
        f"done={str(done).lower()} "
        f"error={error if error else 'null'}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    # rewards must never be empty — insert score as sentinel if no steps completed
    safe_list   = [_safe(r) for r in rewards] if rewards else [_safe(score)]
    rewards_str = ",".join(f"{r:.2f}" for r in safe_list)
    safe_score  = _safe(score)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={safe_score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
You are an expert Indian TDS (Tax Deducted at Source) compliance agent, FY 2025-26.
Read a vendor invoice and compute the exact TDS deduction in INR.

OUTPUT FORMAT: Each turn output ONLY a valid JSON object. No markdown, no explanation.

ACTIONS:
1. Read invoice (always first step):
   {"action_type": "read_invoice", "parameters": {}}

2. Check vendor PAN status (always do this before computing):
   {"action_type": "check_pan", "parameters": {"pan": "<10-char PAN>"}}

3. Query cumulative YTD payments to vendor:
   {"action_type": "query_ytd", "parameters": {"pan": "<PAN>"}}

4. Look up TDS section for a service:
   {"action_type": "lookup_section", "parameters": {"description": "<service description>"}}

5. Check if TDS threshold is crossed:
   {"action_type": "check_threshold", "parameters": {"section": "194J", "amount": 85000}}

6. Look up law text for a section:
   {"action_type": "query_law", "parameters": {"section": "194J"}}

7. Submit your final answer (ends episode):
   {"action_type": "submit_answer", "parameters": {
     "tds_amount_inr": <AMOUNT>, "section": "<194J>", "rate_percent": <RATE>
   }}
   For NO TDS (below threshold):
   {"action_type": "submit_answer", "parameters": {
     "tds_amount_inr": 0.0, "no_tds": "true", "section": "<section>", "rate_percent": 0.0
   }}

TDS RULES FY 2025-26:
  194J Professional (10%): legal, CA/audit, medical, architect, CS — individual/LLP vendor
  194J Technical    (2%):  IT, cloud, software dev, BPO — company (Pvt Ltd) vendor = always technical
  194C Contractor   (2%):  security, catering, housekeeping, manpower, events, printing
  194I Rent        (10%):  office/warehouse/land/building rent
  194I Rent         (2%):  machinery/equipment/vehicle hire (no driver)
  194H Commission   (2%):  sales commission, brokerage, referral fees
  194T Partner     (10%):  NEW — partner salary/drawings from firm
  194Q Goods       (0.1%): goods purchase >50L/year (buyer turnover >10Cr)

CRITICAL RULES:
  1. ALWAYS check PAN first — inoperative PAN = 20% flat (Sec 206AA), overrides all rates
  2. Thresholds: 194J=50,000 | 194C=30,000 single/1,00,000 annual | 194I=6,00,000/yr | 194H=20,000
  3. GST shown separately → TDS on pre-GST amount
     GST bundled in total  → TDS on FULL invoice amount
  4. Goods line items (hardware, products, materials) → NO TDS
     TDS only on service/rent/commission portion
  5. Below threshold → submit with tds_amount_inr=0.0 and no_tds="true"

STRATEGY: read_invoice → check_pan → lookup_section → [query_ytd if near threshold] → submit_answer

Output ONLY the JSON. Nothing else.
""").strip()


# ── LLM interaction ───────────────────────────────────────────────────────────

def build_user_prompt(step: int, obs: dict, history: List[str]) -> str:
    history_block   = "\n".join(history[-8:]) if history else "None"
    invoice_block   = obs.get("invoice_text", "")
    invoice_section = f"\nINVOICE:\n{invoice_block}\n" if invoice_block else ""
    hint            = obs.get("hint", "")
    hint_line       = f"\nHint: {hint}" if hint else ""
    return textwrap.dedent(f"""
Step {step} of {obs.get('max_steps', 10)}.{invoice_section}
Last result: {obs.get('action_result', '')}
Available actions: {obs.get('available_actions', [])}
{hint_line}

Previous steps:
{history_block}

Output your next action as a JSON object only.
""").strip()


def get_agent_action(client: OpenAI, step: int, obs: dict,
                     history: List[str]) -> dict:
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": build_user_prompt(step, obs, history)},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        raw = (completion.choices[0].message.content or "").strip()
        # Strip markdown fences if model wraps JSON
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        return json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"[DEBUG] JSON parse error at step {step}: {e}", flush=True)
        return {"action_type": "submit_answer",
                "parameters": {"tds_amount_inr": 0.0, "section": "194J",
                                "rate_percent": 0.0, "no_tds": "true"}}
    except Exception as e:
        print(f"[DEBUG] LLM call failed at step {step}: {e}", flush=True)
        return {"action_type": "submit_answer",
                "parameters": {"tds_amount_inr": 0.0, "section": "194J",
                                "rate_percent": 0.0, "no_tds": "true"}}


# ── Episode runner ────────────────────────────────────────────────────────────

def run_episode(client: OpenAI, env, task_id: str) -> dict:
    from models import TDSAction

    rewards: List[float] = []
    history: List[str]   = []
    steps_taken = 0
    score       = _SCORE_MIN   # safe default — never 0.0
    success     = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = env.reset(task_id=task_id)

        # Support both local env (returns obs directly) and HTTP client (StepResult wrapper)
        if hasattr(result, "observation"):
            obs  = result.observation.__dict__ if hasattr(result.observation, "__dict__") else {}
            done = result.done
        else:
            obs  = result.__dict__ if hasattr(result, "__dict__") else {}
            done = getattr(result, "done", False)

        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            action_dict = get_agent_action(client, step, obs, history)
            action_type = action_dict.get("action_type", "read_invoice")
            parameters  = action_dict.get("parameters", {})
            action_str  = (
                f"{action_type}("
                + ",".join(f"{k}={v}" for k, v in parameters.items())
                + ")"
            )

            error = None
            try:
                result = env.step(TDSAction(action_type=action_type, parameters=parameters))
                if hasattr(result, "observation"):
                    obs    = result.observation.__dict__ if hasattr(result.observation, "__dict__") else {}
                    reward = float(result.reward) if result.reward is not None else _SCORE_MIN
                    done   = result.done
                else:
                    obs    = result.__dict__ if hasattr(result, "__dict__") else {}
                    reward = float(getattr(result, "reward", _SCORE_MIN) or _SCORE_MIN)
                    done   = getattr(result, "done", False)
            except Exception as e:
                reward = _SCORE_MIN
                done   = False
                error  = str(e)[:120]

            reward = _safe(reward)
            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_str, reward=reward, done=done, error=error)
            history.append(
                f"Step {step}: {action_str} → reward={reward:.2f} | "
                f"{obs.get('action_result', '')[:120]}"
            )

            if done:
                break

        # Score = last reward (the terminal grader score from submit_answer)
        score   = rewards[-1] if rewards else _SCORE_MIN
        score   = _safe(score)
        success = score >= SUCCESS_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Episode error for {task_id}: {e}", flush=True)
        # score stays at _SCORE_MIN — valid, never 0.0

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {"task_id": task_id, "score": score, "success": success, "steps": steps_taken}


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    client   = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    task_ids = ["task_easy", "task_medium", "task_hard", "task_expert"]
    results  = []

    if LOCAL_IMAGE_NAME:
        print(f"[INFO] Docker mode: {LOCAL_IMAGE_NAME}", file=sys.stderr)
        from client import LegaloomEnv
        env_url = os.getenv("ENV_BASE_URL", "http://localhost:7860")
        for task_id in task_ids:
            with LegaloomEnv(base_url=env_url).sync() as env:
                results.append(run_episode(client, env, task_id))
    else:
        from server.legaloom_env_environment import LegaloomEnvironment

        for task_id in task_ids:
            env = LegaloomEnvironment()

            class LocalEnvWrapper:
                def __init__(self, e): self._env = e
                def reset(self, task_id="task_easy", **kw):
                    return self._env.reset(task_id=task_id)
                def step(self, action):
                    return self._env.step(action)

            results.append(run_episode(client, LocalEnvWrapper(env), task_id))

    # Summary to stderr — keeps stdout clean for the evaluator parser
    print("\n[SUMMARY]", file=sys.stderr)
    for r in results:
        print(
            f"  {r['task_id']}: score={r['score']:.2f} "
            f"success={r['success']} steps={r['steps']}",
            file=sys.stderr,
        )
    avg = sum(r["score"] for r in results) / len(results)
    print(f"  Average score: {avg:.2f}", file=sys.stderr)


if __name__ == "__main__":
    main()
