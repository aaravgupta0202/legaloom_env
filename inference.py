"""
inference.py — LegaLoom-Env Baseline Inference Script

Mandatory variables:
    API_BASE_URL  : LLM endpoint
    MODEL_NAME    : model identifier
    HF_TOKEN      : Hugging Face / API key

Stdout format (strictly enforced):
    [START] task=<task_id> env=legaloom_env model=<model>
    [STEP]  step=<n> action=<str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>
"""

import asyncio
import json
import os
import sys
import textwrap
from typing import List, Optional

from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")

BENCHMARK    = "legaloom_env"
MAX_STEPS    = 8
TEMPERATURE  = 0.2
MAX_TOKENS   = 512


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float,
             done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    action_clean = action.replace("\n", " ").replace("\r", "")[:200]
    print(
        f"[STEP] step={step} action={action_clean} "
        f"reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float,
            rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


SYSTEM_PROMPT = textwrap.dedent("""
You are an expert Indian tax compliance agent specialising in TDS (Tax Deducted at Source).

Your job is to analyse invoices and compute the correct TDS deduction in INR.

You interact with a TDS compliance environment using a strict JSON action format.
Each turn you must output ONLY a valid JSON object — no explanation, no markdown, no extra text.

Available actions:

1. Read the invoice:
   {"action_type": "read_invoice", "parameters": {}}

2. Check vendor PAN status:
   {"action_type": "check_pan", "parameters": {"pan": "<PAN_NUMBER>"}}

3. Check if TDS threshold is crossed:
   {"action_type": "check_threshold", "parameters": {"section": "<194J>", "amount": <INR_AMOUNT>}}

4. Look up the applicable TDS section:
   {"action_type": "lookup_section", "parameters": {"description": "<service description>"}}

5. Submit your final answer (ends the episode):
   {"action_type": "submit_answer", "parameters": {"tds_amount_inr": <AMOUNT>, "section": "<194J>", "rate_percent": <RATE>}}

Key TDS rules:
- If PAN is INOPERATIVE: always deduct 20% regardless of section
- 194J Professional Services (legal, audit, consulting): 10%
- 194J Technical Services (IT support, software, cloud): 2% (but 20% if PAN inoperative)
- 194C Contractors (catering, security, manpower): 2% for companies
- Goods/hardware: NO TDS
- Always check PAN before computing the deduction

Strategy:
  Step 1 → read_invoice
  Step 2 → check_pan (use PAN from invoice)
  Step 3 → lookup_section (use service description from invoice)
  Step 4 → submit_answer (compute amount = taxable_amount x rate / 100)

Output ONLY the JSON object. Nothing else.
""").strip()


def build_user_prompt(step: int, obs: dict, history: List[str]) -> str:
    history_block = "\n".join(history[-6:]) if history else "None"
    invoice_block = obs.get("invoice_text", "")
    invoice_section = f"\nINVOICE:\n{invoice_block}\n" if invoice_block else ""

    return textwrap.dedent(f"""
Step {step} of {obs.get('max_steps', 8)}.
{invoice_section}
Last result: {obs.get('action_result', '')}
Available actions: {obs.get('available_actions', [])}
Hint: {obs.get('hint', '')}

Previous steps:
{history_block}

Output your next action as a JSON object only.
""").strip()


def get_agent_action(client: OpenAI, step: int, obs: dict,
                     history: List[str]) -> dict:
    user_prompt = build_user_prompt(step, obs, history)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        raw = (completion.choices[0].message.content or "").strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        return json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"[DEBUG] JSON parse error at step {step}: {e}", flush=True)
        return {"action_type": "read_invoice", "parameters": {}}
    except Exception as e:
        print(f"[DEBUG] LLM call failed at step {step}: {e}", flush=True)
        return {"action_type": "read_invoice", "parameters": {}}


def run_episode(client: OpenAI, env, task_id: str) -> dict:
    from models import TDSAction

    rewards: List[float] = []
    history: List[str]   = []
    steps_taken  = 0
    success      = False
    score        = 0.0

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = env.reset(task_id=task_id)
        obs    = result.observation.__dict__ if hasattr(result.observation, '__dict__') else {}
        done   = result.done

        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            action_dict = get_agent_action(client, step, obs, history)
            action_type = action_dict.get("action_type", "read_invoice")
            parameters  = action_dict.get("parameters", {})

            action_str = (
                f"{action_type}("
                + ",".join(f"{k}={v}" for k, v in parameters.items())
                + ")"
            )

            error = None
            try:
                result = env.step(TDSAction(
                    action_type=action_type,
                    parameters=parameters,
                ))
                obs    = result.observation.__dict__ if hasattr(result.observation, '__dict__') else {}
                reward = float(result.reward or 0.0)
                done   = result.done
            except Exception as e:
                reward = 0.0
                done   = False
                error  = str(e)[:120]

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_str,
                     reward=reward, done=done, error=error)

            history.append(
                f"Step {step}: {action_str} -> reward {reward:+.2f} | "
                f"{obs.get('action_result', '')[:100]}"
            )

            if done:
                break

        score   = min(max(sum(rewards), 0.0), 1.0)
        success = score >= 0.5

    except Exception as e:
        print(f"[DEBUG] Episode error for {task_id}: {e}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken,
                score=score, rewards=rewards)

    return {"task_id": task_id, "score": score,
            "success": success, "steps": steps_taken}


def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    from server.tasks import all_task_ids
    from server.legaloom_env_environment import LegaloomEnvironment

    task_ids = all_task_ids()
    results  = []

    for task_id in task_ids:
        env = LegaloomEnvironment()

        class LocalEnvWrapper:
            def __init__(self, env):
                self._env = env
            def reset(self, task_id="task_easy", **kwargs):
                return self._env.reset(task_id=task_id)
            def step(self, action):
                return self._env.step(action)

        result = run_episode(client, LocalEnvWrapper(env), task_id)
        results.append(result)

    print("\n[SUMMARY]", file=sys.stderr)
    for r in results:
        print(
            f"  {r['task_id']}: score={r['score']:.3f} "
            f"success={r['success']} steps={r['steps']}",
            file=sys.stderr,
        )
    avg = sum(r["score"] for r in results) / len(results)
    print(f"  Average score: {avg:.3f}", file=sys.stderr)


if __name__ == "__main__":
    main()