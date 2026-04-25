"""
LegaLoom GRPO Training Pipeline — Full Episode Rollout

Correct RL training pipeline:
1. rollout_episode: runs full reset→step loop, returns FINAL reward
2. episode_reward_fn: integrates with GRPO trainer, runs batch rollouts
3. No per-step HTTP spam — uses in-process environment
4. num_generations ≥ 4 per prompt

Usage:
    python train_grpo.py --steps 20 --task task_easy

Or in Colab:
    !python train_grpo.py --steps 30
"""

import argparse
import json
import os
import re
import sys
import random
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server.legaloom_env_environment import LegaloomEnvironment
from server.tasks import all_task_ids, sample_task
from server.scoring import clamp_score
from models import TDSAction


SYSTEM_PROMPT = """You are an expert Indian TDS (Tax Deducted at Source) compliance agent, FY 2025-26.

You will be given a vendor invoice. Output the COMPLETE sequence of JSON actions
needed to solve the task, one JSON object per line. End with submit_answer.

ACTIONS (in order):
1. {"action_type": "read_invoice", "parameters": {}}
2. {"action_type": "check_pan", "parameters": {"pan": "<10-char PAN from invoice>"}}
3. {"action_type": "query_ytd", "parameters": {"pan": "<same PAN>"}}
4. {"action_type": "lookup_section", "parameters": {"description": "<service from invoice>"}}
5. {"action_type": "submit_answer", "parameters": {"tds_amount_inr": <number>, "section": "<194J|194C|194I|194H|194T|194Q>", "rate_percent": <number>}}

RULES (FY 2025-26):
- INOPERATIVE PAN → rate = 20% flat (Section 206AA), overrides everything
- 194J Professional 10% (legal/CA/audit, individual/LLP) | 194J Technical 2% (Pvt Ltd vendor)
- 194C Contractor 2% (security/catering/manpower) | 194I Rent 10% (building) or 2% (machinery)
- 194H Commission 2% | 194T Partner 10% (NEW) | 194Q Goods 0.1%
- Below threshold: submit {"tds_amount_inr": 0.0, "no_tds": "true", "section": "<sec>", "rate_percent": 0.0}
  Thresholds: 194J 50K | 194C 30K single/1L year | 194I 6L | 194H 20K
- GST on separate line → TDS on pre-GST. GST bundled → TDS on full amount.

OUTPUT: 4-5 JSON objects, one per line, ending with submit_answer. No markdown, no commentary."""


# Multi-turn rollout uses a different prompt style — one action per turn.
# This is used by rollout_episode (baseline measurement, post-training eval)
# where the model sees env feedback between actions.
ROLLOUT_SYSTEM_PROMPT = """You are an expert Indian TDS (Tax Deducted at Source) compliance agent, FY 2025-26.
Read a vendor invoice and compute the exact TDS deduction in INR.

OUTPUT: Each turn output ONLY a single valid JSON object. No markdown, no commentary.

ACTIONS:
1. {"action_type": "read_invoice", "parameters": {}}
2. {"action_type": "check_pan", "parameters": {"pan": "<10-char PAN>"}}
3. {"action_type": "query_ytd", "parameters": {"pan": "<PAN>"}}
4. {"action_type": "lookup_section", "parameters": {"description": "<service>"}}
5. {"action_type": "check_threshold", "parameters": {"section": "194I", "amount": 65000}}
6. {"action_type": "query_law", "parameters": {"section": "194J"}}
7. {"action_type": "submit_answer", "parameters": {"tds_amount_inr": 6500.0, "section": "194I", "rate_percent": 10.0}}

RULES (FY 2025-26):
- INOPERATIVE PAN → rate = 20% flat (Section 206AA)
- 194J Professional 10% (legal/CA, individual/LLP) | 194J Technical 2% (Pvt Ltd)
- 194C Contractor 2% | 194I Rent 10% (building) or 2% (machinery)
- 194H Commission 2% | 194T Partner 10% (NEW) | 194Q Goods 0.1%
- Below threshold: tds_amount_inr=0.0, no_tds="true"
  Thresholds: 194J 50K | 194C 30K single/1L year | 194I 6L | 194H 20K

STRATEGY: read_invoice → check_pan → lookup_section (or query_ytd) → submit_answer

Output ONE action JSON per turn."""


def _extract_action(text: str) -> Optional[Dict]:
    match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)?\}', text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return None


def rollout_episode(
    model,
    tokenizer,
    env: LegaloomEnvironment,
    task_id: str,
    max_steps: int = 10,
    seed: Optional[int] = None,
    temperature: float = 0.3,
    max_new_tokens: int = 256,
) -> Dict[str, Any]:
    """
    Run a FULL episode: reset → multi-step loop → return final reward.

    This is the core rollout function. Each call:
    1. Resets the environment for a fresh episode
    2. Generates actions from the model step-by-step
    3. Steps the environment with each action
    4. Tracks the full trajectory
    5. Returns ONLY the final reward (not per-step)

    Args:
        model: The language model (HF/Unsloth)
        tokenizer: Corresponding tokenizer
        env: LegaLoom environment instance
        task_id: Which task to run
        max_steps: Maximum steps per episode
        seed: Environment seed for reproducibility
        temperature: Sampling temperature
        max_new_tokens: Max tokens per generation

    Returns:
        Dict with: final_reward, trajectory, steps_used, success
    """
    if seed is None:
        seed = random.randint(1, 999999)

    obs = env.reset(task_id=task_id, seed=seed)
    trajectory = []
    conversation = [
        {"role": "system", "content": ROLLOUT_SYSTEM_PROMPT},
    ]

    initial_context = (
        f"Task: {task_id}\n"
        f"Result: {obs.action_result}\n"
        f"Available: {obs.available_actions}\n\n"
        f"Output your first action as JSON:"
    )
    conversation.append({"role": "user", "content": initial_context})

    final_reward = 0.05
    steps_used = 0

    for step in range(1, max_steps + 1):
        try:
            inputs = tokenizer.apply_chat_template(
                conversation,
                return_tensors="pt",
                add_generation_prompt=True,
            ).to(model.device)

            output = model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.eos_token_id,
            )

            generated = tokenizer.decode(
                output[0][inputs.shape[1]:],
                skip_special_tokens=True,
            ).strip()
        except Exception:
            generated = ""

        action_dict = _extract_action(generated)

        if action_dict is None:
            # Try a more permissive extraction — sometimes model adds prose
            # around the JSON or wraps it in markdown
            import re as _re
            for pat in [r"```json\s*(\{.*?\})\s*```", r"```\s*(\{.*?\})\s*```", r"(\{[^{}]*(?:\{[^{}]*\}[^{}]*)?\})"]:
                m = _re.search(pat, generated, _re.DOTALL)
                if m:
                    try:
                        action_dict = json.loads(m.group(1) if m.lastindex else m.group())
                        break
                    except Exception:
                        continue
            if action_dict is None:
                trajectory.append({
                    "step": step,
                    "generated": generated[:200],
                    "action": None,
                    "reward": 0.05,
                    "error": "no_valid_json",
                })
                # Don't break — let the episode keep trying.
                # If we never get a valid action, we'll exhaust max_steps.
                conversation.append({"role": "assistant", "content": generated})
                conversation.append({"role": "user", "content":
                    "Your output was not valid JSON. Output exactly one JSON object: "
                    "{\"action_type\": \"...\", \"parameters\": {...}}"
                })
                steps_used = step
                continue

        action_type = action_dict.get("action_type", "")
        params = action_dict.get("parameters", {})

        try:
            action = TDSAction(action_type=action_type, parameters=params)
            result = env.step(action)
            step_reward = float(result.reward)
            done = result.done

            trajectory.append({
                "step": step,
                "action": action_dict,
                "reward": step_reward,
                "done": done,
                "action_result": result.action_result[:150],
            })

            if done:
                final_reward = clamp_score(step_reward)
                steps_used = step
                break

            conversation.append({"role": "assistant", "content": generated})
            # Include the invoice text once after read_invoice succeeds
            invoice_block = ""
            if action_type == "read_invoice" and result.invoice_text:
                invoice_block = f"\nINVOICE:\n{result.invoice_text[:1500]}\n"
            obs_text = (
                f"Step {step} result: {result.action_result[:300]}{invoice_block}\n"
                f"Available: {result.available_actions}\n"
                f"Steps used: {result.steps_used}/{result.max_steps}\n\n"
                f"Output your next action as JSON:"
            )
            conversation.append({"role": "user", "content": obs_text})

        except Exception as e:
            trajectory.append({
                "step": step,
                "action": action_dict,
                "reward": 0.05,
                "error": str(e)[:100],
            })
            final_reward = 0.05
            steps_used = step
            break
    else:
        steps_used = max_steps

    success = final_reward >= 0.5

    return {
        "final_reward": final_reward,
        "trajectory": trajectory,
        "steps_used": steps_used,
        "success": success,
        "task_id": task_id,
        "seed": seed,
    }


def rollout_batch(
    model,
    tokenizer,
    task_id: str,
    num_generations: int = 4,
    max_steps: int = 10,
    base_seed: int = 42,
    temperature: float = 0.3,
) -> List[Dict]:
    """Run multiple rollouts for a single task (for GRPO num_generations)."""
    results = []
    for i in range(num_generations):
        env = LegaloomEnvironment()
        result = rollout_episode(
            model, tokenizer, env, task_id,
            max_steps=max_steps,
            seed=base_seed + i,
            temperature=temperature,
        )
        results.append(result)
    return results


def episode_reward_fn(prompts, completions, **kwargs) -> List[float]:
    """
    GRPO-compatible reward function using full episode rollouts in a single completion.

    The model emits a SEQUENCE of JSON actions in one completion. This function:
    1. Parses every JSON action block from the completion
    2. Replays them step-by-step in a fresh environment
    3. Computes a graded reward:
       - Full terminal reward if submit_answer was reached (the goal)
       - Partial credit based on valid action progress otherwise
    
    The partial credit creates reward variance even when the model fails to
    complete the episode, giving GRPO a learning signal. The model is still
    incentivized to reach submit_answer because that pays much more.
    
    Anti-hacking guards:
    - Workflow violations (e.g. submit before check_pan) get clamped low
    - Repeated identical actions don't accumulate reward
    - Invalid JSON or unknown actions count as 0 progress
    - Maximum partial credit is capped well below true success
    """
    task_id = kwargs.get("task_id", "task_easy")
    base_seed = kwargs.get("seed", 42)
    rewards = []

    for i, completion in enumerate(completions):
        text = completion[0]["content"] if isinstance(completion, list) else str(completion)
        seed = base_seed + i

        try:
            env = LegaloomEnvironment()
            env.reset(task_id=task_id, seed=seed)

            import re as _re
            action_blocks = _re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)?\}', text, _re.DOTALL)

            final_reward = 0.01
            submitted = False
            valid_actions_taken = 0
            distinct_action_types = set()
            cumulative_step_reward = 0.0

            for block in action_blocks:
                try:
                    action_dict = json.loads(block)
                except json.JSONDecodeError:
                    continue

                action_type = action_dict.get("action_type", "")
                params = action_dict.get("parameters", {})
                if not action_type:
                    continue

                try:
                    result = env.step(TDSAction(action_type=action_type, parameters=params))
                except Exception:
                    continue

                step_r = float(result.reward) if result.reward is not None else 0.0

                # Only count constructive progress (positive step rewards from env)
                if step_r > 0:
                    valid_actions_taken += 1
                    distinct_action_types.add(action_type)
                    cumulative_step_reward += step_r

                if result.done:
                    # Episode reached submit_answer — terminal grader score is authoritative
                    final_reward = clamp_score(float(result.reward))
                    submitted = True
                    break

            if not submitted:
                # Partial credit: model made valid progress but never submitted.
                # This creates GRPO advantage signal between completions that
                # got further vs ones that didn't. Capped at 0.40 — well below
                # the 0.5 success threshold and below typical correct submission scores.
                progress_score = (
                    0.05                                                   # base for any valid action
                    + 0.05 * min(valid_actions_taken, 4)                   # cap progress reward
                    + 0.04 * min(len(distinct_action_types), 4)            # diversity bonus
                    + min(cumulative_step_reward, 0.20)                    # raw env step rewards
                )
                final_reward = min(progress_score, 0.40)
                final_reward = max(final_reward, 0.01)  # never below SCORE_MIN

        except Exception:
            final_reward = 0.01

        rewards.append(final_reward)

    return rewards


def build_training_dataset(task_ids=None, examples_per_task=20, base_seed=42):
    """Build training prompts from environment resets.
    
    The prompt includes the FULL invoice text so the model can reason
    about it offline. The model emits the complete action sequence
    (read_invoice → check_pan → lookup_section → submit_answer) in one
    completion, separated by newlines.
    """
    from datasets import Dataset

    if task_ids is None:
        task_ids = ["task_easy", "task_medium", "task_hard", "task_expert"]

    examples = []
    for task_id in task_ids:
        for i in range(examples_per_task):
            seed = base_seed + i + hash(task_id) % 1000
            try:
                task = sample_task(task_id, seed=seed, use_procedural=True)
                # Include the FULL invoice text so the model can reason about it
                invoice_text = task.get("invoice_text", "")[:1500]
                prompt_text = (
                    f"INVOICE:\n{invoice_text}\n\n"
                    f"Solve this TDS compliance task. Output the COMPLETE sequence of "
                    f"JSON actions (one per line), ending with submit_answer.\n\n"
                    f"Output the action sequence now:"
                )
                examples.append({
                    "task_id": task_id,
                    "prompt": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt_text},
                    ],
                })
            except Exception:
                continue

    return Dataset.from_list(examples) if examples else None


def run_curriculum_training(
    model,
    tokenizer,
    task_schedule: List[str],
    steps_per_phase: int = 20,
    examples_per_task: int = 60,
    output_dir: str = "./legaloom_grpo_output",
    learning_rate: float = 5e-6,
    num_generations: int = 4,
    max_prompt_length: int = 1536,
    max_completion_length: int = 768,
):
    """Run multi-phase GRPO training across a task schedule.

    Each phase trains for `steps_per_phase` steps on a single task_id from the
    schedule, then moves to the next. The model carries learning across phases.
    Log entries are tagged with `phase` and `task_id` so curves can be plotted
    with phase boundaries.

    Example:
        schedule = ['task_easy', 'task_medium', 'task_hard', 'task_expert']
        log_history = run_curriculum_training(
            model, tokenizer, schedule, steps_per_phase=20
        )
        # Total: 80 steps across all 4 difficulty levels.

    For an interleaved curriculum, repeat the schedule:
        schedule = ['task_easy', 'task_medium', 'task_hard', 'task_expert'] * 2

    This function does NOT initialize the model — pass an already-loaded
    Unsloth/HF model + tokenizer. That keeps notebook control over quantization,
    LoRA setup, and gradient checkpointing.

    Returns the combined log_history list across all phases.
    """
    from trl import GRPOConfig, GRPOTrainer

    try:
        from unsloth import is_bfloat16_supported
        bf16 = is_bfloat16_supported()
    except ImportError:
        import torch
        bf16 = torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False
    fp16 = not bf16

    full_log = []
    for phase_idx, task_id in enumerate(task_schedule):
        print(f"\n=== Phase {phase_idx+1}/{len(task_schedule)}: {task_id} "
              f"({steps_per_phase} steps) ===")

        dataset = build_training_dataset(
            task_ids=[task_id],
            examples_per_task=examples_per_task,
        )
        if dataset is None:
            print(f"  WARNING: empty dataset for {task_id}, skipping phase")
            continue

        def _make_reward_fn(tid):
            def fn(prompts, completions, **kwargs):
                clean_kwargs = {k: v for k, v in kwargs.items() if k != "task_id"}
                return episode_reward_fn(prompts, completions, task_id=tid, **clean_kwargs)
            return fn

        cfg = GRPOConfig(
            learning_rate=learning_rate,
            num_train_epochs=1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            num_generations=num_generations,
            max_prompt_length=max_prompt_length,
            max_completion_length=max_completion_length,
            beta=0.01,
            logging_steps=1,
            max_steps=steps_per_phase,
            output_dir=os.path.join(output_dir, f"phase_{phase_idx}_{task_id}"),
            report_to="none",
            bf16=bf16,
            fp16=fp16,
            save_strategy="no",
        )

        trainer = GRPOTrainer(
            model=model,
            args=cfg,
            train_dataset=dataset,
            reward_funcs=[_make_reward_fn(task_id)],
            processing_class=tokenizer,
        )
        trainer.train()

        # Tag every log entry with phase + task_id so we can split curves later
        for entry in trainer.state.log_history:
            entry["phase"] = phase_idx
            entry["phase_task_id"] = task_id
        full_log.extend(trainer.state.log_history)

        print(f"  ✓ Phase {phase_idx+1} done. {len(trainer.state.log_history)} entries.")

    print(f"\n=== Curriculum training complete. {len(full_log)} total log entries. ===")
    return full_log


def run_training(
    model_name: str = "unsloth/Qwen2.5-3B-Instruct",
    num_steps: int = 20,
    task_id: str = "task_easy",
    output_dir: str = "./legaloom_grpo_output",
    use_unsloth: bool = True,
):
    """Run GRPO training with full episode rollouts."""
    import torch

    if use_unsloth:
        try:
            from unsloth import FastLanguageModel, is_bfloat16_supported

            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=2048,
                dtype=None,
                load_in_4bit=True,
            )
            model = FastLanguageModel.get_peft_model(
                model,
                r=16,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                 "gate_proj", "up_proj", "down_proj"],
                lora_alpha=16,
                lora_dropout=0,
                bias="none",
                use_gradient_checkpointing="unsloth",
                random_state=42,
            )
            bf16 = is_bfloat16_supported()
            fp16 = not bf16
        except ImportError:
            use_unsloth = False

    if not use_unsloth:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb_config, device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        bf16 = torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False
        fp16 = not bf16

    from trl import GRPOConfig, GRPOTrainer

    dataset = build_training_dataset(
        task_ids=[task_id],
        examples_per_task=max(num_steps * 2, 40),
    )

    if dataset is None:
        print("ERROR: Could not build training dataset")
        return None

    def make_reward_fn(tid):
        def fn(prompts, completions, **kwargs):
            # Strip task_id from kwargs to avoid "multiple values" collision
            # TRL passes dataset columns as kwargs; we override with our tid
            clean_kwargs = {k: v for k, v in kwargs.items() if k != "task_id"}
            return episode_reward_fn(prompts, completions, task_id=tid, **clean_kwargs)
        return fn

    grpo_config = GRPOConfig(
        learning_rate=5e-6,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_generations=4,
        max_prompt_length=1536,  # Invoice text is up to ~1500 chars
        max_completion_length=768,  # Need ~4 JSON action blocks × ~150 tokens each
        beta=0.01,
        logging_steps=1,
        max_steps=num_steps,
        output_dir=output_dir,
        report_to="none",
        bf16=bf16,
        fp16=fp16,
        save_strategy="no",
    )

    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=dataset,
        reward_funcs=[make_reward_fn(task_id)],
        processing_class=tokenizer,
    )

    print(f"Starting GRPO training: {num_steps} steps, task={task_id}")
    trainer.train()

    log_history = trainer.state.log_history
    return {
        "trainer": trainer,
        "model": model,
        "tokenizer": tokenizer,
        "log_history": log_history,
    }


def plot_reward_curves(log_history: List[Dict], output_path: str = "reward_curves.png"):
    """Plot reward curves from training log history."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = []
    rewards = []
    losses = []

    for entry in log_history:
        if "loss" in entry:
            steps.append(entry.get("step", len(steps)))
            losses.append(entry["loss"])
        if "reward" in entry:
            if not steps or entry.get("step", 0) != (steps[-1] if steps else -1):
                steps.append(entry.get("step", len(steps)))
            rewards.append(entry["reward"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    if rewards:
        reward_steps = list(range(1, len(rewards) + 1))
        ax.plot(reward_steps, rewards, "b-", linewidth=1.5, alpha=0.6, label="Step reward")
        window = 3
        if len(rewards) >= window:
            ma = []
            for i in range(len(rewards)):
                start = max(0, i - window + 1)
                ma.append(sum(rewards[start:i+1]) / (i - start + 1))
            ax.plot(reward_steps, ma, "r-", linewidth=2.5, label=f"{window}-step moving avg")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Reward")
    ax.set_title("GRPO Training — Reward Curve")
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)

    ax2 = axes[1]
    if losses:
        loss_steps = list(range(1, len(losses) + 1))
        ax2.plot(loss_steps, losses, "g-", linewidth=1.5, alpha=0.8)
    ax2.set_xlabel("Training Step")
    ax2.set_ylabel("Loss")
    ax2.set_title("GRPO Training — Loss Curve")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved reward curves → {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="LegaLoom GRPO Training")
    parser.add_argument("--steps", type=int, default=20, help="Number of GRPO training steps")
    parser.add_argument("--task", type=str, default="task_easy", help="Task ID to train on")
    parser.add_argument("--model", type=str, default="unsloth/Qwen2.5-3B-Instruct")
    parser.add_argument("--output_dir", type=str, default="./legaloom_grpo_output")
    parser.add_argument("--no_unsloth", action="store_true")
    args = parser.parse_args()

    result = run_training(
        model_name=args.model,
        num_steps=args.steps,
        task_id=args.task,
        output_dir=args.output_dir,
        use_unsloth=not args.no_unsloth,
    )

    if result is None:
        sys.exit(1)

    plot_reward_curves(result["log_history"], output_path="reward_curves.png")

    log_path = os.path.join(args.output_dir, "training_log.json")
    with open(log_path, "w") as f:
        json.dump(result["log_history"], f, indent=2, default=str)

    print(f"\nTraining complete. Log saved → {log_path}")


if __name__ == "__main__":
    main()