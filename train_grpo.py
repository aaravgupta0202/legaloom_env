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
Read a vendor invoice and compute the exact TDS deduction in INR.

OUTPUT FORMAT: Each turn output ONLY a valid JSON object. No markdown, no explanation.

ACTIONS:
1. {"action_type": "read_invoice", "parameters": {}}
2. {"action_type": "check_pan", "parameters": {"pan": "<10-char PAN>"}}
3. {"action_type": "query_ytd", "parameters": {"pan": "<PAN>"}}
4. {"action_type": "lookup_section", "parameters": {"description": "<service>"}}
5. {"action_type": "check_threshold", "parameters": {"section": "194I", "amount": 65000}}
6. {"action_type": "query_law", "parameters": {"section": "194J"}}
7. {"action_type": "submit_answer", "parameters": {"tds_amount_inr": 6500.0, "section": "194I", "rate_percent": 10.0}}
No TDS: {"action_type": "submit_answer", "parameters": {"tds_amount_inr": 0.0, "no_tds": "true", "section": "194I", "rate_percent": 0.0}}

RULE 1 — INOPERATIVE PAN: check_pan returns INOPERATIVE → rate = 20% flat (Section 206AA)
RULE 2 — GST: GST on separate line → TDS on pre-GST. GST bundled → TDS on full amount.
RULE 3 — Mixed invoices: Goods/hardware → NO TDS on that portion.
RULE 4 — Thresholds: 194J: 50K/yr | 194C: 30K single / 1L annual | 194I: 6L/yr | 194H: 20K/yr
RULE 5 — Company vs Individual: Pvt Ltd → 194J Technical 2%. Individual/LLP → 194J Professional 10%.

STRATEGY:
Step 1: read_invoice
Step 2: check_pan ← ALWAYS. If INOPERATIVE → submit with rate=20%
Step 3: lookup_section
Step 4: check_threshold (if near section limit)
Step 5: submit_answer

Output ONLY the JSON. Nothing else."""


def _extract_action(text: str) -> Optional[Dict]:
    match = re.search(r'\{[^{}]+\}', text, re.DOTALL)
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
        {"role": "system", "content": SYSTEM_PROMPT},
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
            trajectory.append({
                "step": step,
                "generated": generated[:200],
                "action": None,
                "reward": 0.05,
                "error": "no_valid_json",
            })
            final_reward = 0.05
            steps_used = step
            break

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
            obs_text = (
                f"Step {step} result: {result.action_result[:300]}\n"
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
    GRPO-compatible reward function using FULL episode rollouts.

    The model receives a prompt describing the task. It emits a SEQUENCE
    of JSON actions separated by newlines (one per line). This function:
    1. Parses that action sequence from the completion
    2. Runs the full environment episode step-by-step
    3. Returns ONLY the final terminal reward from submit_answer

    This is honest multi-step RL: the model must produce the entire
    reasoning chain, not just a single action. The trainer does NOT
    inject read_invoice or check_pan — the model must emit them itself.

    Anti-hacking: no format bonus, no per-step free reward. Only the
    final grader score from submit_answer counts.
    """
    task_id = kwargs.get("task_id", "task_easy")
    base_seed = kwargs.get("seed", 42)
    rewards = []

    for i, completion in enumerate(completions):
        text = completion[0]["content"] if isinstance(completion, list) else str(completion)
        seed = base_seed + i  # Each generation gets a slightly different episode

        try:
            env = LegaloomEnvironment()
            env.reset(task_id=task_id, seed=seed)

            # Parse ALL JSON actions from the completion (one per line or block)
            import re as _re
            action_blocks = _re.findall(r'\{[^{}]+\}', text, _re.DOTALL)

            final_reward = 0.05
            submitted = False

            for block in action_blocks:
                try:
                    action_dict = json.loads(block)
                    action_type = action_dict.get("action_type", "")
                    params = action_dict.get("parameters", {})
                    if not action_type:
                        continue
                    result = env.step(TDSAction(action_type=action_type, parameters=params))
                    if result.done:
                        final_reward = clamp_score(float(result.reward))
                        submitted = True
                        break
                except (json.JSONDecodeError, Exception):
                    continue

            # If model never submitted, give minimum reward
            if not submitted:
                final_reward = 0.01  # SCORE_MIN

        except Exception:
            final_reward = 0.01  # SCORE_MIN

        rewards.append(final_reward)

    return rewards


def build_training_dataset(task_ids=None, examples_per_task=20, base_seed=42):
    """Build training prompts from environment resets."""
    from datasets import Dataset

    if task_ids is None:
        task_ids = ["task_easy", "task_medium", "task_hard", "task_expert"]

    examples = []
    for task_id in task_ids:
        for i in range(examples_per_task):
            seed = base_seed + i + hash(task_id) % 1000
            try:
                task = sample_task(task_id, seed=seed, use_procedural=True)  # Procedural prevents memorization
                obs_text = task.get("invoice_text", "")[:400] if task.get("invoice_text") else ""
                prompt_text = (
                    f"Task: {task_id}\n"
                    f"Difficulty: {task['difficulty']}\n"
                    f"Invoice: {task['invoice_id']}\n"
                    f"Max steps: {task['max_steps']}\n"
                    f"Start with read_invoice.\n\n"
                    f"Output your action as JSON:"
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
        max_prompt_length=512,
        max_completion_length=256,
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
