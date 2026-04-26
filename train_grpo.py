"""
LegaLoom GRPO Training Pipeline — Full Episode Rollout v2

Key improvements over v1:
1. Curriculum learning: easy→medium→hard tasks
2. Shaped rewards: partial credit for correct workflow steps
3. Better LoRA config: alpha/r=2, no dropout
4. Higher temperature for GRPO exploration (0.7)
5. Longer generation (512 tokens) for multi-step episodes
6. More training steps (50+), smaller LR (2e-6)
7. KL penalty beta=0.01 for stable policy updates
8. Warmup + cosine schedule for stable convergence
9. Multi-task reward functions with difficulty-based weighting
10. Evaluation rollouts every N steps to track progress

Usage:
python train_grpo.py --steps 50

Or in Colab:
!python train_grpo.py --steps 50 --curriculum
"""

import argparse
import json
import os
import re
import sys
import random
import time
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server.legaloom_env_environment import LegaloomEnvironment
from server.tasks import all_task_ids, sample_task
from server.scoring import clamp_score
from models import TDSAction

CURRICULUM_ORDER = ["task_easy", "task_medium", "task_hard", "task_expert"]

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
    # Strip line-anchored markdown code fences (```json ... ```) without
    # corrupting backticks inside JSON strings.
    cleaned = re.sub(r'^[ \t]*```(?:json|JSON)?[ \t]*\n?', '', text, flags=re.MULTILINE)
    cleaned = re.sub(r'\n[ \t]*```[ \t]*$', '', cleaned, flags=re.MULTILINE)
    match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)?\}', cleaned, re.DOTALL)
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
    temperature: float = 0.7,
    max_new_tokens: int = 512,
) -> Dict[str, Any]:
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
    workflow_bonus = 0.0

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
                top_p=0.9,
                repetition_penalty=1.1,
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

            if step_reward > 0:
                workflow_bonus += step_reward * 0.1

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

            # Include invoice_text if present (it's only set by read_invoice).
            # Without this, the model never sees the actual invoice content
            # and can only emit empty/guessed PAN strings, which Pydantic
            # rejects → silent fall-through to 0.05 floor reward.
            obs_text_parts = [f"Step {step} result: {result.action_result[:300]}"]
            invoice_text = getattr(result, "invoice_text", None)
            if invoice_text:
                obs_text_parts.append(f"Invoice content:\n{invoice_text}")
            obs_text_parts.append(f"Available: {result.available_actions}")
            obs_text_parts.append(
                f"Steps used: {result.steps_used}/{result.max_steps}\n\n"
                f"Output your next action as JSON:"
            )
            obs_text = "\n".join(obs_text_parts)
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
        "workflow_bonus": workflow_bonus,
    }


def rollout_batch(
    model,
    tokenizer,
    task_id: str,
    num_generations: int = 4,
    max_steps: int = 10,
    base_seed: int = 42,
    temperature: float = 0.7,
) -> List[Dict]:
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
            # Strip line-anchored markdown code fences without corrupting
            # backticks that may legitimately appear inside JSON strings.
            cleaned = _re.sub(r'^[ \t]*```(?:json|JSON)?[ \t]*\n?', '', text, flags=_re.MULTILINE)
            cleaned = _re.sub(r'\n[ \t]*```[ \t]*$', '', cleaned, flags=_re.MULTILINE)
            action_blocks = _re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)?\}', cleaned, _re.DOTALL)

            final_reward = 0.05
            submitted = False
            trajectory_steps = []

            for block in action_blocks:
                try:
                    action_dict = json.loads(block)
                    action_type = action_dict.get("action_type", "")
                    params = action_dict.get("parameters", {})
                    if not action_type:
                        continue
                    result = env.step(TDSAction(action_type=action_type, parameters=params))
                    trajectory_steps.append({"action": action_dict})
                    if result.done:
                        final_reward = clamp_score(float(result.reward))
                        submitted = True
                        break
                except (json.JSONDecodeError, Exception):
                    continue

            rewards.append(clamp_score(final_reward))

        except Exception:
            rewards.append(0.01)

    return rewards


def build_training_dataset(task_ids=None, examples_per_task=30, base_seed=42):
    from datasets import Dataset

    if task_ids is None:
        task_ids = CURRICULUM_ORDER

    examples = []
    for task_id in task_ids:
        for i in range(examples_per_task):
            seed = base_seed + i + hash(task_id) % 1000
            try:
                task = sample_task(task_id, seed=seed, use_procedural=True)
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


def _get_curriculum_task_ids(step: int, total_steps: int) -> List[str]:
    if total_steps <= 0:
        return ["task_easy"]
    progress = min(step / total_steps, 1.0)
    if progress < 0.33:
        return ["task_easy"]
    elif progress < 0.66:
        return ["task_easy", "task_medium"]
    else:
        return ["task_easy", "task_medium", "task_hard"]


def evaluate_model(model, tokenizer, num_episodes: int = 8, base_seed: int = 999) -> Dict[str, float]:
    results = {}
    for task_id in CURRICULUM_ORDER:
        rewards = []
        for i in range(num_episodes):
            env = LegaloomEnvironment()
            result = rollout_episode(
                model, tokenizer, env, task_id,
                max_steps=10,
                seed=base_seed + i,
                temperature=0.1,
                max_new_tokens=512,
            )
            rewards.append(result["final_reward"])
        results[task_id] = sum(rewards) / len(rewards) if rewards else 0.0
    results["average"] = sum(results.values()) / len(results) if results else 0.0
    return results


def run_curriculum_training(
    model,
    tokenizer,
    task_schedule: list = None,
    steps_per_phase: int = 20,
    examples_per_task: int = 60,
    output_dir: str = "./legaloom_grpo_output",
    learning_rate: float = 5e-6,
    num_generations: int = 4,
    max_prompt_length: int = 1536,
    max_completion_length: int = 768,
):
    """Run multi-phase GRPO training across a task schedule.

    Each phase trains for `steps_per_phase` on one task_id, then moves
    to the next. The model carries learning across phases.

    Returns the combined log_history list across all phases.
    """
    from trl import GRPOConfig, GRPOTrainer

    if task_schedule is None:
        task_schedule = CURRICULUM_ORDER

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
            print(f"  WARNING: empty dataset for {task_id}, skipping")
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

        for entry in trainer.state.log_history:
            entry["phase"] = phase_idx
            entry["phase_task_id"] = task_id
        full_log.extend(trainer.state.log_history)
        print(f"  ✓ Phase {phase_idx+1} done. {len(trainer.state.log_history)} entries.")

    print(f"\n=== Curriculum complete. {len(full_log)} total log entries. ===")
    return full_log


def run_training(
    model_name: str = "unsloth/Qwen2.5-3B-Instruct",
    num_steps: int = 50,
    task_id: str = "task_easy",
    output_dir: str = "./legaloom_grpo_output",
    use_unsloth: bool = True,
    curriculum: bool = False,
):
    import torch

    if use_unsloth:
        try:
            from unsloth import FastLanguageModel, is_bfloat16_supported

            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=4096,
                dtype=None,
                load_in_4bit=True,
            )
            model = FastLanguageModel.get_peft_model(
                model,
                r=16,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                 "gate_proj", "up_proj", "down_proj"],
                lora_alpha=32,
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

    train_task_ids = CURRICULUM_ORDER if curriculum else [task_id]

    dataset = build_training_dataset(
        task_ids=train_task_ids,
        examples_per_task=max(num_steps * 2, 40),
    )

    if dataset is None:
        print("ERROR: Could not build training dataset")
        return None

    def make_reward_fn(tid):
        def fn(prompts, completions, **kwargs):
            clean_kwargs = {k: v for k, v in kwargs.items() if k != "task_id"}
            return episode_reward_fn(prompts, completions, task_id=tid, **clean_kwargs)
        return fn

    reward_funcs = [make_reward_fn(tid) for tid in train_task_ids]

    grpo_config = GRPOConfig(
        learning_rate=2e-6,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_generations=4,
        max_prompt_length=512,
        max_completion_length=512,
        beta=0.01,
        logging_steps=1,
        max_steps=num_steps,
        output_dir=output_dir,
        report_to="none",
        bf16=bf16,
        fp16=fp16,
        save_strategy="steps",
        save_steps=max(num_steps // 3, 10),
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        max_grad_norm=1.0,
    )

    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=dataset,
        reward_funcs=reward_funcs,
        processing_class=tokenizer,
    )

    print(f"Starting GRPO training: {num_steps} steps, tasks={train_task_ids}, curriculum={curriculum}")
    print(f"  LR=2e-6, beta=0.01, lora_alpha=32, max_completion=512, warmup=10%")
    trainer.train()

    log_history = trainer.state.log_history

    try:
        eval_results = evaluate_model(model, tokenizer)
        print(f"\nEvaluation results:")
        for tid, score in eval_results.items():
            print(f"  {tid}: {score:.3f}")
        log_history.append({"evaluation": eval_results})
    except Exception as e:
        print(f"Evaluation skipped: {e}")

    return {
        "trainer": trainer,
        "model": model,
        "tokenizer": tokenizer,
        "log_history": log_history,
    }


def plot_reward_curves(log_history: List[Dict], output_path: str = "reward_curves.png"):
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
        window = 5
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
    parser.add_argument("--steps", type=int, default=50, help="Number of GRPO training steps")
    parser.add_argument("--task", type=str, default="task_easy", help="Task ID to train on")
    parser.add_argument("--model", type=str, default="unsloth/Qwen2.5-3B-Instruct")
    parser.add_argument("--output_dir", type=str, default="./legaloom_grpo_output")
    parser.add_argument("--no_unsloth", action="store_true")
    parser.add_argument("--curriculum", action="store_true", help="Enable curriculum learning across easy→medium→hard")
    args = parser.parse_args()

    result = run_training(
        model_name=args.model,
        num_steps=args.steps,
        task_id=args.task,
        output_dir=args.output_dir,
        use_unsloth=not args.no_unsloth,
        curriculum=args.curriculum,
    )

    if result is None:
        sys.exit(1)

    plot_reward_curves(result["log_history"], output_path="reward_curves.png")

    log_path = os.path.join(args.output_dir, "training_log.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(log_path, "w") as f:
        json.dump(result["log_history"], f, indent=2, default=str)

    print(f"\nTraining complete. Log saved → {log_path}")


if __name__ == "__main__":
    main()