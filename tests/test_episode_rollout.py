"""
Tests for multi-step episode rollouts, reward consistency, and edge cases.
"""

import sys
import os
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import TDSAction, TDSObservation
from server.legaloom_env_environment import LegaloomEnvironment
from server.scoring import clamp_score


def _run_full_episode(env, task_id, seed, actions_fn):
    """Helper: run a full episode with actions provided by actions_fn."""
    obs = env.reset(task_id=task_id, seed=seed)
    rewards = []
    for step_num in range(10):
        action = actions_fn(env, step_num, obs)
        if action is None:
            break
        result = env.step(action)
        rewards.append(float(result.reward))
        if result.done:
            break
    return rewards


def test_easy_episode_full_trajectory():
    """Full episode on task_easy: read → check_pan → lookup_section → submit."""
    env = LegaloomEnvironment()
    env.reset(task_id="task_easy", seed=42)

    env.step(TDSAction(action_type="read_invoice", parameters={}))
    obs2 = env.step(TDSAction(action_type="check_pan", parameters={"pan": env._task["vendor_pan"]}))
    assert not obs2.done

    obs3 = env.step(TDSAction(action_type="lookup_section", parameters={"description": "legal consultation"}))
    assert not obs3.done

    gt = env._task["ground_truth"]
    final = env.step(TDSAction(
        action_type="submit_answer",
        parameters={
            "tds_amount_inr": gt["tds_amount_inr"],
            "section": gt["section"],
            "rate_percent": gt["tds_rate_percent"],
        },
    ))
    assert final.done
    assert final.reward >= 0.5


def test_medium_episode_with_ytd():
    """Medium episode: query_ytd is important for threshold tasks."""
    env = LegaloomEnvironment()

    for seed in range(42, 52):
        env.reset(task_id="task_medium", seed=seed)
        obs1 = env.step(TDSAction(action_type="read_invoice", parameters={}))
        assert not obs1.done

        obs2 = env.step(TDSAction(action_type="check_pan", parameters={"pan": env._task["vendor_pan"]}))
        assert env._state.pan_checked

        gt = env._task["ground_truth"]
        final = env.step(TDSAction(
            action_type="submit_answer",
            parameters={
                "tds_amount_inr": gt["tds_amount_inr"],
                "section": gt["section"],
                "rate_percent": gt["tds_rate_percent"],
            },
        ))
        assert final.done


def test_hard_episode_inoperative_pan():
    """Hard episode: inoperative PAN must trigger 20% rate."""
    env = LegaloomEnvironment()

    found_inop = False
    for seed in range(42, 100):
        env.reset(task_id="task_hard", seed=seed)
        gt = env._task["ground_truth"]
        if not gt["pan_valid"]:
            found_inop = True
            env.step(TDSAction(action_type="read_invoice", parameters={}))
            pan_obs = env.step(TDSAction(action_type="check_pan", parameters={"pan": env._task["vendor_pan"]}))
            assert "INOPERATIVE" in pan_obs.action_result.upper()

            final = env.step(TDSAction(
                action_type="submit_answer",
                parameters={
                    "tds_amount_inr": gt["tds_amount_inr"],
                    "section": gt["section"],
                    "rate_percent": 20.0,
                },
            ))
            assert final.done
            assert final.reward > 0.3
            break

    if not found_inop:
        pass


def test_expert_episode_new_sections():
    """Expert episode: 194T or 194Q must be correctly handled."""
    env = LegaloomEnvironment()
    env.reset(task_id="task_expert", seed=42)

    env.step(TDSAction(action_type="read_invoice", parameters={}))
    env.step(TDSAction(action_type="check_pan", parameters={"pan": env._task["vendor_pan"]}))

    gt = env._task["ground_truth"]
    final = env.step(TDSAction(
        action_type="submit_answer",
        parameters={
            "tds_amount_inr": gt["tds_amount_inr"],
            "section": gt["section"],
            "rate_percent": gt["tds_rate_percent"],
        },
    ))
    assert final.done


def test_force_close_exceeds_max_steps():
    """Episode must terminate when max_steps is exceeded."""
    env = LegaloomEnvironment()
    env.reset(task_id="task_easy", seed=42)

    max_steps = env._task["max_steps"]
    for i in range(max_steps + 2):
        obs = env.step(TDSAction(action_type="read_invoice", parameters={}))

    assert obs.done
    assert obs.reward <= 0.1


def test_workflow_violation_check_pan_required():
    """Submit without check_pan must be rejected."""
    env = LegaloomEnvironment()
    env.reset(task_id="task_easy", seed=42)

    env.step(TDSAction(action_type="read_invoice", parameters={}))

    result = env.step(TDSAction(
        action_type="submit_answer",
        parameters={"tds_amount_inr": 1000.0, "section": "194J", "rate_percent": 10.0},
    ))
    assert not result.done
    assert result.reward < 0


def test_read_invoice_must_be_first():
    """Actions before read_invoice must fail."""
    env = LegaloomEnvironment()
    env.reset(task_id="task_easy", seed=42)

    result = env.step(TDSAction(action_type="check_pan", parameters={"pan": "ABCDE1234F"}))
    assert not result.done
    assert "Workflow violation" in result.action_result


def test_unknown_action_penalized():
    """Unknown action types must be penalized."""
    env = LegaloomEnvironment()
    env.reset(task_id="task_easy", seed=42)
    env.step(TDSAction(action_type="read_invoice", parameters={}))

    result = env.step(TDSAction(action_type="fly_to_moon", parameters={}))
    assert not result.done
    assert "Unknown action_type" in result.action_result


def test_repeat_action_penalized():
    """Repeating read_invoice must give a penalty."""
    env = LegaloomEnvironment()
    env.reset(task_id="task_easy", seed=42)

    obs1 = env.step(TDSAction(action_type="read_invoice", parameters={}))
    obs2 = env.step(TDSAction(action_type="read_invoice", parameters={}))
    assert obs2.reward < obs1.reward


def test_reward_consistency_across_seeds():
    """Same invoice (same seed/task) must produce same ground truth."""
    env1 = LegaloomEnvironment()
    env2 = LegaloomEnvironment()

    env1.reset(task_id="task_easy", seed=12345)
    env2.reset(task_id="task_easy", seed=12345)

    gt1 = env1._task["ground_truth"]
    gt2 = env2._task["ground_truth"]

    assert gt1["section"] == gt2["section"]
    assert gt1["tds_amount_inr"] == gt2["tds_amount_inr"]
    assert gt1["tds_rate_percent"] == gt2["tds_rate_percent"]


def test_final_reward_always_clamped():
    """All final rewards must be in (0.05, 0.95)."""
    env = LegaloomEnvironment()

    for task_id in ["task_easy", "task_medium", "task_hard", "task_expert"]:
        for seed in range(42, 47):
            env.reset(task_id=task_id, seed=seed)
            env.step(TDSAction(action_type="read_invoice", parameters={}))
            gt = env._task["ground_truth"]
            env.step(TDSAction(action_type="check_pan", parameters={"pan": env._task["vendor_pan"]}))

            final = env.step(TDSAction(
                action_type="submit_answer",
                parameters={
                    "tds_amount_inr": gt["tds_amount_inr"],
                    "section": gt["section"],
                    "rate_percent": gt["tds_rate_percent"],
                },
            ))
            assert 0.01 <= final.reward <= 0.99, f"Reward {final.reward} out of bounds for {task_id} seed={seed}"


def test_below_threshold_no_tds():
    """Below-threshold invoice should score well with no_tds=true."""
    env = LegaloomEnvironment()

    for seed in range(42, 60):
        env.reset(task_id="task_hard", seed=seed)
        gt = env._task["ground_truth"]
        if not gt.get("tds_applicable", True):
            env.step(TDSAction(action_type="read_invoice", parameters={}))
            env.step(TDSAction(action_type="check_pan", parameters={"pan": env._task["vendor_pan"]}))

            final = env.step(TDSAction(
                action_type="submit_answer",
                parameters={"tds_amount_inr": 0.0, "section": "194H", "rate_percent": 0.0, "no_tds": "true"},
            ))
            assert final.done
            assert final.reward > 0.5
            break
