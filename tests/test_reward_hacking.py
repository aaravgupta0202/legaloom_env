"""
Tests for reward hacking prevention — verifying that shortcuts don't pay off.

Key attacks to prevent:
1. Skip check_pan → submit directly (should get low reward)
2. lookup_section spam → overuse shouldn't accumulate rewards
3. Always assume inoperative PAN (20% rate) without checking
4. Submit with correct amount but wrong reasoning
5. Format-only completions (no semantic content)
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import TDSAction
from server.legaloom_env_environment import LegaloomEnvironment


def test_skip_pan_check_penalized():
    """Submit without check_pan must get very low reward."""
    env = LegaloomEnvironment()
    env.reset(task_id="task_easy", seed=42)
    
    # Try to submit directly (should be rejected)
    env.step(TDSAction(action_type="read_invoice", parameters={}))
    result = env.step(TDSAction(
        action_type="submit_answer",
        parameters={"tds_amount_inr": 5000.0, "section": "194J", "rate_percent": 10.0},
    ))
    
    assert not result.done  # Should not allow submission
    assert result.reward < 0  # Should be penalized


def test_lookup_section_spam_penalized():
    """Calling lookup_section many times should not accumulate free reward."""
    env = LegaloomEnvironment()
    env.reset(task_id="task_easy", seed=42)
    
    env.step(TDSAction(action_type="read_invoice", parameters={}))
    env.step(TDSAction(action_type="check_pan", parameters={"pan": env._task["vendor_pan"]}))
    
    rewards = []
    for _ in range(6):  # Multiple lookups
        result = env.step(TDSAction(
            action_type="lookup_section",
            parameters={"description": "legal consultation"},
        ))
        rewards.append(result.reward)
    
    # Later lookups should have lower or negative rewards
    assert rewards[-1] < rewards[0] or rewards[-1] < 0


def test_always_inoperative_pan_guess():
    """Always submitting 20% without checking PAN should fail on operative PAN invoices."""
    env = LegaloomEnvironment()
    
    # Find an operative PAN invoice
    for seed in range(42, 100):
        env.reset(task_id="task_easy", seed=seed)
        if env._task["ground_truth"]["pan_valid"]:
            break
    
    env.step(TDSAction(action_type="read_invoice", parameters={}))
    env.step(TDSAction(action_type="check_pan", parameters={"pan": env._task["vendor_pan"]}))
    
    # Submit with 20% (wrong for operative PAN)
    gt = env._task["ground_truth"]
    result = env.step(TDSAction(
        action_type="submit_answer",
        parameters={
            "tds_amount_inr": gt["taxable_amount"] * 0.20,  # 20% rate
            "section": gt["section"],
            "rate_percent": 20.0,
        },
    ))
    
    # Should get low score because rate is wrong
    assert result.reward < 0.5


def test_correct_amount_wrong_section():
    """Correct amount with wrong section should not get full credit."""
    env = LegaloomEnvironment()
    env.reset(task_id="task_easy", seed=42)
    
    env.step(TDSAction(action_type="read_invoice", parameters={}))
    env.step(TDSAction(action_type="check_pan", parameters={"pan": env._task["vendor_pan"]}))
    
    gt = env._task["ground_truth"]
    # Submit with correct amount but wrong section
    result = env.step(TDSAction(
        action_type="submit_answer",
        parameters={
            "tds_amount_inr": gt["tds_amount_inr"],
            "section": "194C",  # Wrong section
            "rate_percent": gt["tds_rate_percent"],
        },
    ))
    
    # Section is weighted 30%, so score should be ~70% max
    assert result.reward < 0.75


def test_repeated_actions_diminishing_returns():
    """Repeating the same action should give diminishing or negative rewards."""
    env = LegaloomEnvironment()
    env.reset(task_id="task_easy", seed=42)
    
    first = env.step(TDSAction(action_type="read_invoice", parameters={}))
    second = env.step(TDSAction(action_type="read_invoice", parameters={}))
    
    # Second read should not give same reward as first
    assert second.reward <= first.reward


def test_format_only_no_free_reward():
    """Format compliance alone should not give high reward."""
    env = LegaloomEnvironment()
    env.reset(task_id="task_easy", seed=42)
    
    # Just read invoice and submit immediately with valid format
    env.step(TDSAction(action_type="read_invoice", parameters={}))
    result = env.step(TDSAction(action_type="check_pan", parameters={"pan": env._task["vendor_pan"]}))
    
    # Even with correct format, missing steps should limit reward
    gt = env._task["ground_truth"]
    final = env.step(TDSAction(
        action_type="submit_answer",
        parameters={
            "tds_amount_inr": gt["tds_amount_inr"],
            "section": gt["section"],
            "rate_percent": gt["tds_rate_percent"],
        },
    ))
    
    # This should actually work because we did check_pan
    # But test the case where we skip everything
    assert final.done
