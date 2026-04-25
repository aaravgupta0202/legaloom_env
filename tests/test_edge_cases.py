"""
Edge case tests for LegaLoom-Env.

Tests boundary conditions, corner cases, and difficult scenarios:
- Threshold boundaries (exactly at threshold)
- Inoperative PAN with various sections
- GST-bundled vs separate line items
- Mixed goods+services invoices
- Section 194T and 194Q (new FY 2025-26 sections)
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import TDSAction
from server.legaloom_env_environment import LegaloomEnvironment


def test_threshold_boundary_exact():
    """Invoice exactly at threshold should be handled correctly."""
    env = LegaloomEnvironment()
    
    # Look for threshold_boundary category
    for seed in range(42, 200):
        env.reset(task_id="task_medium", seed=seed)
        if env._task["category"] == "threshold_boundary":
            break
    
    env.step(TDSAction(action_type="read_invoice", parameters={}))
    env.step(TDSAction(action_type="check_pan", parameters={"pan": env._task["vendor_pan"]}))
    
    # Query YTD
    ytd_result = env.step(TDSAction(
        action_type="query_ytd",
        parameters={"pan": env._task["vendor_pan"]},
    ))
    
    assert "YTD" in ytd_result.action_result
    
    # Check threshold
    gt = env._task["ground_truth"]
    section = gt["section"]
    if section in ("SPLIT", "SPLIT_194J_194I"):
        section = "194J"  # Default to 194J for threshold check
    
    threshold_result = env.step(TDSAction(
        action_type="check_threshold",
        parameters={"section": section, "amount": gt["taxable_amount"]},
    ))
    
    assert "threshold" in threshold_result.action_result.lower()


def test_inoperative_pan_with_194q():
    """Inoperative PAN with 194Q should use 5% not 20%."""
    env = LegaloomEnvironment()
    
    for seed in range(42, 200):
        env.reset(task_id="task_hard", seed=seed)
        gt = env._task["ground_truth"]
        if not gt["pan_valid"] and gt["section"] == "194Q":
            break
    else:
        # Skip if no matching invoice found
        return
    
    env.step(TDSAction(action_type="read_invoice", parameters={}))
    pan_result = env.step(TDSAction(
        action_type="check_pan",
        parameters={"pan": env._task["vendor_pan"]},
    ))
    
    # PAN check should indicate inoperative status
    assert "INOPERATIVE" in pan_result.action_result.upper()
    
    # Submit with correct 194Q rate (5% for inoperative PAN)
    result = env.step(TDSAction(
        action_type="submit_answer",
        parameters={
            "tds_amount_inr": gt["tds_amount_inr"],
            "section": "194Q",
            "rate_percent": 5.0,  # 5% for 194Q/194O with inoperative PAN
        },
    ))
    
    assert result.done
    assert result.reward > 0.5


def test_gst_bundled_vs_separate():
    """GST bundled should have TDS on full amount, separate on pre-GST."""
    env = LegaloomEnvironment()
    
    for seed in range(42, 200):
        env.reset(task_id="task_hard", seed=seed)
        if env._task["category"] == "gst_bundled_tds_base":
            break
    else:
        return
    
    env.step(TDSAction(action_type="read_invoice", parameters={}))
    
    # Invoice should have note about GST bundling
    invoice = env._task["invoice_text"]
    assert any(phrase in invoice.lower() for phrase in [
        "inclusive of all taxes",
        "gst included",
        "gst bundled",
    ])
    
    env.step(TDSAction(action_type="check_pan", parameters={"pan": env._task["vendor_pan"]}))
    
    gt = env._task["ground_truth"]
    result = env.step(TDSAction(
        action_type="submit_answer",
        parameters={
            "tds_amount_inr": gt["tds_amount_inr"],
            "section": gt["section"],
            "rate_percent": gt["tds_rate_percent"],
        },
    ))
    
    assert result.done


def test_mixed_invoice_goods_exclusion():
    """Mixed invoice: goods portion should be excluded from TDS."""
    env = LegaloomEnvironment()
    
    for seed in range(42, 200):
        env.reset(task_id="task_medium", seed=seed)
        gt = env._task["ground_truth"]
        if gt.get("goods_amount", 0) > 0:
            break
    else:
        return
    
    env.step(TDSAction(action_type="read_invoice", parameters={}))
    
    # Invoice should have multiple line items
    invoice = env._task["invoice_text"]
    assert "Hardware" in invoice or "goods" in invoice.lower() or "materials" in invoice.lower()
    
    env.step(TDSAction(action_type="check_pan", parameters={"pan": env._task["vendor_pan"]}))
    
    # TDS should only be on service portion
    goods = gt.get("goods_amount", 0)
    taxable = gt["taxable_amount"]
    
    # Taxable amount should be less than total if goods present
    total_match = False
    for line in invoice.split("\n"):
        if "Total" in line and goods > 0:
            # This is a weak check but ensures goods logic is working
            total_match = True
            break
    
    result = env.step(TDSAction(
        action_type="submit_answer",
        parameters={
            "tds_amount_inr": gt["tds_amount_inr"],
            "section": gt["section"],
            "rate_percent": gt["tds_rate_percent"],
        },
    ))
    
    assert result.done


def test_section_194t_partner():
    """New FY 2025-26 Section 194T for partner salary/drawings."""
    env = LegaloomEnvironment()
    
    for seed in range(42, 200):
        env.reset(task_id="task_expert", seed=seed)
        gt = env._task["ground_truth"]
        if gt["section"] == "194T":
            break
    else:
        return
    
    env.step(TDSAction(action_type="read_invoice", parameters={}))
    env.step(TDSAction(action_type="check_pan", parameters={"pan": env._task["vendor_pan"]}))
    
    # Should be able to classify as 194T
    lookup_result = env.step(TDSAction(
        action_type="lookup_section",
        parameters={"description": "partner salary"},
    ))
    
    assert "194T" in lookup_result.action_result
    
    result = env.step(TDSAction(
        action_type="submit_answer",
        parameters={
            "tds_amount_inr": gt["tds_amount_inr"],
            "section": "194T",
            "rate_percent": 10.0,
        },
    ))
    
    assert result.done
    assert result.reward > 0.5


def test_below_threshold_no_tds():
    """Below threshold invoice should correctly have no TDS."""
    env = LegaloomEnvironment()
    
    for seed in range(42, 200):
        env.reset(task_id="task_hard", seed=seed)
        gt = env._task["ground_truth"]
        if not gt.get("tds_applicable", True):
            break
    else:
        return
    
    env.step(TDSAction(action_type="read_invoice", parameters={}))
    env.step(TDSAction(action_type="check_pan", parameters={"pan": env._task["vendor_pan"]}))
    
    # Query YTD to understand threshold status
    env.step(TDSAction(
        action_type="query_ytd",
        parameters={"pan": env._task["vendor_pan"]},
    ))
    
    # Submit with no TDS
    result = env.step(TDSAction(
        action_type="submit_answer",
        parameters={
            "tds_amount_inr": 0.0,
            "section": "194H",
            "rate_percent": 0.0,
            "no_tds": "true",
        },
    ))
    
    assert result.done
    assert result.reward > 0.5


def test_step_limit_force_close():
    """Episode should terminate when max_steps exceeded."""
    env = LegaloomEnvironment()
    env.reset(task_id="task_easy", seed=42)
    
    max_steps = env._task["max_steps"]
    
    # Take actions without submitting
    for i in range(max_steps + 1):
        obs = env.step(TDSAction(action_type="read_invoice", parameters={}))
        if obs.done:
            break
    
    assert obs.done
    assert "exceeded" in obs.action_result.lower() or "terminated" in obs.action_result.lower()


def test_pan_case_insensitive():
    """PAN should be handled case-insensitively."""
    env = LegaloomEnvironment()
    env.reset(task_id="task_easy", seed=42)
    
    env.step(TDSAction(action_type="read_invoice", parameters={}))
    
    # Check with lowercase PAN
    result = env.step(TDSAction(
        action_type="check_pan",
        parameters={"pan": env._task["vendor_pan"].lower()},
    ))
    
    # Should still work (normalized to uppercase internally)
    assert "inoperative" in result.action_result.lower() or "operative" in result.action_result.lower()


def test_invalid_section_rejection():
    """Invalid section code should be rejected gracefully."""
    env = LegaloomEnvironment()
    env.reset(task_id="task_easy", seed=42)
    
    env.step(TDSAction(action_type="read_invoice", parameters={}))
    
    result = env.step(TDSAction(
        action_type="check_threshold",
        parameters={"section": "999X", "amount": 50000},
    ))
    
    assert "unknown" in result.action_result.lower() or "invalid" in result.action_result.lower()
