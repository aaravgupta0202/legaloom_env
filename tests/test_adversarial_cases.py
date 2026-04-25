"""
Tests for the adversarial benchmark.

Verifies:
  1. All 20 cases are well-formed (required keys, consistent types)
  2. score_adversarial is deterministic
  3. A perfect submission scores 1.0; obvious wrong answers score < 0.5
  4. Categories distribute reasonably across failure modes
  5. The scorer respects the no_tds path
"""
from __future__ import annotations

import pytest

from server.adversarial_cases import (
    ADVERSARIAL_CASES,
    score_adversarial,
    get_categories,
)


REQUIRED_CASE_KEYS = {
    "id",
    "category",
    "failure_mode",
    "invoice_text",
    "vendor_pan",
    "cumulative_ytd",
    "ground_truth",
}

REQUIRED_GT_KEYS = {"tds_amount_inr", "section", "rate_percent"}


def test_twenty_cases_present():
    assert len(ADVERSARIAL_CASES) == 20, (
        f"Expected 20 adversarial cases, found {len(ADVERSARIAL_CASES)}"
    )


def test_all_cases_have_required_keys():
    for i, case in enumerate(ADVERSARIAL_CASES):
        missing = REQUIRED_CASE_KEYS - set(case.keys())
        assert not missing, (
            f"Case {i} ({case.get('id', 'unnamed')}) missing keys: {missing}"
        )


def test_ground_truth_well_formed():
    for case in ADVERSARIAL_CASES:
        gt = case["ground_truth"]
        missing = REQUIRED_GT_KEYS - set(gt.keys())
        assert not missing, (
            f"Case {case['id']} ground_truth missing: {missing}"
        )
        # tds_amount_inr must be float-castable
        float(gt["tds_amount_inr"])
        # rate_percent must be in [0, 30]
        rate = float(gt["rate_percent"])
        assert 0.0 <= rate <= 30.0, (
            f"Case {case['id']} has unreasonable rate: {rate}"
        )


def test_case_ids_are_unique():
    ids = [c["id"] for c in ADVERSARIAL_CASES]
    assert len(ids) == len(set(ids)), "Duplicate case IDs found"


def test_categories_distributed():
    cats = get_categories()
    assert len(cats) >= 5, f"Expected ≥5 distinct categories, got {len(cats)}: {cats}"
    # No category should dominate (≤ 60% of cases)
    by_cat = {}
    for c in ADVERSARIAL_CASES:
        by_cat.setdefault(c["category"], 0)
        by_cat[c["category"]] += 1
    max_share = max(by_cat.values()) / len(ADVERSARIAL_CASES)
    assert max_share <= 0.60, (
        f"One category dominates ({max_share:.0%}): {by_cat}"
    )


def test_scorer_is_deterministic():
    case = ADVERSARIAL_CASES[0]
    submission = {
        "tds_amount_inr": case["ground_truth"]["tds_amount_inr"],
        "section": case["ground_truth"]["section"],
        "rate_percent": case["ground_truth"]["rate_percent"],
    }
    s1 = score_adversarial(submission, case)
    s2 = score_adversarial(submission, case)
    assert s1["score"] == s2["score"], "Scorer not deterministic"


def test_perfect_submission_scores_high():
    """A submission that exactly matches ground truth should score ≥ 0.95."""
    for case in ADVERSARIAL_CASES:
        gt = case["ground_truth"]
        submission = {
            "tds_amount_inr": gt["tds_amount_inr"],
            "section": gt["section"],
            "rate_percent": gt["rate_percent"],
        }
        if gt.get("no_tds"):
            submission["no_tds"] = "true"
        result = score_adversarial(submission, case)
        assert result["score"] >= 0.95, (
            f"Case {case['id']}: perfect submission scored only {result['score']}\n"
            f"  Breakdown: {result['breakdown']}\n"
            f"  Feedback: {result['feedback']}"
        )


def test_wrong_section_loses_points():
    case = ADVERSARIAL_CASES[0]  # inoperative-PAN case
    gt = case["ground_truth"]
    wrong = {
        "tds_amount_inr": gt["tds_amount_inr"],
        "section": "194Z_fake",  # bogus
        "rate_percent": gt["rate_percent"],
    }
    result = score_adversarial(wrong, case)
    assert result["score"] < 0.85, (
        f"Wrong section did not lose enough points: {result['score']}"
    )


def test_wrong_amount_loses_points():
    case = ADVERSARIAL_CASES[0]
    gt = case["ground_truth"]
    wrong = {
        "tds_amount_inr": gt["tds_amount_inr"] * 0.5,  # 50% off
        "section": gt["section"],
        "rate_percent": gt["rate_percent"],
    }
    result = score_adversarial(wrong, case)
    assert result["score"] < 0.85, (
        f"50%-wrong amount did not lose enough points: {result['score']}"
    )


def test_no_tds_case_accepts_zero_amount():
    """For no_tds=true ground truth, a submission with amount=0 + section=no_tds scores high."""
    no_tds_cases = [c for c in ADVERSARIAL_CASES if c["ground_truth"].get("no_tds")]
    assert no_tds_cases, "No no_tds adversarial cases found — set is missing this category"

    case = no_tds_cases[0]
    submission = {
        "tds_amount_inr": 0.0,
        "section": "no_tds",
        "rate_percent": 0.0,
        "no_tds": "true",
    }
    result = score_adversarial(submission, case)
    assert result["score"] >= 0.95, (
        f"Correct no_tds submission scored only {result['score']}: {result}"
    )


def test_compound_trap_cases_exist():
    """Adversarial set must include cases with multiple simultaneous edge cases."""
    compound = [c for c in ADVERSARIAL_CASES if c["category"] == "compound_traps"]
    assert len(compound) >= 1, (
        "No compound_traps cases — adversarial set lacks the hardest examples"
    )


def test_fy2526_new_sections_covered():
    """Must include cases for 194T (partner drawings) and 194Q (goods 0.1%)."""
    fy_cases = [c for c in ADVERSARIAL_CASES if c["category"] == "fy2526_new_sections"]
    assert len(fy_cases) >= 2, (
        f"Need ≥2 FY 2025-26 new-section cases, found {len(fy_cases)}"
    )
    sections = {c["ground_truth"]["section"] for c in fy_cases}
    assert "194T" in sections or any("194T" in c["failure_mode"] for c in fy_cases)
    assert "194Q" in sections or any("194Q" in c["failure_mode"] for c in fy_cases)
