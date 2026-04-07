"""
Explicit grader functions for LegaLoom-Env.

Each grader takes the submitted answer and ground truth,
and returns a normalised score in [0.0, 1.0].
These functions are deterministic — same inputs always give same output.
"""

from typing import Dict, Any

AMOUNT_TOLERANCE_INR = 1.0


def grade_submission(params: Dict[str, Any], ground_truth: Dict[str, Any]) -> Dict[str, Any]:
    """
    Primary grader. Evaluates a submit_answer action against ground truth.

    Args:
        params       : dict from submit_answer parameters
        ground_truth : invoice ground truth dict from invoice_db.json

    Returns:
        {
            "score":    float in [0.0, 1.0],
            "correct":  bool,
            "feedback": list of str,
            "breakdown": dict
        }
    """
    submitted_amount  = float(params.get("tds_amount_inr", -1))
    submitted_section = str(params.get("section", "")).strip().upper()
    submitted_rate    = float(params.get("rate_percent", -1))
    no_tds_flag       = str(params.get("no_tds", "")).lower() == "true"

    feedback   = []
    breakdown  = {}
    score      = 0.0

    # ── Case 1: No TDS applicable (below threshold) ──────────────────
    if not ground_truth["tds_applicable"]:
        correct = (submitted_amount == 0.0 or no_tds_flag)
        if correct:
            score = 1.0
            feedback.append("✓ Correctly identified: no TDS applicable (below threshold).")
        else:
            score = 0.0
            feedback.append(
                f"✗ No TDS required (below threshold). "
                f"Submitted INR {submitted_amount:,.2f} — should be 0."
            )
        breakdown["no_tds_correct"] = correct
        score = round(min(max(score, 0.0), 1.0), 4)
        return {"score": score, "correct": correct, "feedback": feedback, "breakdown": breakdown}

    # ── Case 2: Inoperative PAN — must be 20% ────────────────────────
    if not ground_truth["pan_valid"]:
        pan_ok = abs(submitted_rate - 20.0) < 0.01
        breakdown["pan_inoperative_detected"] = pan_ok
        if pan_ok:
            score += 0.40
            feedback.append("✓ Inoperative PAN detected — 20% rate applied.")
        else:
            feedback.append(
                f"✗ PAN is INOPERATIVE. Must apply 20% (Section 206AA). "
                f"Submitted {submitted_rate}%."
            )

    # ── Case 3: Section correct ───────────────────────────────────────
    expected_section = ground_truth["section"]
    split = expected_section in ("SPLIT", "SPLIT_194J_194I")
    section_ok = (
        submitted_section == expected_section or
        (split and submitted_section in ("194J", "194I", "194C"))
    )
    breakdown["section_correct"] = section_ok
    if section_ok:
        score += 0.20
        feedback.append(f"✓ Section {submitted_section} is correct.")
    else:
        feedback.append(f"✗ Section wrong: submitted {submitted_section}, expected {expected_section}.")

    # ── Case 4: Rate correct ──────────────────────────────────────────
    rate_ok = abs(submitted_rate - ground_truth["tds_rate_percent"]) < 0.01
    breakdown["rate_correct"] = rate_ok
    if rate_ok:
        score += 0.10
        feedback.append(f"✓ Rate {submitted_rate}% is correct.")
    else:
        feedback.append(
            f"✗ Rate wrong: submitted {submitted_rate}%, "
            f"expected {ground_truth['tds_rate_percent']}%."
        )

    # ── Case 5: Goods exclusion ───────────────────────────────────────
    goods = ground_truth.get("goods_amount", 0.0)
    if goods > 0:
        goods_ok = submitted_amount <= ground_truth["taxable_amount"] + AMOUNT_TOLERANCE_INR
        breakdown["goods_excluded"] = goods_ok
        if goods_ok:
            score += 0.10
            feedback.append(f"✓ Goods (INR {goods:,.0f}) correctly excluded from TDS base.")
        else:
            feedback.append(f"✗ Goods (INR {goods:,.0f}) must be excluded — TDS on services only.")

    # ── Case 6: Final amount ──────────────────────────────────────────
    amount_ok = abs(submitted_amount - ground_truth["tds_amount_inr"]) <= AMOUNT_TOLERANCE_INR
    breakdown["amount_correct"] = amount_ok
    if amount_ok:
        score += 0.40 if goods == 0 else 0.30
        feedback.append(
            f"✓ TDS amount INR {submitted_amount:,.2f} is CORRECT "
            f"(expected INR {ground_truth['tds_amount_inr']:,.2f})."
        )
    else:
        feedback.append(
            f"✗ TDS amount wrong: submitted INR {submitted_amount:,.2f}, "
            f"correct is INR {ground_truth['tds_amount_inr']:,.2f}."
        )

    correct = amount_ok and (section_ok or not ground_truth["pan_valid"])
    score   = round(min(max(score, 0.0), 1.0), 4)  # guaranteed [0.0, 1.0], no float jitter

    return {"score": score, "correct": correct, "feedback": feedback, "breakdown": breakdown}


def grade_easy(params: Dict[str, Any], ground_truth: Dict[str, Any]) -> float:
    """Grader for task_easy. Returns score in [0.0, 1.0]."""
    return grade_submission(params, ground_truth)["score"]


def grade_medium(params: Dict[str, Any], ground_truth: Dict[str, Any]) -> float:
    """Grader for task_medium. Returns score in [0.0, 1.0]."""
    return grade_submission(params, ground_truth)["score"]


def grade_hard(params: Dict[str, Any], ground_truth: Dict[str, Any]) -> float:
    """Grader for task_hard. Returns score in [0.0, 1.0]."""
    return grade_submission(params, ground_truth)["score"]


def grade_expert(params: Dict[str, Any], ground_truth: Dict[str, Any]) -> float:
    """Grader for task_expert. Returns score in [0.0, 1.0]."""
    return grade_submission(params, ground_truth)["score"]


# Map task_id → grader function
GRADERS = {
    "task_easy":   grade_easy,
    "task_medium": grade_medium,
    "task_hard":   grade_hard,
    "task_expert": grade_expert,
}
