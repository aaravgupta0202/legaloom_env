"""
Grader functions for LegaLoom-Env.
Each grader returns a score strictly in (0, 1) via _clamp().
"""

from typing import Any, Dict

AMOUNT_TOLERANCE_INR = 1.0
_SCORE_MIN = 0.05   # lowest meaningful score (wrong answer)
_SCORE_MAX = 0.95   # highest score (perfect answer)


def _clamp(score: float) -> float:
    """Map any raw score to strictly open (0, 1)."""
    return round(min(max(float(score), _SCORE_MIN), _SCORE_MAX), 4)


def grade_submission(
    params: Dict[str, Any],
    ground_truth: Dict[str, Any],
    task_id: str = "task_easy",
) -> Dict[str, Any]:
    """
    Grade a submitted answer against ground truth.
    Returns score strictly in (_SCORE_MIN, _SCORE_MAX).
    """
    submitted_amount  = float(params.get("tds_amount_inr", -1))
    submitted_section = str(params.get("section", "")).strip().upper()
    submitted_rate    = float(params.get("rate_percent", -1))
    no_tds_flag       = str(params.get("no_tds", "")).lower() == "true"

    feedback, breakdown = [], {}
    score = 0.0  # raw accumulator — scaled at the end

    # ── Case 1: TDS not applicable (below threshold) ────────────────────────
    if not ground_truth["tds_applicable"]:
        correct   = (submitted_amount == 0.0 or no_tds_flag)
        raw_score = 0.85 if correct else 0.10
        feedback.append(
            "✓ No TDS applicable — correctly identified." if correct else
            f"✗ No TDS required. Submitted INR {submitted_amount:,.2f}, should be 0."
        )
        breakdown["no_tds_correct"] = correct
        return {"score": _clamp(raw_score), "correct": correct,
                "feedback": feedback, "breakdown": breakdown}

    # ── Determine weights by task difficulty ────────────────────────────────
    goods       = ground_truth.get("goods_amount", 0.0)
    is_inop_pan = not ground_truth["pan_valid"]

    if task_id == "task_easy":
        W_PAN, W_SECT, W_RATE, W_GOODS, W_AMOUNT = 0.0, 0.30, 0.30, 0.0, 0.40
    elif task_id == "task_medium":
        if goods > 0:
            W_PAN, W_SECT, W_RATE, W_GOODS, W_AMOUNT = 0.0, 0.25, 0.15, 0.20, 0.40
        else:
            W_PAN, W_SECT, W_RATE, W_GOODS, W_AMOUNT = 0.0, 0.25, 0.15, 0.0,  0.60
    elif task_id == "task_expert":
        W_PAN, W_SECT, W_RATE, W_GOODS, W_AMOUNT = 0.0, 0.40, 0.25, 0.0, 0.35
    else:  # task_hard
        if is_inop_pan:
            W_PAN, W_SECT, W_RATE, W_GOODS, W_AMOUNT = (
                (0.35, 0.10, 0.15, 0.10, 0.30) if goods > 0 else
                (0.35, 0.10, 0.15, 0.0,  0.40)
            )
        else:
            W_PAN, W_SECT, W_RATE, W_GOODS, W_AMOUNT = (
                (0.0, 0.20, 0.15, 0.20, 0.45) if goods > 0 else
                (0.0, 0.20, 0.15, 0.0,  0.65)
            )

    # Normalise weights to sum = 1.0
    total_w = W_PAN + W_SECT + W_RATE + W_GOODS + W_AMOUNT
    if total_w > 0:
        W_PAN /= total_w; W_SECT /= total_w; W_RATE  /= total_w
        W_GOODS /= total_w; W_AMOUNT /= total_w

    # ── Inoperative PAN ─────────────────────────────────────────────────────
    if is_inop_pan:
        pan_ok = abs(submitted_rate - 20.0) < 0.01
        breakdown["pan_inoperative_detected"] = pan_ok
        if pan_ok:
            score += W_PAN
            feedback.append("✓ Inoperative PAN detected — 20% rate applied.")
        else:
            feedback.append(f"✗ PAN INOPERATIVE — must apply 20% (Sec 206AA), got {submitted_rate}%.")

    # ── Section ─────────────────────────────────────────────────────────────
    expected_section = ground_truth["section"]
    split      = expected_section in ("SPLIT", "SPLIT_194J_194I")
    section_ok = (
        submitted_section == expected_section or
        (split and submitted_section in ("194J", "194I", "194C"))
    )
    breakdown["section_correct"] = section_ok
    if section_ok:
        score += W_SECT
        feedback.append(f"✓ Section {submitted_section} correct.")
    else:
        feedback.append(f"✗ Section wrong: got {submitted_section}, expected {expected_section}.")

    # ── Rate ────────────────────────────────────────────────────────────────
    rate_ok = abs(submitted_rate - ground_truth["tds_rate_percent"]) < 0.01
    breakdown["rate_correct"] = rate_ok
    if rate_ok:
        score += W_RATE
        feedback.append(f"✓ Rate {submitted_rate}% correct.")
    else:
        feedback.append(f"✗ Rate wrong: got {submitted_rate}%, expected {ground_truth['tds_rate_percent']}%.")

    # ── Goods exclusion ─────────────────────────────────────────────────────
    if goods > 0:
        goods_ok = submitted_amount <= ground_truth["taxable_amount"] + AMOUNT_TOLERANCE_INR
        breakdown["goods_excluded"] = goods_ok
        if goods_ok:
            score += W_GOODS
            feedback.append(f"✓ Goods (INR {goods:,.0f}) correctly excluded from TDS base.")
        else:
            feedback.append(f"✗ Goods (INR {goods:,.0f}) must be excluded — TDS on services only.")

    # ── Final TDS amount ─────────────────────────────────────────────────────
    amount_ok = abs(submitted_amount - ground_truth["tds_amount_inr"]) <= AMOUNT_TOLERANCE_INR
    breakdown["amount_correct"] = amount_ok
    if amount_ok:
        score += W_AMOUNT
        feedback.append(f"✓ TDS INR {submitted_amount:,.2f} CORRECT (expected INR {ground_truth['tds_amount_inr']:,.2f}).")
    else:
        feedback.append(f"✗ Amount wrong: got INR {submitted_amount:,.2f}, expected INR {ground_truth['tds_amount_inr']:,.2f}.")

    # ── Scale raw [0,1] score into strictly open (_SCORE_MIN, _SCORE_MAX) ───
    # score=0.0 (all wrong) → _SCORE_MIN; score=1.0 (perfect) → _SCORE_MAX
    scaled  = _SCORE_MIN + score * (_SCORE_MAX - _SCORE_MIN)
    correct = amount_ok and (section_ok or is_inop_pan)
    return {"score": _clamp(scaled), "correct": correct,
            "feedback": feedback, "breakdown": breakdown}


def grade_easy(p, gt):   return grade_submission(p, gt, "task_easy")["score"]
def grade_medium(p, gt): return grade_submission(p, gt, "task_medium")["score"]
def grade_hard(p, gt):   return grade_submission(p, gt, "task_hard")["score"]
def grade_expert(p, gt): return grade_submission(p, gt, "task_expert")["score"]

GRADERS = {
    "task_easy":   grade_easy,
    "task_medium": grade_medium,
    "task_hard":   grade_hard,
    "task_expert": grade_expert,
}
