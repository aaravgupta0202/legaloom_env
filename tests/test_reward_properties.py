"""
Property-based tests for the reward function.

Unlike example tests (which check specific known cases), property tests
generate thousands of random submissions and verify structural invariants
that should hold for ALL inputs:

    1. Score is always in [0, 1]
    2. Output dict has the contract keys (score, feedback, breakdown)
    3. Submitting nonsense (negative amounts, empty section) never crashes
    4. Submitting an exact copy of ground truth always scores high (≥ 0.7)
    5. Submitting random wrong data scores < 1.0
    6. The grader is deterministic — same input → same output
    7. Wrong section + wrong rate + wrong amount cannot exceed 0.5
    8. The no_tds path correctly handles every ground_truth.tds_applicable value

These are the properties that make the reward function "coherent" in the
Round 2 PDF sense. If any of them fail, the reward function has a bug
that even the unit tests didn't catch.

Run:
    pytest tests/test_reward_properties.py -v --hypothesis-show-statistics
"""
from __future__ import annotations

from hypothesis import given, strategies as st, settings, HealthCheck

from server.graders import grade_submission


# ---------------------------------------------------------------------------
# Strategies — generate plausible submissions and ground truths
# ---------------------------------------------------------------------------

# Submission generator: realistic ranges, occasional invalid values
submission_strategy = st.fixed_dictionaries({
    "tds_amount_inr": st.one_of(
        st.floats(min_value=-100.0, max_value=10_000_000.0, allow_nan=False, allow_infinity=False),
        st.just(0.0),
    ),
    "section": st.sampled_from([
        "194J", "194C", "194I", "194H", "194T", "194Q", "206AA", "no_tds",
        "194Z_invalid", "", "FAKE", "194j",  # case + invalid
    ]),
    "rate_percent": st.floats(min_value=0.0, max_value=30.0, allow_nan=False, allow_infinity=False),
    "no_tds": st.sampled_from(["true", "false", "True", "", None]),
})

# Ground truth generator: covers both no_tds and TDS-applicable cases
gt_strategy = st.fixed_dictionaries({
    "tds_applicable": st.booleans(),
    "section": st.sampled_from(["194J", "194C", "194I", "194H", "194T", "194Q", "206AA"]),
    "tds_rate_percent": st.sampled_from([0.1, 2.0, 5.0, 10.0, 20.0]),
    "tds_amount_inr": st.floats(min_value=0.0, max_value=1_000_000.0, allow_nan=False),
    "pan_valid": st.booleans(),
    "goods_amount": st.floats(min_value=0.0, max_value=500_000.0, allow_nan=False),
    "note": st.sampled_from(["", "GST bundled in TDS base", "regular invoice"]),
    "taxable_amount": st.floats(min_value=0.0, max_value=1_000_000.0, allow_nan=False),
})

task_id_strategy = st.sampled_from([
    "task_easy", "task_medium", "task_hard", "task_expert"
])


# ---------------------------------------------------------------------------
# Property 1: Score is always in [0, 1]
# ---------------------------------------------------------------------------

@given(submission=submission_strategy, gt=gt_strategy, task_id=task_id_strategy)
@settings(max_examples=200, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
def test_score_always_in_unit_interval(submission, gt, task_id):
    """No matter what's submitted, score must be in [0, 1]."""
    result = grade_submission(submission, gt, task_id=task_id)
    score = result["score"]
    assert 0.0 <= score <= 1.0, (
        f"Score out of bounds: {score}\n"
        f"submission: {submission}\n"
        f"ground_truth: {gt}\n"
        f"task_id: {task_id}"
    )


# ---------------------------------------------------------------------------
# Property 2: Output contract is always satisfied
# ---------------------------------------------------------------------------

REQUIRED_KEYS = {"score", "feedback", "breakdown"}


@given(submission=submission_strategy, gt=gt_strategy, task_id=task_id_strategy)
@settings(max_examples=100, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
def test_output_contract(submission, gt, task_id):
    """grade_submission always returns the contract keys."""
    result = grade_submission(submission, gt, task_id=task_id)
    missing = REQUIRED_KEYS - set(result.keys())
    assert not missing, f"Missing keys: {missing}"
    assert isinstance(result["feedback"], list)
    assert isinstance(result["breakdown"], dict)
    assert isinstance(result["score"], float)


# ---------------------------------------------------------------------------
# Property 3: Grader never crashes
# ---------------------------------------------------------------------------

@given(submission=submission_strategy, gt=gt_strategy, task_id=task_id_strategy)
@settings(max_examples=200, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
def test_grader_never_raises(submission, gt, task_id):
    """No matter how malformed the submission, grader returns a result.

    Robustness property: a model emitting garbage during training (a real
    failure mode for small models) cannot crash the trainer.
    """
    try:
        result = grade_submission(submission, gt, task_id=task_id)
        assert result is not None
    except Exception as e:
        # Hypothesis will shrink to find the minimal crash case
        raise AssertionError(
            f"Grader raised on:\n"
            f"  submission: {submission}\n"
            f"  ground_truth: {gt}\n"
            f"  task_id: {task_id}\n"
            f"  error: {type(e).__name__}: {e}"
        )


# ---------------------------------------------------------------------------
# Property 4: Exact-match submission scores high
# ---------------------------------------------------------------------------

@given(gt=gt_strategy, task_id=task_id_strategy)
@settings(max_examples=100, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
def test_exact_match_scores_high(gt, task_id):
    """Submitting ground truth values verbatim must score ≥ 0.7.

    Exception: when pan_valid=False (inoperative PAN), the correct answer is
    Section 206AA at 20%, not the underlying section/rate. The grader
    correctly penalizes the literal-match in that case, so we skip it here.
    A separate test below covers the inoperative-PAN exact-match path.
    """
    if not gt["pan_valid"] and gt["tds_applicable"]:
        # 206AA override applies — literal section/rate match isn't the
        # "exact" answer in the grader's eyes. Skip; covered separately.
        return

    if not gt["tds_applicable"]:
        # no_tds case
        submission = {
            "tds_amount_inr": 0.0,
            "section": "no_tds",
            "rate_percent": 0.0,
            "no_tds": "true",
        }
    else:
        submission = {
            "tds_amount_inr": gt["tds_amount_inr"],
            "section": gt["section"],
            "rate_percent": gt["tds_rate_percent"],
            "no_tds": "false",
        }
    result = grade_submission(submission, gt, task_id=task_id)
    score = result["score"]
    assert score >= 0.7, (
        f"Exact-match submission scored only {score}\n"
        f"submission: {submission}\n"
        f"ground_truth: {gt}\n"
        f"task_id: {task_id}\n"
        f"breakdown: {result.get('breakdown')}\n"
        f"feedback: {result.get('feedback')}"
    )


@given(
    gt=st.fixed_dictionaries({
        "tds_applicable": st.just(True),
        "section": st.sampled_from(["194J", "194C", "194I", "194H"]),
        "tds_rate_percent": st.just(20.0),  # 206AA override applied
        "tds_amount_inr": st.floats(min_value=100.0, max_value=500_000.0, allow_nan=False),
        "pan_valid": st.just(False),  # inoperative PAN
        "goods_amount": st.just(0.0),
        "note": st.just(""),
        "taxable_amount": st.floats(min_value=100.0, max_value=500_000.0, allow_nan=False),
    }),
    task_id=task_id_strategy,
)
@settings(max_examples=50, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
def test_inoperative_pan_exact_match_scores_high(gt, task_id):
    """For inoperative PAN: submitting the underlying section + 20% override
    rate + correct amount must score ≥ 0.7. (The grader expects the underlying
    section like '194J', not the literal string '206AA' — the rate of 20%
    is what indicates the 206AA override has been applied.)"""
    submission = {
        "tds_amount_inr": gt["tds_amount_inr"],
        "section": gt["section"],
        "rate_percent": 20.0,
        "no_tds": "false",
    }
    result = grade_submission(submission, gt, task_id=task_id)
    assert result["score"] >= 0.7, (
        f"Inoperative-PAN exact-match scored only {result['score']}\n"
        f"submission: {submission}, gt: {gt}, task_id: {task_id}\n"
        f"breakdown: {result.get('breakdown')}"
    )


# ---------------------------------------------------------------------------
# Property 5: Determinism
# ---------------------------------------------------------------------------

@given(submission=submission_strategy, gt=gt_strategy, task_id=task_id_strategy)
@settings(max_examples=100, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
def test_grader_is_deterministic(submission, gt, task_id):
    """Same input → same output. No reliance on time, randomness, or env state."""
    r1 = grade_submission(submission, gt, task_id=task_id)
    r2 = grade_submission(submission, gt, task_id=task_id)
    r3 = grade_submission(submission, gt, task_id=task_id)
    assert r1["score"] == r2["score"] == r3["score"], (
        f"Non-deterministic: {r1['score']} vs {r2['score']} vs {r3['score']}\n"
        f"submission: {submission}, gt: {gt}, task_id: {task_id}"
    )


# ---------------------------------------------------------------------------
# Property 6: Wrong-everything cannot score above 0.5
# ---------------------------------------------------------------------------

@given(gt=gt_strategy, task_id=task_id_strategy)
@settings(max_examples=100, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
def test_wrong_everything_caps_at_half(gt, task_id):
    """If section, rate, AND amount are all wrong, score must be ≤ 0.5."""
    if not gt["tds_applicable"]:
        # For no_tds: claim TDS is applicable when it isn't
        wrong_submission = {
            "tds_amount_inr": 99999.99,
            "section": "194Z_FAKE",
            "rate_percent": 99.0,
            "no_tds": "false",
        }
    else:
        # Pick a section different from gt
        wrong_section = "WRONG_SECT" if gt["section"] != "WRONG_SECT" else "OTHER"
        wrong_submission = {
            "tds_amount_inr": gt["tds_amount_inr"] + 9999.0 + 1,  # off by ≥ ₹1
            "section": wrong_section,
            "rate_percent": (gt["tds_rate_percent"] + 5.0) % 30.0 + 1.0,  # different rate
            "no_tds": "false",
        }
    result = grade_submission(wrong_submission, gt, task_id=task_id)
    assert result["score"] <= 0.5, (
        f"Wrong-everything submission scored {result['score']} (should be ≤ 0.5)\n"
        f"wrong: {wrong_submission}\n"
        f"gt: {gt}, task_id: {task_id}\n"
        f"breakdown: {result.get('breakdown')}"
    )


# ---------------------------------------------------------------------------
# Property 7: no_tds=true on a TDS-applicable case loses points
# ---------------------------------------------------------------------------

@given(gt=gt_strategy, task_id=task_id_strategy)
@settings(max_examples=100, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
def test_no_tds_when_tds_applies_loses(gt, task_id):
    """Claiming no_tds=true when TDS does apply must score below 0.5."""
    if not gt["tds_applicable"]:
        return  # property doesn't apply
    bad_submission = {
        "tds_amount_inr": 0.0,
        "section": "no_tds",
        "rate_percent": 0.0,
        "no_tds": "true",
    }
    result = grade_submission(bad_submission, gt, task_id=task_id)
    assert result["score"] < 0.5, (
        f"Wrongly claiming no_tds scored {result['score']} on TDS-applicable case\n"
        f"gt: {gt}, task_id: {task_id}"
    )


# ---------------------------------------------------------------------------
# Property 8: Submitting TDS when no_tds was correct loses points
# ---------------------------------------------------------------------------

@given(gt=gt_strategy, task_id=task_id_strategy)
@settings(max_examples=100, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
def test_submitting_tds_when_no_tds_correct_loses(gt, task_id):
    """Submitting a non-zero TDS amount when no_tds was correct must score 0."""
    if gt["tds_applicable"]:
        return  # only applies to no_tds ground truths
    bad_submission = {
        "tds_amount_inr": 50000.0,
        "section": "194J",
        "rate_percent": 10.0,
        "no_tds": "false",
    }
    result = grade_submission(bad_submission, gt, task_id=task_id)
    # Should score very low (the grader handles no_tds with raw_score 0 then
    # clamp_score, which produces near-zero, not exactly 0)
    assert result["score"] <= 0.1, (
        f"Submitting TDS when no_tds was correct scored {result['score']}\n"
        f"gt: {gt}, task_id: {task_id}"
    )
