"""
Unified scoring policy for LegaLoom-Env.
Single source of truth for score/reward normalization.

Hackathon rule: every value in the [END] rewards= list must be
strictly between 0 and 1 (not 0.0 and not 1.0).
We enforce this at the boundary with SCORE_MIN=0.01 / SCORE_MAX=0.99
so even .2f formatting never produces "0.00" or "1.00".
"""
from __future__ import annotations
from typing import Final

SCORE_MIN: Final[float] = 0.01
SCORE_MAX: Final[float] = 0.99
ROUND_DIGITS: Final[int] = 4

STEP_REWARD_MIN: Final[float] = -1.0
STEP_REWARD_MAX: Final[float]  = 1.0


def clamp_score(value: float) -> float:
    """Clamp final episode score to [SCORE_MIN, SCORE_MAX]."""
    v = float(value)
    return round(min(max(v, SCORE_MIN), SCORE_MAX), ROUND_DIGITS)


def normalize_step_reward(value: float) -> float:
    """Clamp intermediate step rewards to [-1, 1]."""
    v = float(value)
    return round(min(max(v, STEP_REWARD_MIN), STEP_REWARD_MAX), ROUND_DIGITS)
