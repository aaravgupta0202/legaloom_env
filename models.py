"""
Pydantic models for LegaLoom-Env.

The only constraint from the hackathon: reward must be strictly in (0, 1).
We enforce this with a simple clamp — no artificial floors/ceilings beyond that.
The validator clamps ALL values unconditionally (the original had an `if v > 0.0:`
guard that let 0.0 and negatives pass through unchanged).
"""

from typing import Any, Dict, List, Optional
from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field, field_validator


def _clamp(v: float) -> float:
    """Clamp to strictly open (0, 1) — use a tiny epsilon so we never hit the boundary."""
    v = float(v)
    return round(min(max(v, 1e-6), 1 - 1e-6), 6)


class TDSAction(Action):
    """
    Agent action for the TDS compliance environment.

    Valid action_types:
      read_invoice    — fetch the invoice text (always first step)
      check_pan       — verify vendor PAN status
      check_threshold — check if cumulative YTD payments cross threshold
      query_ytd       — query year-to-date payments for a vendor
      lookup_section  — identify which TDS section applies to a service
      query_law       — fetch law text for a section
      submit_answer   — submit final TDS amount in INR (ends episode)
    """
    action_type: str = Field(
        ...,
        description="One of: read_invoice | check_pan | check_threshold | query_ytd | lookup_section | query_law | submit_answer",
    )
    parameters: Dict[str, Any] = Field(default_factory=dict)


class TDSObservation(Observation):
    """
    Environment response after each agent action.
    reward is always strictly in (0, 1) — the validator enforces this for every value.
    """
    reward: float = Field(default=0.1)

    @field_validator("reward", mode="before")
    @classmethod
    def clamp_reward(cls, v: float) -> float:
        """
        Unconditionally clamp to strictly open (0, 1).
        Original had `if v > 0.0:` guard — removed so 0.0 and negatives are also clamped.
        """
        return _clamp(v)

    invoice_text:      str       = Field(default="")
    action_result:     str       = Field(default="")
    available_actions: List[str] = Field(default_factory=lambda: ["read_invoice"])
    steps_used:        int       = Field(default=0)
    max_steps:         int       = Field(default=8)
    hint:              str       = Field(default="")


class TDSState(State):
    task_id:            str  = Field(default="")
    difficulty:         str  = Field(default="")
    pan_checked:        bool = Field(default=False)
    section_identified: bool = Field(default=False)
    answer_submitted:   bool = Field(default=False)
