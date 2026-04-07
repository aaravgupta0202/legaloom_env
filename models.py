"""
Pydantic models for LegaLoom-Env.
Defines the typed Action, Observation, and State contracts
that the agent, server, and client all share.
"""

from typing import Any, Dict, List, Optional
from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field


# ---------------------------------------------------------------------------
# Action — what the agent sends to the environment each step
# ---------------------------------------------------------------------------

class TDSAction(Action):
    """
    One step the agent takes inside the TDS compliance environment.

    The agent picks an action_type and supplies relevant parameters.
    Valid action_types:

      "read_invoice"     — request the full invoice text (first step)
      "check_pan"        — verify PAN status for a given PAN number
      "check_threshold"  — ask if cumulative YTD payments cross the TDS threshold
      "lookup_section"   — ask what TDS section applies to a described service
      "submit_answer"    — submit final TDS deduction in INR (terminates episode)
    """

    action_type: str = Field(
        ...,
        description=(
            "One of: read_invoice | check_pan | check_threshold "
            "| lookup_section | submit_answer"
        ),
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Action-specific parameters. Examples:\n"
            "  check_pan       -> {pan: 'ABCDE1234F'}\n"
            "  check_threshold -> {section: '194J', amount: 85000}\n"
            "  lookup_section  -> {description: 'IT support contract'}\n"
            "  submit_answer   -> {tds_amount_inr: 8500.0, section: '194J', "
            "rate_percent: 10.0}"
        ),
    )


# ---------------------------------------------------------------------------
# Observation — what the environment returns after each step
# ---------------------------------------------------------------------------

class TDSObservation(Observation):
    """
    What the agent sees after each action.

    Inherits from Observation which already provides:
      done   : bool
      reward : Optional[float]

    We override reward to always be float (never None) for spec compliance.
    """

    # Override reward to always be float — spec requires numeric reward
    reward: float = Field(
        default=0.0,
        description="Reward for this step. Always a float (0.0 when no reward).",
    )

    # The invoice text — populated on first read_invoice action
    invoice_text: str = Field(
        default="",
        description="Full invoice text. Populated after read_invoice action.",
    )

    # Result of the agent's last action in plain English
    action_result: str = Field(
        default="",
        description="Environment's response to the agent's last action.",
    )

    # Which actions the agent can legally take right now
    available_actions: List[str] = Field(
        default_factory=lambda: ["read_invoice"],
        description="List of valid action_type values for the next step.",
    )

    # Running step counter so agent knows how many steps it has used
    steps_used: int = Field(
        default=0,
        description="Number of steps taken so far in this episode.",
    )

    # Max steps allowed for this task
    max_steps: int = Field(
        default=8,
        description="Maximum steps allowed before episode is force-closed.",
    )

    # General guidance — suppressed on hard task to increase difficulty
    hint: str = Field(
        default="",
        description="Optional guidance. Empty string on hard difficulty.",
    )


# ---------------------------------------------------------------------------
# State — episode metadata returned by state() endpoint
# ---------------------------------------------------------------------------

class TDSState(State):
    """
    Metadata about the current episode.

    Inherits from State which already provides:
      episode_id  : Optional[str]
      step_count  : int
    """

    task_id: str = Field(
        default="",
        description="Which task is currently active: task_easy | task_medium | task_hard",
    )

    difficulty: str = Field(
        default="",
        description="easy | medium | hard",
    )

    # Track which sub-checks the agent has completed (for reward shaping)
    pan_checked: bool = Field(
        default=False,
        description="True once agent has called check_pan.",
    )

    section_identified: bool = Field(
        default=False,
        description="True once agent has called lookup_section.",
    )

    answer_submitted: bool = Field(
        default=False,
        description="True once agent has called submit_answer.",
    )
