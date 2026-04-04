"""
Pydantic models for LegaLoom-Env.
Defines the typed Action, Observation, and State contracts
that the agent, server, and client all share.
"""

from typing import Any, Dict, List, Optional
from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field


class TDSAction(Action):
    """
    One step the agent takes inside the TDS compliance environment.

    Valid action_types:
      read_invoice     — request the full invoice text (first step)
      check_pan        — verify PAN status for a given PAN number
      check_threshold  — ask if cumulative YTD payments cross the TDS threshold
      lookup_section   — ask what TDS section applies to a described service
      submit_answer    — submit final TDS deduction in INR (terminates episode)
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


class TDSObservation(Observation):
    """
    What the agent sees after each action.
    Inherits done and reward from Observation base class.
    """
    invoice_text: str = Field(
        default="",
        description="Full invoice text. Populated after read_invoice action.",
    )
    action_result: str = Field(
        default="",
        description="Environment's response to the agent's last action.",
    )
    available_actions: List[str] = Field(
        default_factory=lambda: ["read_invoice"],
        description="List of valid action_type values for the next step.",
    )
    steps_used: int = Field(
        default=0,
        description="Number of steps taken so far in this episode.",
    )
    max_steps: int = Field(
        default=8,
        description="Maximum steps allowed before episode is force-closed.",
    )
    hint: str = Field(
        default="",
        description="Optional guidance. Empty string on hard difficulty.",
    )


class TDSState(State):
    """
    Metadata about the current episode.
    Inherits episode_id and step_count from State base class.
    """
    task_id: str = Field(default="")
    difficulty: str = Field(default="")
    pan_checked: bool = Field(default=False)
    section_identified: bool = Field(default=False)
    answer_submitted: bool = Field(default=False)