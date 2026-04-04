"""
LegaLoom-Env — TDS Compliance Environment.
"""

from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import TDSAction, TDSObservation, TDSState
except ImportError:
    from models import TDSAction, TDSObservation, TDSState

try:
    from .tds_rules import get_rate, threshold_crossed, compute_tds, TDS_SECTIONS
    from .pan_registry import is_pan_valid, is_company, pan_status_message
    from .tasks import get_task, all_task_ids
except ImportError:
    from server.tds_rules import get_rate, threshold_crossed, compute_tds, TDS_SECTIONS
    from server.pan_registry import is_pan_valid, is_company, pan_status_message
    from server.tasks import get_task, all_task_ids


AMOUNT_TOLERANCE_INR = 1.0


class LegaloomEnvironment(Environment):

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = TDSState()
        self._task = None
        self._invoice_read = False
        self._reward_earned = {}
        self._episode_reward = 0.0

    def reset(self, task_id: str = "task_easy", **kwargs) -> TDSObservation:
        if task_id not in all_task_ids():
            task_id = "task_easy"

        self._task = get_task(task_id)
        self._invoice_read = False
        self._reward_earned = {k: False for k in self._task["reward_breakpoints"]}
        self._episode_reward = 0.0

        self._state = TDSState(
            episode_id=str(uuid4()),
            step_count=0,
            task_id=task_id,
            difficulty=self._task["difficulty"],
            pan_checked=False,
            section_identified=False,
            answer_submitted=False,
        )

        return TDSObservation(
            done=False,
            reward=0.0,
            invoice_text="",
            action_result=(
                f"New episode started. Task: {task_id} "
                f"({self._task['difficulty']} difficulty). "
                f"You have {self._task['max_steps']} steps. "
                "Start with action_type='read_invoice' to see the invoice."
            ),
            available_actions=["read_invoice"],
            steps_used=0,
            max_steps=self._task["max_steps"],
            hint=self._build_hint(),
        )

    def step(self, action: TDSAction, **kwargs) -> TDSObservation:
        # Guard: if reset() was never called, auto-initialise with task_easy
        if self._task is None:
            self.reset(task_id="task_easy")

        self._state.step_count += 1
        steps_used = self._state.step_count
        max_steps = self._task["max_steps"]

        if steps_used > max_steps:
            return self._force_close(steps_used, max_steps)

        action_type = (action.action_type or "").strip().lower()
        params = action.parameters or {}

        if action_type == "read_invoice":
            return self._handle_read_invoice(steps_used, max_steps)
        elif action_type == "check_pan":
            return self._handle_check_pan(params, steps_used, max_steps)
        elif action_type == "check_threshold":
            return self._handle_check_threshold(params, steps_used, max_steps)
        elif action_type == "lookup_section":
            return self._handle_lookup_section(params, steps_used, max_steps)
        elif action_type == "submit_answer":
            return self._handle_submit_answer(params, steps_used)
        else:
            return TDSObservation(
                done=False,
                reward=-0.05,
                invoice_text=self._task["invoice"] if self._invoice_read else "",
                action_result=(
                    f"Unknown action_type: '{action_type}'. "
                    "Valid: read_invoice, check_pan, check_threshold, "
                    "lookup_section, submit_answer."
                ),
                available_actions=self._available_actions(),
                steps_used=steps_used,
                max_steps=max_steps,
                hint=self._build_hint(),
            )

    def _handle_read_invoice(self, steps_used, max_steps):
        self._invoice_read = True
        return TDSObservation(
            done=False,
            reward=0.0,
            invoice_text=self._task["invoice"],
            action_result=(
                "Invoice retrieved. Read it carefully. "
                "Note the vendor PAN, service description, and amount. "
                "Recommended next step: check_pan to verify PAN status."
            ),
            available_actions=self._available_actions(),
            steps_used=steps_used,
            max_steps=max_steps,
            hint=self._build_hint(),
        )

    def _handle_check_pan(self, params, steps_used, max_steps):
        pan = str(params.get("pan", "")).strip().upper()
        if not pan:
            return TDSObservation(
                done=False,
                reward=0.0,
                invoice_text=self._task["invoice"] if self._invoice_read else "",
                action_result='check_pan requires parameter pan. Example: {"pan": "ABCDE1234F"}',
                available_actions=self._available_actions(),
                steps_used=steps_used,
                max_steps=max_steps,
                hint=self._build_hint(),
            )

        reward = self._award("pan_checked")
        self._state.pan_checked = True
        result = pan_status_message(pan)

        return TDSObservation(
            done=False,
            reward=reward,
            invoice_text=self._task["invoice"] if self._invoice_read else "",
            action_result=result,
            available_actions=self._available_actions(),
            steps_used=steps_used,
            max_steps=max_steps,
            hint=self._build_hint(),
        )

    def _handle_check_threshold(self, params, steps_used, max_steps):
        section = str(params.get("section", "")).strip().upper()
        amount = float(params.get("amount", 0))
        ytd = self._task.get("cumulative_ytd", 0.0)

        if not section:
            result = 'check_threshold requires section and amount. Example: {"section": "194J", "amount": 85000}'
        elif section not in TDS_SECTIONS:
            result = f"Unknown section '{section}'. Valid: {list(TDS_SECTIONS.keys())}"
        else:
            crossed = threshold_crossed(section, amount, ytd)
            s = TDS_SECTIONS[section]
            result = (
                f"Section {section} threshold check for INR {amount:,.0f}: "
                f"{'TDS IS applicable' if crossed else 'TDS NOT applicable — below threshold'}. "
                f"(Single limit: INR {s['threshold_single']:,} | "
                f"Annual limit: INR {s['threshold_annual']:,} | "
                f"YTD paid so far: INR {ytd:,.0f})"
            )

        return TDSObservation(
            done=False,
            reward=0.0,
            invoice_text=self._task["invoice"] if self._invoice_read else "",
            action_result=result,
            available_actions=self._available_actions(),
            steps_used=steps_used,
            max_steps=max_steps,
            hint=self._build_hint(),
        )

    def _handle_lookup_section(self, params, steps_used, max_steps):
        description = str(params.get("description", "")).strip().lower()
        reward = 0.0

        if not description:
            result = 'lookup_section requires description. Example: {"description": "legal consultation"}'
        else:
            matched = self._match_section(description)
            if matched:
                s = TDS_SECTIONS[matched]
                result = (
                    f"Best match for '{description}': Section {matched} — "
                    f"{s['description']}. "
                    f"Nature: {s['nature']}. "
                    f"Default rate: {s['rate_default']}%."
                )
                gt_section = self._task["ground_truth"]["section"]
                if matched == gt_section:
                    reward = self._award("section_correct")
                    self._state.section_identified = True
            else:
                result = (
                    f"Could not match '{description}' to a known TDS section. "
                    "Try a more specific description of the service."
                )

        return TDSObservation(
            done=False,
            reward=reward,
            invoice_text=self._task["invoice"] if self._invoice_read else "",
            action_result=result,
            available_actions=self._available_actions(),
            steps_used=steps_used,
            max_steps=max_steps,
            hint=self._build_hint(),
        )

    def _handle_submit_answer(self, params, steps_used):
        self._state.answer_submitted = True
        gt = self._task["ground_truth"]

        submitted_amount = float(params.get("tds_amount_inr", -1))
        submitted_rate   = float(params.get("rate_percent", -1))
        reward = 0.0
        feedback_parts = []

        if not gt["pan_valid"]:
            if submitted_rate == 20.0 or str(params.get("pan_status", "")).lower() == "inoperative":
                reward += self._award("pan_inoperative_identified")
                feedback_parts.append("Correctly identified inoperative PAN.")
            else:
                feedback_parts.append("MISSED: PAN was inoperative — 20% rate should have been applied.")

        if abs(submitted_rate - gt["tds_rate_percent"]) < 0.01:
            reward += self._award("rate_correct")
            feedback_parts.append(f"Rate {submitted_rate}% is correct.")
        else:
            feedback_parts.append(
                f"Rate incorrect: submitted {submitted_rate}%, "
                f"correct is {gt['tds_rate_percent']}%."
            )

        amount_correct = abs(submitted_amount - gt["tds_amount_inr"]) <= AMOUNT_TOLERANCE_INR
        if amount_correct:
            reward += self._award("amount_exact")
            feedback_parts.append(f"TDS amount INR {submitted_amount:,.2f} is CORRECT.")
        else:
            feedback_parts.append(
                f"TDS amount incorrect: submitted INR {submitted_amount:,.2f}, "
                f"correct is INR {gt['tds_amount_inr']:,.2f}."
            )

        if "goods_excluded" in self._task["reward_breakpoints"]:
            if submitted_amount <= gt["taxable_amount"] + AMOUNT_TOLERANCE_INR:
                reward += self._award("goods_excluded")
                feedback_parts.append("Correctly excluded goods from TDS calculation.")

        total_score = min(self._episode_reward, 1.0)
        result = (
            f"Episode complete. {' | '.join(feedback_parts)} "
            f"Final score: {total_score:.3f}."
        )

        return TDSObservation(
            done=True,
            reward=reward,
            invoice_text=self._task["invoice"],
            action_result=result,
            available_actions=[],
            steps_used=steps_used,
            max_steps=self._task["max_steps"],
            hint="",
        )

    def _award(self, breakpoint_key: str) -> float:
        if self._reward_earned.get(breakpoint_key, True):
            return 0.0
        reward = self._task["reward_breakpoints"].get(breakpoint_key, 0.0)
        self._reward_earned[breakpoint_key] = True
        self._episode_reward += reward
        return reward

    def _available_actions(self) -> list:
        actions = []
        if not self._invoice_read:
            actions.append("read_invoice")
        else:
            actions.extend(["check_pan", "check_threshold",
                            "lookup_section", "submit_answer"])
        seen = set()
        return [a for a in actions if not (a in seen or seen.add(a))]

    def _build_hint(self) -> str:
        if self._task and self._task["difficulty"] == "hard":
            return ""
        if not self._invoice_read:
            return "Start by calling read_invoice to see the invoice."
        if not self._state.pan_checked:
            return "Call check_pan with the PAN number from the invoice."
        if not self._state.section_identified:
            return "Call lookup_section with the service description from the invoice."
        return "Ready to submit. Call submit_answer with tds_amount_inr, section, and rate_percent."

    def _match_section(self, description: str) -> str:
        description = description.lower()
        keyword_map = [
            ("194I", ["rent", "lease", "office space", "warehouse",
                      "equipment hire", "machinery hire", "vehicle hire"]),
            ("194H", ["commission", "brokerage", "referral fee",
                      "agency fee", "dealer margin"]),
            ("194C", ["catering", "housekeeping", "security",
                      "manpower", "labour", "event management",
                      "transportation", "printing", "contract work"]),
            ("194J", ["legal", "audit", "consulting", "advisory",
                      "chartered accountant", "medical", "engineering",
                      "software", "it support", "cloud", "technical",
                      "maintenance", "data processing", "support contract"]),
        ]
        for section_code, keywords in keyword_map:
            if any(kw in description for kw in keywords):
                return section_code
        return ""

    def _force_close(self, steps_used, max_steps):
        return TDSObservation(
            done=True,
            reward=0.0,
            invoice_text=self._task["invoice"] if self._invoice_read else "",
            action_result=(
                f"Episode terminated: exceeded {max_steps} steps without submitting. "
                "Score: 0.000."
            ),
            available_actions=[],
            steps_used=steps_used,
            max_steps=max_steps,
            hint="",
        )

    @property
    def state(self) -> TDSState:
        return self._state