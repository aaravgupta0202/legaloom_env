"""
LegaLoom-Env — TDS Compliance Environment (Upgraded)

Upgraded features vs v1:
  - 260-invoice database (random sampling per episode)
  - 4 difficulty levels (easy / medium / hard / expert)
  - New actions: check_threshold, query_ytd, query_law
  - GST base logic (TDS on pre-GST or gross depending on invoice)
  - Goods exclusion tracking
  - All FY 2025-26 rules (194T, updated thresholds)
  - Reward shaped across every reasoning step
"""

from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import TDSAction, TDSObservation, TDSState
except ImportError:
    from models import TDSAction, TDSObservation, TDSState

try:
    from .tds_rules import (get_rate, threshold_crossed, compute_tds,
                             TDS_SECTIONS, section_summary, classify_service)
    from .pan_registry import is_pan_valid, is_company, pan_status_message
    from .tasks import get_task, all_task_ids, sample_task
except ImportError:
    from server.tds_rules import (get_rate, threshold_crossed, compute_tds,
                                   TDS_SECTIONS, section_summary, classify_service)
    from server.pan_registry import is_pan_valid, is_company, pan_status_message
    from server.tasks import get_task, all_task_ids, sample_task


AMOUNT_TOLERANCE_INR = 1.0


class LegaloomEnvironment(Environment):

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state         = TDSState()
        self._task          = None
        self._invoice_read  = False
        self._ytd_queried   = False
        self._law_queried   = False
        self._reward_earned = {}
        self._episode_reward= 0.0
        self._current_task_id = "task_easy"

    # ------------------------------------------------------------------
    # reset()
    # ------------------------------------------------------------------

    def reset(self, task_id: str = "task_easy", seed: int = None, **kwargs) -> TDSObservation:
        if task_id not in all_task_ids():
            task_id = "task_easy"

        self._task          = sample_task(task_id, seed=seed)
        self._current_task_id = task_id
        self._invoice_read  = False
        self._ytd_queried   = False
        self._law_queried   = False
        self._reward_earned = {k: False for k in self._task["reward_breakpoints"]}
        self._episode_reward= 0.0

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
                f"Invoice: {self._task['invoice_id']}. "
                f"You have {self._task['max_steps']} steps. "
                "Start with action_type='read_invoice'."
            ),
            available_actions=["read_invoice"],
            steps_used=0,
            max_steps=self._task["max_steps"],
            hint=self._build_hint(),
        )

    # ------------------------------------------------------------------
    # step()
    # ------------------------------------------------------------------

    def step(self, action: TDSAction, **kwargs) -> TDSObservation:
        if self._task is None:
            self.reset(task_id=self._current_task_id)

        self._state.step_count += 1
        steps_used = self._state.step_count
        max_steps  = self._task["max_steps"]

        if steps_used > max_steps:
            return self._force_close(steps_used, max_steps)

        action_type = (action.action_type or "").strip().lower()
        params      = action.parameters or {}

        handlers = {
            "read_invoice":    self._handle_read_invoice,
            "check_pan":       self._handle_check_pan,
            "check_threshold": self._handle_check_threshold,
            "query_ytd":       self._handle_query_ytd,
            "lookup_section":  self._handle_lookup_section,
            "query_law":       self._handle_query_law,
            "submit_answer":   self._handle_submit_answer,
        }

        handler = handlers.get(action_type)
        if handler:
            if action_type == "submit_answer":
                return handler(params, steps_used)
            return handler(params, steps_used, max_steps)

        return TDSObservation(
            done=False, reward=-0.05,
            invoice_text=self._invoice_text(),
            action_result=(
                f"Unknown action_type: '{action_type}'. "
                "Valid: read_invoice, check_pan, check_threshold, "
                "query_ytd, lookup_section, query_law, submit_answer."
            ),
            available_actions=self._available_actions(),
            steps_used=steps_used, max_steps=max_steps,
            hint=self._build_hint(),
        )

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------

    def _handle_read_invoice(self, params, steps_used, max_steps):
        self._invoice_read = True
        gt = self._task["ground_truth"]
        guidance = "Note the vendor PAN, service description, and amount(s)."
        if gt.get("goods_amount", 0) > 0:
            guidance += " This invoice has multiple line items — check each carefully."
        if not gt["pan_valid"]:
            guidance += " IMPORTANT: Always verify PAN status before computing TDS."
        return TDSObservation(
            done=False, reward=0.0,
            invoice_text=self._task["invoice_text"],
            action_result=f"Invoice retrieved. {guidance}",
            available_actions=self._available_actions(),
            steps_used=steps_used, max_steps=max_steps,
            hint=self._build_hint(),
        )

    def _handle_check_pan(self, params, steps_used, max_steps):
        pan = str(params.get("pan", "")).strip().upper()
        if not pan:
            return self._error_obs(
                'check_pan requires parameter pan. Example: {"pan": "ABCDE1234F"}',
                steps_used, max_steps
            )

        reward = self._award("pan_checked")
        self._state.pan_checked = True

        # Use our PAN registry if available, else check against ground truth
        from server.pan_registry import PAN_DB
        record = PAN_DB.get(pan)
        if record:
            result = pan_status_message(pan)
        else:
            # Fall back to ground truth
            gt = self._task["ground_truth"]
            vendor_pan = self._task["vendor_pan"]
            if pan == vendor_pan:
                if gt["pan_valid"]:
                    result = f"PAN {pan} is operative and valid. TDS at section rate applies."
                else:
                    result = (
                        f"PAN {pan} is INOPERATIVE (not linked to Aadhaar). "
                        "TDS rate: 20% regardless of section — Section 206AA applies."
                    )
            else:
                result = f"PAN {pan} not found in registry. Verify the PAN from the invoice."

        # Award extra if correctly flagging inoperative
        gt = self._task["ground_truth"]
        if not gt["pan_valid"] and "INOPERATIVE" in result.upper():
            reward += self._award("pan_inoperative_flagged")

        return TDSObservation(
            done=False, reward=reward,
            invoice_text=self._invoice_text(),
            action_result=result,
            available_actions=self._available_actions(),
            steps_used=steps_used, max_steps=max_steps,
            hint=self._build_hint(),
        )

    def _handle_check_threshold(self, params, steps_used, max_steps):
        section = str(params.get("section", "")).strip().upper()
        amount  = float(params.get("amount", 0))
        ytd     = self._task.get("cumulative_ytd", 0.0)

        if not section:
            return self._error_obs(
                'check_threshold requires section and amount. Example: {"section": "194J", "amount": 85000}',
                steps_used, max_steps
            )
        if section not in TDS_SECTIONS:
            return self._error_obs(
                f"Unknown section '{section}'. Valid: {list(TDS_SECTIONS.keys())}",
                steps_used, max_steps
            )

        crossed = threshold_crossed(section, amount, ytd)
        s = TDS_SECTIONS[section]
        result = (
            f"Section {section} threshold check — Invoice: INR {amount:,.0f} | "
            f"YTD paid: INR {ytd:,.0f} | Running total: INR {ytd+amount:,.0f}. "
            f"Single limit: INR {s['threshold_single']:,} | "
            f"Annual limit: INR {s['threshold_annual']:,}. "
            f"Result: {'TDS IS applicable — threshold crossed.' if crossed else 'TDS NOT applicable — below threshold. No deduction required.'}"
        )

        reward = self._award("threshold_checked")

        return TDSObservation(
            done=False, reward=reward,
            invoice_text=self._invoice_text(),
            action_result=result,
            available_actions=self._available_actions(),
            steps_used=steps_used, max_steps=max_steps,
            hint=self._build_hint(),
        )

    def _handle_query_ytd(self, params, steps_used, max_steps):
        pan = str(params.get("pan", "")).strip().upper()
        ytd = self._task.get("cumulative_ytd", 0.0)
        self._ytd_queried = True

        if not pan:
            result = (
                f"YTD payments to vendor for current financial year: "
                f"INR {ytd:,.0f}. "
                "This is the cumulative amount paid BEFORE this invoice."
            )
        else:
            vendor_pan = self._task["vendor_pan"]
            if pan == vendor_pan:
                result = (
                    f"YTD payments to vendor PAN {pan} in FY 2024-25: "
                    f"INR {ytd:,.0f}. "
                    "Add the current invoice amount to determine if annual threshold is crossed."
                )
            else:
                result = f"No payment history found for PAN {pan} in current financial year."

        return TDSObservation(
            done=False, reward=0.0,
            invoice_text=self._invoice_text(),
            action_result=result,
            available_actions=self._available_actions(),
            steps_used=steps_used, max_steps=max_steps,
            hint=self._build_hint(),
        )

    def _handle_lookup_section(self, params, steps_used, max_steps):
        description = str(params.get("description", "")).strip()
        if not description:
            return self._error_obs(
                'lookup_section requires description. Example: {"description": "legal consultation"}',
                steps_used, max_steps
            )

        gt = self._task["ground_truth"]
        vendor_pan = self._task["vendor_pan"]
        is_company_vendor = is_company(vendor_pan)

        result_dict = classify_service(description, vendor_is_company=is_company_vendor)
        matched_section = result_dict["section"]
        matched_rate    = result_dict["rate"]

        result = (
            f"Service classification for '{description}': "
            f"Section {matched_section} — {TDS_SECTIONS.get(matched_section, {}).get('description', 'Unknown')}. "
            f"Applicable rate: {matched_rate}% "
            f"(vendor is {'company' if is_company_vendor else 'individual/LLP'}). "
            f"Confidence: {result_dict.get('confidence', 'medium')}."
        )

        # Award if correct section matched
        reward = 0.0
        expected_section = gt["section"]
        if matched_section == expected_section and expected_section not in ("SPLIT", "SPLIT_194J_194I"):
            reward = self._award("section_correct")
            self._state.section_identified = True

        return TDSObservation(
            done=False, reward=reward,
            invoice_text=self._invoice_text(),
            action_result=result,
            available_actions=self._available_actions(),
            steps_used=steps_used, max_steps=max_steps,
            hint=self._build_hint(),
        )

    def _handle_query_law(self, params, steps_used, max_steps):
        section = str(params.get("section", "")).strip().upper()
        self._law_queried = True

        if not section or section not in TDS_SECTIONS:
            # Return overview of all sections
            result = "TDS Sections overview (FY 2025-26):\n"
            for code, data in TDS_SECTIONS.items():
                result += (
                    f"  {code}: {data['description']} — "
                    f"Rate: {data['rate_default']}% — "
                    f"Threshold: INR {data.get('threshold_annual',0):,}/year\n"
                )
            result += "\nUse query_law with section parameter for details. E.g.: {\"section\": \"194J\"}"
        else:
            s = TDS_SECTIONS[section]
            result = (
                f"Section {section} — {s['description']} (FY 2025-26)\n"
                f"Nature: {s['nature']}\n"
                f"Default rate: {s['rate_default']}%\n"
            )
            if "rate_company" in s:
                result += f"Company rate: {s['rate_company']}%\n"
            if "rate_technical" in s:
                result += f"Technical services rate: {s['rate_technical']}%\n"
            if "rate_machinery" in s:
                result += f"Machinery/equipment rate: {s['rate_machinery']}%\n"
            result += (
                f"No-PAN / inoperative PAN rate: {s['rate_no_pan']}% (Section 206AA)\n"
                f"Single payment threshold: INR {s['threshold_single']:,}\n"
                f"Annual threshold: INR {s['threshold_annual']:,}\n"
            )

        return TDSObservation(
            done=False, reward=0.0,
            invoice_text=self._invoice_text(),
            action_result=result,
            available_actions=self._available_actions(),
            steps_used=steps_used, max_steps=max_steps,
            hint=self._build_hint(),
        )

    def _handle_submit_answer(self, params, steps_used) -> TDSObservation:
        self._state.answer_submitted = True
        gt     = self._task["ground_truth"]
        reward = 0.0
        fb     = []

        submitted_amount  = float(params.get("tds_amount_inr", -1))
        submitted_section = str(params.get("section", "")).strip().upper()
        submitted_rate    = float(params.get("rate_percent", -1))

        # --- PAN inoperative check ---
        if not gt["pan_valid"]:
            if submitted_rate == 20.0 or str(params.get("pan_status","")).lower() == "inoperative":
                reward += self._award("pan_inoperative_flagged")
                fb.append("✓ Correctly identified inoperative PAN — 20% override applied.")
            else:
                fb.append(f"✗ PAN was INOPERATIVE. Should apply 20% (Section 206AA), not {submitted_rate}%.")

        # --- No TDS case ---
        if not gt["tds_applicable"]:
            if submitted_amount == 0.0 or str(params.get("no_tds","")).lower() == "true":
                reward += self._award("amount_exact")
                fb.append("✓ Correctly identified no TDS applicable — below threshold.")
            else:
                fb.append(f"✗ No TDS should be deducted (below threshold). Submitted INR {submitted_amount:,.2f}.")
            return self._end_episode(reward, fb, steps_used)

        # --- Rate check ---
        if abs(submitted_rate - gt["tds_rate_percent"]) < 0.01:
            reward += self._award("rate_correct") if "rate_correct" in self._task["reward_breakpoints"] else 0.0
            fb.append(f"✓ Rate {submitted_rate}% is correct.")
        else:
            fb.append(f"✗ Rate incorrect: submitted {submitted_rate}%, correct is {gt['tds_rate_percent']}%.")

        # --- Goods exclusion check ---
        goods = gt.get("goods_amount", 0.0)
        if goods > 0:
            if submitted_amount <= gt["taxable_amount"] + AMOUNT_TOLERANCE_INR:
                reward += self._award("goods_excluded")
                fb.append(f"✓ Correctly excluded goods (INR {goods:,.0f}) from TDS base.")
            else:
                fb.append(f"✗ Goods (INR {goods:,.0f}) should be excluded from TDS. TDS on services only.")

        # --- GST base check ---
        if self._task["category"] == "gst_bundled_tds_base":
            # Correct answer uses full (bundled) amount
            if abs(submitted_amount - gt["tds_amount_inr"]) <= AMOUNT_TOLERANCE_INR:
                reward += self._award("gst_base_correct")
                fb.append("✓ Correctly applied TDS on full (GST-inclusive) invoice amount.")

        # --- Final amount ---
        correct = abs(submitted_amount - gt["tds_amount_inr"]) <= AMOUNT_TOLERANCE_INR
        if correct:
            reward += self._award("amount_exact")
            fb.append(f"✓ TDS amount INR {submitted_amount:,.2f} is CORRECT.")
        else:
            fb.append(
                f"✗ TDS amount incorrect: submitted INR {submitted_amount:,.2f}, "
                f"correct is INR {gt['tds_amount_inr']:,.2f}."
            )

        return self._end_episode(reward, fb, steps_used)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _end_episode(self, reward, feedback_parts, steps_used) -> TDSObservation:
        total_score = min(self._episode_reward, 1.0)
        result = (
            f"Episode complete. {' | '.join(feedback_parts)} "
            f"Final score: {total_score:.3f}."
        )
        return TDSObservation(
            done=True, reward=reward,
            invoice_text=self._task["invoice_text"],
            action_result=result,
            available_actions=[],
            steps_used=steps_used,
            max_steps=self._task["max_steps"],
            hint="",
        )

    def _award(self, key: str) -> float:
        if self._reward_earned.get(key, True):
            return 0.0
        reward = self._task["reward_breakpoints"].get(key, 0.0)
        self._reward_earned[key] = True
        self._episode_reward += reward
        return reward

    def _invoice_text(self) -> str:
        return self._task["invoice_text"] if self._invoice_read else ""

    def _available_actions(self) -> list:
        if not self._invoice_read:
            return ["read_invoice"]
        actions = ["check_pan", "check_threshold", "query_ytd",
                   "lookup_section", "query_law", "submit_answer"]
        seen = set()
        return [a for a in actions if not (a in seen or seen.add(a))]

    def _build_hint(self) -> str:
        if self._task and not self._task.get("hint_enabled", True):
            return ""
        if not self._invoice_read:
            return "Start by calling read_invoice to see the invoice."
        if not self._state.pan_checked:
            return "Call check_pan with the vendor PAN from the invoice."
        if not self._state.section_identified:
            return "Call lookup_section with the service description."
        if self._ytd_queried is False and self._task.get("cumulative_ytd", 0) > 0:
            return "Call query_ytd to check cumulative payments before computing TDS."
        return "Call submit_answer with tds_amount_inr, section, and rate_percent."

    def _error_obs(self, message: str, steps_used: int, max_steps: int) -> TDSObservation:
        return TDSObservation(
            done=False, reward=0.0,
            invoice_text=self._invoice_text(),
            action_result=message,
            available_actions=self._available_actions(),
            steps_used=steps_used, max_steps=max_steps,
            hint=self._build_hint(),
        )

    def _force_close(self, steps_used: int, max_steps: int) -> TDSObservation:
        return TDSObservation(
            done=True, reward=0.0,
            invoice_text=self._invoice_text(),
            action_result=(
                f"Episode terminated: exceeded {max_steps} steps without submitting. "
                "Score: 0.000."
            ),
            available_actions=[],
            steps_used=steps_used, max_steps=max_steps,
            hint="",
        )

    @property
    def state(self) -> TDSState:
        return self._state
