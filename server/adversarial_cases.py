"""
Adversarial test set for LegaLoom-Env.

20 hand-curated TDS scenarios designed to expose specific LLM failure modes:
  - Inoperative-PAN under-correction (model knows about 206AA but applies it
    only when PAN is exactly the test format)
  - Mixed goods+services where goods come first (positional bias)
  - Threshold boundary off-by-one (cumulative YTD = 49,999, current = 2)
  - New FY 2025-26 sections (194T partner, 194Q goods 0.1%) absent from
    pretraining
  - 194I machinery vs building (rate differs: 2% vs 10%)
  - Operative PAN with section-rate ambiguity (194J professional 10% vs
    technical 2%)

Each case is a hand-written invoice with a known ground truth. The set is
held out — never appears in training data, never in DIFFICULTY_POOLS, only
used as a benchmark.

Use:
    from server.adversarial_cases import ADVERSARIAL_CASES, score_adversarial
    cases = ADVERSARIAL_CASES  # list of 20 dicts
    result = score_adversarial(submission_dict, case)
"""
from __future__ import annotations

from typing import Dict, Any, List


ADVERSARIAL_CASES: List[Dict[str, Any]] = [
    # === Inoperative PAN under low base rate (3 cases) ===
    {
        "id": "adv_001_inop_194c",
        "category": "inoperative_pan_low_base_rate",
        "failure_mode": "Inoperative PAN applied to 2% base rate (194C contractor) - 10x rate jump",
        "invoice_text": (
            "TAX INVOICE\n"
            "From: Bharat Security & Manpower LLP, Mumbai\n"
            "PAN: BHRTS9876L\n"
            "GSTIN: 27BHRTS9876L1Z5\n"
            "To: Acme Industries Pvt Ltd\n"
            "Service: Manpower supply for Q3 - security guards\n"
            "Amount: 85000 (excluding GST)\n"
            "GST 18%: 15300\n"
            "Total: 100300"
        ),
        "vendor_pan": "BHRTS9876L",
        "cumulative_ytd": 0,
        "ground_truth": {
            "tds_amount_inr": 17000.0,
            "section": "194C",
            "rate_percent": 20.0,
            "pan_status": "inoperative",
        },
    },
    {
        "id": "adv_002_inop_194h",
        "category": "inoperative_pan_low_base_rate",
        "failure_mode": "Inoperative PAN on 2% commission section",
        "invoice_text": (
            "Vendor: Sunrise Distribution Services\n"
            "PAN: SNRSE5432K\n"
            "Service: Sales commission - Q2 FY 25-26\n"
            "Commission Amount: 240000\n"
            "Note: Vendor PAN flagged as inoperative on filing portal."
        ),
        "vendor_pan": "SNRSE5432K",
        "cumulative_ytd": 0,
        "ground_truth": {
            "tds_amount_inr": 48000.0,
            "section": "194H",
            "rate_percent": 20.0,
            "pan_status": "inoperative",
        },
    },
    {
        "id": "adv_003_inop_194q",
        "category": "inoperative_pan_low_base_rate",
        "failure_mode": "Inoperative PAN on 0.1% goods purchase (194Q) - 200x rate jump",
        "invoice_text": (
            "Goods Invoice\n"
            "Vendor: Steel Suppliers Mumbai Pvt Ltd\n"
            "PAN: STLSP1234M (status: inoperative)\n"
            "Goods: TMT bars - 50 MT\n"
            "Value: 6500000\n"
            "Buyer turnover prev FY: 15 Cr (above 10 Cr threshold)\n"
            "YTD purchases from this vendor: 8000000 (above 50L threshold)"
        ),
        "vendor_pan": "STLSP1234M",
        "cumulative_ytd": 8000000,
        "ground_truth": {
            "tds_amount_inr": 1300000.0,
            "section": "194Q",
            "rate_percent": 20.0,
            "pan_status": "inoperative",
        },
    },

    # === Mixed goods+services positional bias (2 cases) ===
    {
        "id": "adv_004_mixed_goods_first",
        "category": "mixed_goods_services_positional",
        "failure_mode": "Goods listed first (200K), services last (50K) - TDS only on services portion",
        "invoice_text": (
            "Invoice\n"
            "From: TechServices India Pvt Ltd\n"
            "PAN: TCSRV4321P (Company)\n"
            "Item 1 (Goods): Networking hardware - 200000\n"
            "Item 2 (Goods): Server rack - 50000\n"
            "Item 3 (Service): Installation and configuration - 50000\n"
            "Subtotal: 300000\n"
            "GST 18% on services only: 9000\n"
            "Total: 309000"
        ),
        "vendor_pan": "TCSRV4321P",
        "cumulative_ytd": 0,
        "ground_truth": {
            "tds_amount_inr": 1000.0,
            "section": "194J",
            "rate_percent": 2.0,
            "tds_base": 50000,
        },
    },
    {
        "id": "adv_005_mixed_60_40",
        "category": "mixed_goods_services_positional",
        "failure_mode": "Services 60% of value but listed second - must split correctly",
        "invoice_text": (
            "From: Global Office Solutions\n"
            "PAN: GLBOF8765N (LLP)\n"
            "Furniture (Goods): 40000\n"
            "Office layout consulting (Service): 60000\n"
            "Total: 100000 (excl. GST)"
        ),
        "vendor_pan": "GLBOF8765N",
        "cumulative_ytd": 0,
        "ground_truth": {
            "tds_amount_inr": 6000.0,
            "section": "194J",
            "rate_percent": 10.0,
            "tds_base": 60000,
        },
    },

    # === Threshold boundary edge cases (2 cases) ===
    {
        "id": "adv_006_threshold_just_above",
        "category": "threshold_boundary",
        "failure_mode": "YTD 49999 + current 2 = 50001 (crosses 194J threshold by 1 rupee)",
        "invoice_text": (
            "Vendor: Legal Advisor - Mr. R Verma (Individual)\n"
            "PAN: VERMA1234A\n"
            "Service: Contract review - draft revision\n"
            "Bill: 2 (token charge)\n"
            "Note: YTD payments to this vendor: 49999"
        ),
        "vendor_pan": "VERMA1234A",
        "cumulative_ytd": 49999,
        "ground_truth": {
            "tds_amount_inr": 5000.10,
            "section": "194J",
            "rate_percent": 10.0,
        },
    },
    {
        "id": "adv_007_threshold_just_below",
        "category": "threshold_boundary",
        "failure_mode": "YTD 49000 + current 500 = 49500 (still below 50K threshold)",
        "invoice_text": (
            "Vendor: Mr. K Singh (Individual professional)\n"
            "PAN: KSNGH7654B\n"
            "Service: Tax filing assistance\n"
            "Bill: 500\n"
            "YTD: 49000 from same vendor"
        ),
        "vendor_pan": "KSNGH7654B",
        "cumulative_ytd": 49000,
        "ground_truth": {
            "tds_amount_inr": 0.0,
            "section": "no_tds",
            "rate_percent": 0.0,
            "no_tds": True,
            "reason": "below_threshold",
        },
    },

    # === New FY 2025-26 sections (3 cases) ===
    {
        "id": "adv_008_194t_partner_drawings",
        "category": "fy2526_new_sections",
        "failure_mode": "194T partner drawings - added Apr 2025, not in pretraining",
        "invoice_text": (
            "Drawings statement - FY 2025-26\n"
            "Partner: Mr. A Khanna (designated partner)\n"
            "PAN: KHNNA9876C\n"
            "Firm: Khanna & Associates LLP\n"
            "Drawings (salary equivalent): 300000 for the quarter\n"
            "Section: 194T (FY 2025-26 onwards)"
        ),
        "vendor_pan": "KHNNA9876C",
        "cumulative_ytd": 0,
        "ground_truth": {
            "tds_amount_inr": 30000.0,
            "section": "194T",
            "rate_percent": 10.0,
        },
    },
    {
        "id": "adv_009_194t_below_threshold",
        "category": "fy2526_new_sections",
        "failure_mode": "194T threshold is 20K - drawings of 15K are below",
        "invoice_text": (
            "Partner drawings: Mr. P Mehta\n"
            "PAN: MHTAP3456D\n"
            "Amount: 15000\n"
            "YTD partner drawings: 15000"
        ),
        "vendor_pan": "MHTAP3456D",
        "cumulative_ytd": 15000,
        "ground_truth": {
            "tds_amount_inr": 0.0,
            "section": "no_tds",
            "rate_percent": 0.0,
            "no_tds": True,
            "reason": "below_194t_threshold",
        },
    },
    {
        "id": "adv_010_194q_goods_above_threshold",
        "category": "fy2526_new_sections",
        "failure_mode": "194Q at 0.1% - easy to confuse with TCS section 206C(1H)",
        "invoice_text": (
            "Goods purchase invoice\n"
            "Vendor: Cement Corp Ltd\n"
            "PAN: CMNTC4321X (operative)\n"
            "Goods: Portland cement - 200 MT @ 35000/MT\n"
            "Value: 7000000\n"
            "Buyer turnover (prev FY): 50 Cr\n"
            "YTD purchases from this vendor: 20000000"
        ),
        "vendor_pan": "CMNTC4321X",
        "cumulative_ytd": 20000000,
        "ground_truth": {
            "tds_amount_inr": 7000.0,
            "section": "194Q",
            "rate_percent": 0.1,
        },
    },

    # === 194I machinery vs building (2 cases) ===
    {
        "id": "adv_011_194i_machinery",
        "category": "section_subtype_ambiguity",
        "failure_mode": "194I machinery rent at 2% - model often applies 10% (building rate)",
        "invoice_text": (
            "Hire charges invoice\n"
            "Vendor: Heavy Equipment Rentals Pvt Ltd\n"
            "PAN: HVYEQ7890Y\n"
            "Description: Crane hire - monthly rent for construction site\n"
            "Amount: 70000 per month\n"
            "Annual contract value: 840000 (above 6L threshold)"
        ),
        "vendor_pan": "HVYEQ7890Y",
        "cumulative_ytd": 0,
        "ground_truth": {
            "tds_amount_inr": 1400.0,
            "section": "194I",
            "rate_percent": 2.0,
            "subtype": "machinery",
        },
    },
    {
        "id": "adv_012_194i_building",
        "category": "section_subtype_ambiguity",
        "failure_mode": "194I building rent at 10% - model may apply 2% (confusing with machinery)",
        "invoice_text": (
            "Rental invoice\n"
            "Vendor: Property Holdings LLP\n"
            "PAN: PRPHL5432Z\n"
            "Description: Office space rent - 2nd floor commercial building, 1500 sqft\n"
            "Monthly rent: 80000\n"
            "Annual: 960000"
        ),
        "vendor_pan": "PRPHL5432Z",
        "cumulative_ytd": 0,
        "ground_truth": {
            "tds_amount_inr": 8000.0,
            "section": "194I",
            "rate_percent": 10.0,
            "subtype": "building",
        },
    },

    # === Entity type rate change (2 cases) ===
    {
        "id": "adv_013_194j_pvt_ltd_technical",
        "category": "entity_type_rate_change",
        "failure_mode": "Pvt Ltd vendor -> 194J technical 2%, not professional 10%",
        "invoice_text": (
            "Vendor: CloudOps Tech Pvt Ltd\n"
            "PAN: CLDPS8888T (Company)\n"
            "Service: AWS infrastructure management - monthly\n"
            "Amount: 350000"
        ),
        "vendor_pan": "CLDPS8888T",
        "cumulative_ytd": 0,
        "ground_truth": {
            "tds_amount_inr": 7000.0,
            "section": "194J",
            "rate_percent": 2.0,
            "subtype": "technical",
        },
    },
    {
        "id": "adv_014_194j_individual_professional",
        "category": "entity_type_rate_change",
        "failure_mode": "Individual CA -> 194J professional 10%, not technical 2%",
        "invoice_text": (
            "Vendor: CA Mr. S Iyer (Individual practitioner)\n"
            "PAN: SIYER1111E\n"
            "Service: Statutory audit - FY 24-25\n"
            "Audit fees: 180000"
        ),
        "vendor_pan": "SIYER1111E",
        "cumulative_ytd": 0,
        "ground_truth": {
            "tds_amount_inr": 18000.0,
            "section": "194J",
            "rate_percent": 10.0,
            "subtype": "professional",
        },
    },

    # === GST base handling (2 cases) ===
    {
        "id": "adv_015_gst_separate",
        "category": "gst_base_handling",
        "failure_mode": "GST shown separately - TDS base is pre-GST amount only",
        "invoice_text": (
            "Tax Invoice\n"
            "Vendor: Audit Partners LLP\n"
            "PAN: ADTPR3333F\n"
            "Service fees: 100000\n"
            "CGST 9%: 9000\n"
            "SGST 9%: 9000\n"
            "Total payable: 118000"
        ),
        "vendor_pan": "ADTPR3333F",
        "cumulative_ytd": 0,
        "ground_truth": {
            "tds_amount_inr": 10000.0,
            "section": "194J",
            "rate_percent": 10.0,
            "tds_base": 100000,
        },
    },
    {
        "id": "adv_016_gst_inclusive",
        "category": "gst_base_handling",
        "failure_mode": "GST-inclusive single total - must back-calculate pre-GST base",
        "invoice_text": (
            "Vendor: Service Pro India\n"
            "PAN: SRVPR5555G (LLP)\n"
            "Single line: Professional services rendered - 118000 inclusive of 18% GST\n"
            "Note: TDS to be calculated on pre-GST base."
        ),
        "vendor_pan": "SRVPR5555G",
        "cumulative_ytd": 0,
        "ground_truth": {
            "tds_amount_inr": 10000.0,
            "section": "194J",
            "rate_percent": 10.0,
            "tds_base": 100000,
        },
    },

    # === Conflicting evidence / red herrings (2 cases) ===
    {
        "id": "adv_017_misleading_note",
        "category": "conflicting_evidence",
        "failure_mode": "Invoice has misleading 'no TDS applicable' note but section 194J applies",
        "invoice_text": (
            "Vendor: Legal Counsel Office (Individual)\n"
            "PAN: LGCNS7777H\n"
            "Service: Litigation support\n"
            "Amount: 120000\n"
            "Vendor note: 'TDS not applicable - please pay full amount'\n"
            "(YTD: 0)"
        ),
        "vendor_pan": "LGCNS7777H",
        "cumulative_ytd": 0,
        "ground_truth": {
            "tds_amount_inr": 12000.0,
            "section": "194J",
            "rate_percent": 10.0,
            "note": "Vendor's claim is wrong - statutory TDS cannot be waived",
        },
    },
    {
        "id": "adv_018_old_threshold_reference",
        "category": "conflicting_evidence",
        "failure_mode": "Vendor cites pre-FY25-26 threshold (15K for 194H) - current is 20K",
        "invoice_text": (
            "Vendor: Brokerage Services Co\n"
            "PAN: BRKSV2222J (firm)\n"
            "Commission: 17000\n"
            "Vendor note: 'Above 15000 threshold, deduct 5% TDS'\n"
            "(Note: 15K threshold was old; FY 25-26 threshold is 20K)\n"
            "YTD: 0"
        ),
        "vendor_pan": "BRKSV2222J",
        "cumulative_ytd": 0,
        "ground_truth": {
            "tds_amount_inr": 0.0,
            "section": "no_tds",
            "rate_percent": 0.0,
            "no_tds": True,
            "reason": "below_fy2526_194h_threshold",
        },
    },

    # === Compound traps (2 cases) ===
    {
        "id": "adv_019_compound_inop_mixed",
        "category": "compound_traps",
        "failure_mode": "Inoperative PAN AND mixed goods+services AND threshold boundary",
        "invoice_text": (
            "Vendor: TechSupplies & Services LLP\n"
            "PAN: TSPSV9999K (status: inoperative)\n"
            "Item 1 (Goods): Cables - 30000\n"
            "Item 2 (Service): Installation labor - 25000\n"
            "Subtotal: 55000\n"
            "YTD payments: 40000 (will cross 50K threshold this bill)"
        ),
        "vendor_pan": "TSPSV9999K",
        "cumulative_ytd": 40000,
        "ground_truth": {
            "tds_amount_inr": 5000.0,
            "section": "194J",
            "rate_percent": 20.0,
            "tds_base": 25000,
            "pan_status": "inoperative",
        },
    },
    {
        "id": "adv_020_compound_194t_threshold",
        "category": "compound_traps",
        "failure_mode": "194T (new section) AND threshold boundary AND drawings vs profit-share confusion",
        "invoice_text": (
            "Partner statement - Q2 FY 25-26\n"
            "Partner: Ms. R Sharma (designated partner)\n"
            "PAN: RSHMA8888L\n"
            "Drawings (salary equivalent): 19500\n"
            "Profit share (separate from drawings): 50000\n"
            "YTD drawings: 19500\n"
            "Note: 194T applies to drawings only; profit share is exempt."
        ),
        "vendor_pan": "RSHMA8888L",
        "cumulative_ytd": 19500,
        "ground_truth": {
            "tds_amount_inr": 0.0,
            "section": "no_tds",
            "rate_percent": 0.0,
            "no_tds": True,
            "reason": "drawings_below_194t_threshold_profit_share_exempt",
        },
    },
]


def score_adversarial(submission: Dict[str, Any], case: Dict[str, Any]) -> Dict[str, Any]:
    """
    Grade a submission against an adversarial case. Returns a dict with
    `score` in [0.0, 1.0], `breakdown` (component scores), and `feedback`.

    Composite scoring:
      - section_match (40%): exact section code match
      - rate_match    (30%): rate within 0.1pp tolerance
      - amount_match  (30%): amount within 1.0 INR OR within 0.5% relative
    """
    gt = case["ground_truth"]
    feedback: List[str] = []

    # Section match
    sub_section = str(submission.get("section", "")).strip().upper().replace(" ", "")
    gt_section = str(gt.get("section", "")).strip().upper().replace(" ", "")
    if sub_section == gt_section:
        section_score = 1.0
    elif sub_section.startswith(gt_section[:3]) and gt_section.startswith("194"):
        section_score = 0.5
        feedback.append(f"Section partial match: {sub_section} vs {gt_section}")
    else:
        section_score = 0.0
        feedback.append(f"Section wrong: {sub_section} vs expected {gt_section}")

    # Rate match
    try:
        sub_rate = float(submission.get("rate_percent", -1))
        gt_rate = float(gt.get("rate_percent", -1))
        if abs(sub_rate - gt_rate) < 0.1:
            rate_score = 1.0
        elif abs(sub_rate - gt_rate) < 1.0:
            rate_score = 0.5
            feedback.append(f"Rate close but not exact: {sub_rate}% vs {gt_rate}%")
        else:
            rate_score = 0.0
            feedback.append(f"Rate wrong: {sub_rate}% vs expected {gt_rate}%")
    except (TypeError, ValueError):
        rate_score = 0.0
        feedback.append("Rate could not be parsed")

    # Amount match
    try:
        sub_amount = float(submission.get("tds_amount_inr", -1))
        gt_amount = float(gt.get("tds_amount_inr", 0))
        if gt.get("no_tds"):
            no_tds_claimed = (
                str(submission.get("no_tds", "")).lower() == "true"
                or sub_amount == 0
                or sub_section == "NO_TDS"
            )
            amount_score = 1.0 if no_tds_claimed else 0.0
            if not no_tds_claimed:
                feedback.append("Failed to claim no_tds for below-threshold case")
        elif abs(sub_amount - gt_amount) <= 1.0:
            amount_score = 1.0
        elif gt_amount > 0 and abs(sub_amount - gt_amount) / gt_amount <= 0.005:
            amount_score = 1.0
        elif gt_amount > 0 and abs(sub_amount - gt_amount) / gt_amount <= 0.05:
            amount_score = 0.5
            feedback.append(f"Amount within 5%: {sub_amount:.2f} vs {gt_amount:.2f}")
        else:
            amount_score = 0.0
            feedback.append(f"Amount wrong: {sub_amount:.2f} vs expected {gt_amount:.2f}")
    except (TypeError, ValueError):
        amount_score = 0.0
        feedback.append("Amount could not be parsed")

    composite = 0.4 * section_score + 0.3 * rate_score + 0.3 * amount_score
    return {
        "score": round(composite, 4),
        "breakdown": {
            "section_score": section_score,
            "rate_score": rate_score,
            "amount_score": amount_score,
        },
        "feedback": feedback,
        "case_id": case["id"],
        "category": case["category"],
        "failure_mode": case["failure_mode"],
    }


def get_categories() -> List[str]:
    """List all adversarial-case category labels."""
    return sorted({c["category"] for c in ADVERSARIAL_CASES})
