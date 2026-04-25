"""
Procedural Invoice Generator for LegaLoom-Env.

Generates diverse TDS invoices on-the-fly to prevent memorization.
Each invoice is deterministically reproducible from a seed but varies:
- Vendor names and PANs
- Invoice amounts (within section-appropriate ranges)
- Service descriptions (synonym variations)
- PAN operative/inoperative status
- GST bundling patterns
- Mixed goods+services ratios
"""

import hashlib
import random
from typing import Any, Dict, List, Optional, Tuple

from .tds_rules import TDS_SECTIONS, compute_tds, get_rate, threshold_crossed
from .pan_registry import PAN_DB, is_company

VENDOR_FIRST = [
    "Sharma", "Patel", "Kumar", "Singh", "Gupta", "Mehta", "Joshi",
    "Rao", "Reddy", "Nair", "Iyer", "Chopra", "Malhotra", "Verma",
    "Agarwal", "Bansal", "Chadha", "Dhawan", "Eshwar", "Fernandes",
    "Gokhale", "Hegde", "Iqbal", "Jain", "Kapoor", "Luthra",
    "Mishra", "Nanda", "Oak", "Pillai", "Quadir", "Rangan",
    "Saxena", "Thakur", "Uppal", "Vaidya", "Wadhwa",
]

VENDOR_SUFFIXES_COMPANY = [
    "Pvt Ltd", "Technologies Pvt Ltd", "Solutions Pvt Ltd",
    "Services Pvt Ltd", "Infra Pvt Ltd", "Consulting Pvt Ltd",
    "Group Pvt Ltd", "Systems Pvt Ltd", "Partners Pvt Ltd",
]

VENDOR_SUFFIXES_INDIVIDUAL = [
    "& Associates", "& Co", "Consultancy",
    "Tax Services", "Legal Advisory", "Studio",
]

CITIES = [
    "Mumbai", "Delhi", "Bangalore", "Chennai", "Hyderabad",
    "Pune", "Kolkata", "Ahmedabad", "Jaipur", "Lucknow",
    "Chandigarh", "Coimbatore", "Indore", "Nagpur", "Kochi",
]

STATES = {
    "Mumbai": "Maharashtra", "Delhi": "Delhi", "Bangalore": "Karnataka",
    "Chennai": "Tamil Nadu", "Hyderabad": "Telangana", "Pune": "Maharashtra",
    "Kolkata": "West Bengal", "Ahmedabad": "Gujarat", "Jaipur": "Rajasthan",
    "Lucknow": "Uttar Pradesh", "Chandigarh": "Chandigarh",
    "Coimbatore": "Tamil Nadu", "Indore": "Madhya Pradesh",
    "Nagpur": "Maharashtra", "Kochi": "Kerala",
}

GSTIN_PREFIXES = {
    "Mumbai": "27", "Delhi": "07", "Bangalore": "29",
    "Chennai": "33", "Hyderabad": "36", "Pune": "27",
    "Kolkata": "19", "Ahmedabad": "24", "Jaipur": "08",
    "Lucknow": "09", "Chandigarh": "04", "Coimbatore": "33",
    "Indore": "23", "Nagpur": "27", "Kochi": "32",
}

SERVICE_DESCRIPTIONS = {
    "194J_professional": [
        "Legal Consultation & Advisory Services",
        "Corporate Legal Advisory",
        "Contract Drafting & Review",
        "Statutory Audit Services",
        "Chartered Accountant Fees",
        "Medical Consultation",
        "Architectural Design Services",
        "Company Secretary Services",
        "Engineering Consultancy",
        "Interior Design Consultation",
        "Tax Advisory & Compliance Services",
        "Litigation Support & Representation",
        "Due Diligence Services",
        "Regulatory Compliance Advisory",
        "Valuation Advisory Services",
    ],
    "194J_technical": [
        "IT Support & Maintenance Services",
        "Software Development Services",
        "Cloud Hosting & Infrastructure",
        "Data Processing & Analytics",
        "BPO Services",
        "Network Maintenance & Administration",
        "Platform Services & SaaS Subscription",
        "Managed IT Services",
        "System Integration Services",
        "Technical Consultancy",
    ],
    "194C_contractor": [
        "Security Services",
        "Housekeeping & Facility Management",
        "Catering Services",
        "Manpower Supply & Contract Staffing",
        "Event Management Services",
        "Printing & Stationery Supply",
        "Construction Work Contract",
        "Waste Management Services",
        "Pest Control Services",
        "Courier & Logistics Services",
        "Labour Supply Contract",
    ],
    "194I_rent_building": [
        "Office Rent",
        "Commercial Space Rent",
        "Warehouse Rent",
        "Co-working Space Rent",
        "Business Centre Rent",
        "Cold Storage Rent",
        "Godown Rent",
    ],
    "194I_rent_machinery": [
        "Equipment Rental",
        "Machinery Hire",
        "Vehicle Hire (without driver)",
        "Server Rack Rental",
        "Crane Hire",
        "Generator Hire",
        "Plant Hire",
    ],
    "194H_commission": [
        "Sales Commission",
        "Agency Commission",
        "Brokerage Fees",
        "Referral Fee",
        "Distribution Commission",
        "Dealer Margin",
        "Channel Partner Commission",
    ],
    "194T_partner": [
        "Partner Salary",
        "Partner Remuneration",
        "Partner Commission",
        "Partner Bonus",
        "Interest on Partner Capital",
        "Sitting Fees for Partners",
    ],
    "194Q_goods": [
        "Raw Material Procurement",
        "Steel & Metal Purchase",
        "Electronic Components Purchase",
        "Packaging Material Procurement",
        "Industrial Equipment Purchase",
        "Bulk Goods Order",
        "Inventory Purchase",
    ],
}

AMOUNT_RANGES = {
    "194J_professional": (50000, 500000),
    "194J_technical": (50000, 800000),
    "194C_contractor": (15000, 300000),
    "194I_rent_building": (50000, 900000),
    "194I_rent_machinery": (30000, 400000),
    "194H_commission": (15000, 150000),
    "194T_partner": (15000, 200000),
    "194Q_goods": (5000000, 20000000),
}

INOPERATIVE_PAN_PREFIXES = ["AAXCC", "AAXFF"]
OPERATIVE_PAN_PREFIXES_COMPANY = ["AABCT", "AABCI", "AABCD", "AABCN", "AABCW", "AABCS", "AABCP"]
OPERATIVE_PAN_PREFIXES_INDIVIDUAL = ["AADFS", "AAFFK", "ABCPG", "AAFFV", "AADPM", "AAFFT", "AABPI"]
OPERATIVE_PAN_PREFIXES_FIRM = ["AAQCS", "AABCG", "AAQCC", "AABCE", "AAQFP", "AABCM", "AAQFS", "AAQFR", "AAFFM"]


def _generate_pan(rng: random.Random, pan_type: str, inoperative: bool = False) -> str:
    chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    digits = "0123456789"

    if inoperative:
        prefix = rng.choice(INOPERATIVE_PAN_PREFIXES)
    elif pan_type == "company":
        prefix = rng.choice(OPERATIVE_PAN_PREFIXES_COMPANY)
    elif pan_type in ("individual", "firm"):
        prefix = rng.choice(OPERATIVE_PAN_PREFIXES_INDIVIDUAL + OPERATIVE_PAN_PREFIXES_FIRM)
    else:
        prefix = rng.choice(OPERATIVE_PAN_PREFIXES_COMPANY + OPERATIVE_PAN_PREFIXES_INDIVIDUAL)

    if len(prefix) >= 5:
        first5 = prefix[:5]
    else:
        first5 = prefix + "".join(rng.choices(chars, k=5 - len(prefix)))

    mid4 = "".join(rng.choices(digits, k=4))
    last = rng.choice(chars)
    return first5 + mid4 + last


def _generate_vendor_name(rng: random.Random, pan_type: str) -> str:
    first = rng.choice(VENDOR_FIRST)
    if pan_type in ("company", "llp"):
        suffix = rng.choice(VENDOR_SUFFIXES_COMPANY)
        return f"{first} {suffix}"
    return f"{first} {rng.choice(VENDOR_SUFFIXES_INDIVIDUAL)}"


def _generate_gstin(rng: random.Random, city: str, pan: str) -> str:
    prefix = GSTIN_PREFIXES.get(city, "27")
    return f"{prefix}{pan}1Z5"


def _round_indian(amount: float) -> float:
    return round(amount, 2)


def generate_invoice(
    category: str,
    seed: int,
    difficulty: str = "easy",
    force_inoperative_pan: bool = False,
    force_gst_bundled: bool = False,
    force_below_threshold: bool = False,
    cumulative_ytd: Optional[float] = None,
) -> Dict[str, Any]:
    rng = random.Random(seed)

    section_map = {
        "194J_professional": ("194J", "professional"),
        "194J_technical": ("194J", "technical"),
        "194C_contractor": ("194C", "contractor"),
        "194I_rent_building": ("194I", "building"),
        "194I_rent_machinery": ("194I", "machinery"),
        "194H_commission": ("194H", "commission"),
        "194T_partner": ("194T", "partner"),
        "194Q_goods": ("194Q", "goods"),
        "mixed_invoice": ("SPLIT", "mixed"),
        "inoperative_pan": None,
        "gst_bundled_tds_base": None,
        "threshold_boundary": None,
        "below_threshold_new_limits": None,
        "194T_partner_extra": ("194T", "partner"),
    }

    actual_category = category
    if category == "inoperative_pan":
        sub = rng.choice(["194J_professional", "194C_contractor", "194I_rent_building", "194H_commission"])
        section_code, sub_type = section_map[sub]
        force_inoperative_pan = True
        actual_category = sub
    elif category == "gst_bundled_tds_base":
        sub = rng.choice(["194J_professional", "194I_rent_building", "194C_contractor"])
        section_code, sub_type = section_map[sub]
        force_gst_bundled = True
        actual_category = sub
    elif category == "threshold_boundary":
        sub = rng.choice(["194J_professional", "194H_commission", "194I_rent_building"])
        section_code, sub_type = section_map[sub]
        actual_category = sub
    elif category == "below_threshold_new_limits":
        sub = rng.choice(["194H_commission", "194C_contractor", "194J_professional"])
        section_code, sub_type = section_map[sub]
        force_below_threshold = True
        actual_category = sub
    elif category == "mixed_invoice":
        section_code, sub_type = "SPLIT", "mixed"
    else:
        section_code, sub_type = section_map.get(category, ("194J", "professional"))

    pan_type_choices = {
        "194J_professional": ["llp", "individual", "firm"],
        "194J_technical": ["company"],
        "194C_contractor": ["company", "firm"],
        "194I_rent_building": ["company", "individual"],
        "194I_rent_machinery": ["company"],
        "194H_commission": ["firm", "individual"],
        "194T_partner": ["firm"],
        "194Q_goods": ["company"],
    }
    pan_type = rng.choice(pan_type_choices.get(actual_category, ["company"]))

    if force_inoperative_pan:
        pan_status = "inoperative"
    else:
        pan_status = "operative"

    pan = _generate_pan(rng, pan_type, inoperative=(pan_status == "inoperative"))
    vendor_name = _generate_vendor_name(rng, pan_type)
    city = rng.choice(CITIES)
    state = STATES[city]
    gstin = _generate_gstin(rng, city, pan)

    amount_range = AMOUNT_RANGES.get(actual_category, (50000, 500000))
    gst_rate = 0.18

    if force_below_threshold:
        section_data = TDS_SECTIONS.get(section_code, {})
        annual_threshold = section_data.get("threshold_annual", 50000)
        amount = _round_indian(rng.uniform(5000, annual_threshold * 0.8))
        if cumulative_ytd is None:
            cumulative_ytd = 0.0
    elif category == "threshold_boundary":
        section_data = TDS_SECTIONS.get(section_code, {})
        annual_threshold = section_data.get("threshold_annual", 50000)
        if cumulative_ytd is None:
            cumulative_ytd = _round_indian(rng.uniform(annual_threshold * 0.3, annual_threshold * 0.7))
        remaining = max(0, annual_threshold - cumulative_ytd)
        if rng.random() < 0.5:
            amount = _round_indian(rng.uniform(max(5000, remaining - 10000), max(remaining - 1000, 5000)))
        else:
            amount = _round_indian(rng.uniform(remaining + 1000, remaining + 50000))
    else:
        amount = _round_indian(rng.uniform(amount_range[0], amount_range[1]))
        if cumulative_ytd is None:
            cumulative_ytd = _round_indian(rng.uniform(0, amount * 3))

    goods_amount = 0.0
    service_amount = amount
    if category == "mixed_invoice":
        goods_pct = rng.uniform(0.2, 0.6)
        goods_amount = _round_indian(amount * goods_pct)
        service_amount = _round_indian(amount * (1 - goods_pct))

    gst_shown_separately = not force_gst_bundled
    if force_gst_bundled:
        note = rng.choice([
            "inclusive of all taxes",
            "gst included in invoice value",
            "gst bundled",
            "gst not shown separately",
        ])
    else:
        note = ""

    if gst_shown_separately:
        taxable_amount = service_amount
        gst_amount = _round_indian(taxable_amount * gst_rate)
    else:
        taxable_amount = amount
        gst_amount = 0.0

    pan_valid = pan_status == "operative"

    if section_code == "194J":
        if sub_type == "technical" or is_company(pan):
            tds_rate = 2.0
        else:
            tds_rate = 10.0
    elif section_code == "194C":
        tds_rate = 2.0 if pan_type in ("company", "llp") else 1.0
    elif section_code == "194I":
        tds_rate = 2.0 if sub_type == "machinery" else 10.0
    elif section_code == "194H":
        tds_rate = 2.0
    elif section_code == "194T":
        tds_rate = 10.0
    elif section_code == "194Q":
        tds_rate = 0.1
    else:
        tds_rate = 10.0

    if not pan_valid:
        if section_code in ("194Q", "194O"):
            tds_rate = 5.0
        else:
            tds_rate = 20.0

    tds_applicable = threshold_crossed(section_code, taxable_amount, cumulative_ytd)

    if force_below_threshold:
        tds_applicable = False
        tds_rate = 0.0

    if tds_applicable:
        tds_amount = compute_tds(taxable_amount, tds_rate)
    else:
        tds_amount = 0.0
        tds_rate = 0.0

    service_desc = rng.choice(SERVICE_DESCRIPTIONS.get(actual_category, ["Professional Services"]))
    invoice_number = f"INV-{seed:06d}"
    invoice_date = f"15-{rng.choice(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])}-2024"

    address_street = f"{rng.randint(1,999)}, {rng.choice(['MG Road', 'Station Road', 'Park Street', 'Civil Lines', 'Sector {}'.format(rng.randint(1,62)), 'Main Road'])}"
    address_pin = rng.randint(100001, 700001)

    if category == "mixed_invoice":
        desc_line = f"{service_desc} + Hardware/Materials Supply"
    else:
        desc_line = service_desc

    if gst_shown_separately and gst_amount > 0:
        cgst = _round_indian(gst_amount / 2)
        sgst = _round_indian(gst_amount / 2)
        total_invoice = _round_indian(amount + gst_amount)
        gst_block = (
            f"Taxable Amount{taxable_amount:>20,.2f}\n"
            f"CGST @ 9%{cgst:>23,.2f}\n"
            f"SGST @ 9%{sgst:>23,.2f}\n"
            f"{'─' * 50}\n"
            f"Total Invoice Value{total_invoice:>17,.2f}"
        )
    else:
        total_invoice = amount
        if force_gst_bundled:
            gst_block = (
                f"Total Amount (inclusive of all taxes){amount:>10,.2f}\n"
                f"{'─' * 50}\n"
                f"Total Invoice Value{total_invoice:>17,.2f}"
            )
        else:
            gst_block = f"Total Amount{amount:>28,.2f}\n{'─' * 50}\nTotal Invoice Value{total_invoice:>17,.2f}"

    if category == "mixed_invoice":
        items_block = (
            f"{'Description':<35} {'SAC':<8} {'Amount (INR)':>12}\n"
            f"{'─' * 60}\n"
            f"{service_desc:<35} 998212  {service_amount:>12,.2f}\n"
            f"Hardware / Materials Supply          998311  {goods_amount:>12,.2f}\n"
            f"{'─' * 60}"
        )
    else:
        sac = rng.choice(["998212", "998311", "998313", "998519", "997219", "997331"])
        items_block = (
            f"{'Description':<35} {'SAC':<8} {'Amount (INR)':>12}\n"
            f"{'─' * 60}\n"
            f"{desc_line:<35} {sac}  {amount:>12,.2f}\n"
            f"{'─' * 60}"
        )

    invoice_text = (
        f"TAX INVOICE\n\n"
        f"Vendor : {vendor_name}\n"
        f"Address : {address_street}, {city}, {state} — {address_pin}\n"
        f"GSTIN : {gstin}\n"
        f"PAN : {pan}\n\n"
        f"Bill To : Meridian Technologies Pvt Ltd\n"
        f"GSTIN : 27AAACM1234F1Z8\n\n"
        f"Invoice No : {invoice_number}\n"
        f"Invoice Date: {invoice_date}\n"
        f"Place of Supply: {state}\n\n"
        f"{items_block}\n"
        f"{gst_block}\n\n"
        f"Payment Terms : Net 30 days\n"
        f"Bank Details : HDFC Bank, A/c: 50100123456789, IFSC: HDFC0001234\n\n"
        f"Authorised Signatory"
    )

    if note:
        invoice_text += f"\n\nNote: {note}"

    section_gt = section_code
    if category == "mixed_invoice":
        section_gt = "SPLIT"

    ground_truth = {
        "section": section_gt,
        "tds_rate_percent": tds_rate,
        "taxable_amount": taxable_amount,
        "tds_amount_inr": tds_amount,
        "pan_valid": pan_valid,
        "tds_applicable": tds_applicable,
        "cumulative_ytd": cumulative_ytd,
        "goods_amount": goods_amount,
        "note": note,
    }

    task_hint = f"{service_desc}"
    if not pan_valid:
        task_hint += " — INOPERATIVE PAN — 206AA applies"
    if force_gst_bundled:
        task_hint += " — GST bundled — TDS on full amount"
    if force_below_threshold:
        task_hint += " — below threshold — no TDS"

    return {
        "invoice_id": invoice_number,
        "difficulty": difficulty,
        "category": category,
        "task_hint": task_hint,
        "invoice_text": invoice_text,
        "vendor_pan": pan,
        "cumulative_ytd": cumulative_ytd,
        "ground_truth": ground_truth,
    }


def generate_batch(
    category: str,
    count: int,
    base_seed: int = 1000,
    difficulty: str = "easy",
    **kwargs,
) -> List[Dict[str, Any]]:
    invoices = []
    for i in range(count):
        seed = base_seed + i
        inv = generate_invoice(category, seed, difficulty=difficulty, **kwargs)
        invoices.append(inv)
    return invoices


def extend_db_with_generated(existing_db: List[Dict], categories: Optional[List[str]] = None, per_category: int = 50, base_seed: int = 100000) -> List[Dict]:
    if categories is None:
        categories = [
            "194J_professional", "194J_technical", "194C_contractor",
            "194I_rent_building", "194I_rent_machinery", "194H_commission",
            "inoperative_pan", "gst_bundled_tds_base", "threshold_boundary",
            "below_threshold_new_limits", "mixed_invoice", "194T_partner", "194Q_goods",
        ]
    generated = []
    for cat in categories:
        difficulty = "easy"
        if cat in ("inoperative_pan", "gst_bundled_tds_base"):
            difficulty = "hard"
        elif cat in ("threshold_boundary", "mixed_invoice"):
            difficulty = "medium"
        elif cat in ("194T_partner", "194Q_goods"):
            difficulty = "expert"

        batch = generate_batch(cat, per_category, base_seed=base_seed + hash(cat) % 10000, difficulty=difficulty)
        generated.extend(batch)

    return existing_db + generated
