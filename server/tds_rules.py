"""
TDS Rules Engine for LegaLoom-Env.
Section rates, thresholds, and deduction logic for Indian TDS compliance.
"""

TDS_SECTIONS = {
    "194C": {
        "description": "Payments to Contractors",
        "nature": (
            "Work contracts: labour supply, catering, housekeeping, "
            "event management, transportation, security, printing, advertising"
        ),
        "rate_default": 1.0,
        "rate_company": 2.0,
        "rate_no_pan": 20.0,
        "threshold_single": 30000,
        "threshold_annual": 100000,
    },
    "194J": {
        "description": "Fees for Professional or Technical Services",
        "nature": (
            "Professional: legal, audit, medical, engineering, consulting, architecture. "
            "Technical (sub-rate 2%): software dev, IT support, cloud services, data processing"
        ),
        "rate_default": 10.0,
        "rate_technical": 2.0,
        "rate_no_pan": 20.0,
        "threshold_single": 30000,
        "threshold_annual": 30000,
    },
    "194I": {
        "description": "Rent",
        "nature": (
            "Rent for land, building, furniture, fittings (10%). "
            "Plant, machinery, equipment hire (2%)"
        ),
        "rate_default": 10.0,
        "rate_machinery": 2.0,
        "rate_no_pan": 20.0,
        "threshold_single": 0,
        "threshold_annual": 240000,
    },
    "194H": {
        "description": "Commission or Brokerage",
        "nature": "Sales commission, agency commission, brokerage, referral fees, dealer margins",
        "rate_default": 5.0,
        "rate_no_pan": 20.0,
        "threshold_single": 15000,
        "threshold_annual": 15000,
    },
    "194Q": {
        "description": "Purchase of Goods",
        "nature": (
            "Goods purchase by buyer with turnover > 10 crore. "
            "Applies only when cumulative purchase from one vendor exceeds 50 lakh in a year."
        ),
        "rate_default": 0.1,
        "rate_no_pan": 5.0,
        "threshold_single": 0,
        "threshold_annual": 5000000,
    },
}

PAN_INVALID_RATE = 20.0

NO_TDS_CATEGORIES = [
    "goods", "products", "merchandise", "hardware",
    "raw material", "stationery", "office supplies", "inventory",
]


def get_rate(section_code: str, vendor_is_company: bool = True,
             pan_valid: bool = True, is_machinery: bool = False,
             is_technical: bool = False) -> float:
    if not pan_valid:
        return PAN_INVALID_RATE
    section = TDS_SECTIONS.get(section_code)
    if not section:
        return 0.0
    if section_code == "194C":
        return section["rate_company"] if vendor_is_company else section["rate_default"]
    if section_code == "194I" and is_machinery:
        return section["rate_machinery"]
    if section_code == "194J" and is_technical:
        return section["rate_technical"]
    return section["rate_default"]


def threshold_crossed(section_code: str, invoice_amount: float,
                      cumulative_ytd: float = 0.0) -> bool:
    section = TDS_SECTIONS.get(section_code)
    if not section:
        return False
    single_limit  = section.get("threshold_single", 0)
    annual_limit  = section.get("threshold_annual", 0)
    running_total = cumulative_ytd + invoice_amount
    if single_limit == 0 and annual_limit == 0:
        return True
    if single_limit > 0 and invoice_amount >= single_limit:
        return True
    if annual_limit > 0 and running_total >= annual_limit:
        return True
    return False


def compute_tds(taxable_amount: float, rate_percent: float) -> float:
    return round(taxable_amount * rate_percent / 100, 2)


def section_summary(section_code: str) -> str:
    s = TDS_SECTIONS.get(section_code)
    if not s:
        return f"Unknown section: {section_code}"
    return f"{section_code} — {s['description']} (default rate {s['rate_default']}%)"