"""
Task definitions for LegaLoom-Env.
Three TDS compliance tasks of increasing difficulty.
"""

TASK_1 = {
    "task_id": "task_easy",
    "difficulty": "easy",
    "description": "Single invoice, clear professional service under 194J",
    "max_steps": 6,
    "invoice": """
INVOICE

Vendor  : Sharma & Associates LLP
PAN     : ABCDE1234F
GSTIN   : 29ABCDE1234F1ZK
Date    : 12-Nov-2024
Invoice : INV-2024-0891

Services Rendered:
  Legal Consultation & Advisory Services    INR 1,50,000

  (Retainer for November 2024 — corporate legal advisory)

Total Amount Due                            INR 1,50,000

Payment Terms: Net 30 days
Bank: HDFC Bank, A/c: 12345678901234, IFSC: HDFC0001234
""".strip(),
    "vendor_pan": "ABCDE1234F",
    "cumulative_ytd": 0.0,
    "ground_truth": {
        "section": "194J",
        "tds_rate_percent": 10.0,
        "taxable_amount": 150000.0,
        "tds_amount_inr": 15000.0,
        "pan_valid": True,
        "threshold_applicable": True,
    },
    "reward_breakpoints": {
        "pan_checked": 0.10,
        "section_correct": 0.30,
        "rate_correct": 0.20,
        "amount_exact": 0.40,
    },
}

TASK_2 = {
    "task_id": "task_medium",
    "difficulty": "medium",
    "description": "Mixed invoice with goods and services; threshold boundary check",
    "max_steps": 8,
    "invoice": """
INVOICE

Vendor  : TechServ Solutions Pvt Ltd
PAN     : BCDFE5678G
GSTIN   : 27BCDFE5678G1ZP
Date    : 05-Dec-2024
Invoice : INV-2024-1147

Line Items:
  1. Network Switches (Hardware — 12 units)   INR 95,000
  2. Annual IT Support & Maintenance Contract INR 85,000

     (Covers helpdesk, on-site support, system monitoring)

Subtotal                                      INR 1,80,000
GST @18% on Services only                    INR  15,300
Total Amount Due                              INR 1,95,300

Note: Hardware supplied under separate delivery challan DC-2024-0445.
Payment Terms: Net 45 days
""".strip(),
    "vendor_pan": "BCDFE5678G",
    "cumulative_ytd": 0.0,
    "ground_truth": {
        "section": "194J",
        "tds_rate_percent": 10.0,
        "taxable_amount": 85000.0,
        "tds_amount_inr": 8500.0,
        "pan_valid": True,
        "threshold_applicable": True,
        "goods_amount": 95000.0,
        "goods_tds": 0.0,
    },
    "reward_breakpoints": {
        "pan_checked": 0.10,
        "goods_excluded": 0.20,
        "section_correct": 0.20,
        "rate_correct": 0.10,
        "amount_exact": 0.40,
    },
}

TASK_3 = {
    "task_id": "task_hard",
    "difficulty": "hard",
    "description": "Inoperative PAN detected; 20% fallback rate applies",
    "max_steps": 8,
    "invoice": """
INVOICE

Vendor  : CloudMatrix Infrastructure Pvt Ltd
PAN     : ZZZZZ9999Z
GSTIN   : 07ZZZZZ9999Z1ZQ
Date    : 18-Jan-2025
Invoice : INV-2025-0042

Services Rendered:
  Cloud Infrastructure & Platform Services   INR 2,40,000

  (Includes: managed cloud hosting, auto-scaling compute,
   object storage, CDN access, and 24x7 NOC support
   for Q4 FY2024-25)

Total Amount Due                             INR 2,40,000

Payment Terms: Net 30 days
Bank: ICICI Bank, A/c: 987654321098, IFSC: ICIC0001234

** Vendor requests TDS certificate after deduction **
""".strip(),
    "vendor_pan": "ZZZZZ9999Z",
    "cumulative_ytd": 0.0,
    "ground_truth": {
        "section": "194J",
        "tds_rate_percent": 20.0,
        "taxable_amount": 240000.0,
        "tds_amount_inr": 48000.0,
        "pan_valid": False,
        "threshold_applicable": True,
        "pan_status": "inoperative",
    },
    "reward_breakpoints": {
        "pan_checked": 0.20,
        "pan_inoperative_identified": 0.30,
        "rate_correct": 0.10,
        "amount_exact": 0.40,
    },
}

TASKS = {
    "task_easy":   TASK_1,
    "task_medium": TASK_2,
    "task_hard":   TASK_3,
}

TASK_ORDER = ["task_easy", "task_medium", "task_hard"]


def get_task(task_id: str) -> dict:
    if task_id not in TASKS:
        raise KeyError(f"Unknown task_id: {task_id!r}. Valid: {list(TASKS.keys())}")
    return TASKS[task_id]


def all_task_ids() -> list:
    return TASK_ORDER.copy()