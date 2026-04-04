"""
PAN Registry for LegaLoom-Env.
Mock database of vendor PAN numbers and their operative status.
In production this would call the Income Tax e-filing API.
"""

from typing import Optional

PAN_DB = {
    "ABCDE1234F": {
        "name": "Sharma & Associates LLP",
        "status": "operative",
        "pan_type": "company",
    },
    "BCDFE5678G": {
        "name": "TechServ Solutions Pvt Ltd",
        "status": "operative",
        "pan_type": "company",
    },
    "CDEFG9012H": {
        "name": "Rajesh Kumar",
        "status": "operative",
        "pan_type": "individual",
    },
    "PQRST1122A": {
        "name": "QuickCater Services",
        "status": "operative",
        "pan_type": "company",
    },
    "ZZZZZ9999Z": {
        "name": "CloudMatrix Infrastructure Pvt Ltd",
        "status": "inoperative",
        "pan_type": "company",
    },
}


def lookup_pan(pan: str) -> Optional[dict]:
    return PAN_DB.get(pan.upper().strip())


def is_pan_valid(pan: str) -> bool:
    record = lookup_pan(pan)
    if record is None:
        return False
    return record["status"] == "operative"


def is_company(pan: str) -> bool:
    record = lookup_pan(pan)
    if record is None:
        return True
    return record["pan_type"] == "company"


def pan_status_message(pan: str) -> str:
    record = lookup_pan(pan)
    if record is None:
        return f"PAN {pan} not found in registry. TDS rate: 20%."
    if record["status"] == "inoperative":
        return (
            f"PAN {pan} ({record['name']}) is INOPERATIVE "
            f"(not linked to Aadhaar). TDS rate: 20% regardless of section."
        )
    return (
        f"PAN {pan} ({record['name']}) is operative. "
        f"Vendor type: {record['pan_type']}."
    )