"""Billing lookup and refund-eligibility tools, backed by app/data/payments.json."""
import json
from datetime import datetime
from pathlib import Path

from langchain_core.tools import tool

from app import config

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _load_payments() -> list[dict]:
    with open(DATA_DIR / "payments.json", "r", encoding="utf-8") as f:
        return json.load(f)


@tool
def payment_lookup(customer_id: str) -> list[dict]:
    """Look up all recent payments for a customer_id, most recent first."""
    matches = [p for p in _load_payments() if p["customer_id"] == customer_id]
    matches.sort(key=lambda p: p["date"], reverse=True)
    return matches or [{"error": f"No payments found for customer {customer_id}"}]


@tool
def find_duplicate_payments(customer_id: str, max_days_apart: int = 3) -> dict:
    """Check a customer's payment history for duplicate charges (same amount, close dates).

    Returns {"duplicate_found": bool, "original": payment|None, "duplicate": payment|None}.
    """
    payments = [p for p in _load_payments() if p["customer_id"] == customer_id]
    payments.sort(key=lambda p: p["date"])
    for i, first in enumerate(payments):
        for second in payments[i + 1 :]:
            if first["amount"] != second["amount"]:
                continue
            days_apart = abs(
                (datetime.fromisoformat(second["date"]) - datetime.fromisoformat(first["date"])).days
            )
            if days_apart <= max_days_apart:
                return {"duplicate_found": True, "original": first, "duplicate": second}
    return {"duplicate_found": False, "original": None, "duplicate": None}


@tool
def refund_eligibility(payment: dict, policy_text: str) -> dict:
    """Given a payment record and the applicable policy text, determine refund eligibility.

    Returns {"eligible": bool, "reason": str, "requires_human_approval": bool}.
    """
    amount = payment.get("amount", 0)
    status = payment.get("status")
    if status != "succeeded":
        return {
            "eligible": False,
            "reason": f"Payment status is '{status}', not 'succeeded', so it cannot be refunded.",
            "requires_human_approval": False,
        }
    requires_human_approval = amount > config.AUTO_REFUND_THRESHOLD
    reason = (
        f"Payment of ${amount:.2f} succeeded and is refundable under policy: {policy_text}"
    )
    if requires_human_approval:
        reason += f" Amount exceeds the ${config.AUTO_REFUND_THRESHOLD:.0f} auto-refund threshold."
    return {
        "eligible": True,
        "reason": reason,
        "requires_human_approval": requires_human_approval,
    }
