"""Tool tests run against the local JSON fixtures - no LLM, no network."""
from app.tools.billing_tools import find_duplicate_payments, payment_lookup, refund_eligibility
from app.tools.customer_tools import customer_lookup
from app.tools.policy_tools import policy_lookup


def test_customer_lookup_found():
    result = customer_lookup.invoke({"customer_id": "C123"})
    assert result["name"] == "Jordan Lee"
    assert result["account_status"] == "active"


def test_customer_lookup_not_found():
    result = customer_lookup.invoke({"customer_id": "does-not-exist"})
    assert "error" in result


def test_payment_lookup_returns_most_recent_first():
    payments = payment_lookup.invoke({"customer_id": "C126"})
    assert len(payments) == 2
    assert payments[0]["date"] >= payments[1]["date"]


def test_payment_lookup_no_payments():
    result = payment_lookup.invoke({"customer_id": "does-not-exist"})
    assert "error" in result[0]


def test_find_duplicate_payments_detects_duplicate():
    result = find_duplicate_payments.invoke({"customer_id": "C123"})
    assert result["duplicate_found"] is True
    assert result["original"]["payment_id"] == "P455"
    assert result["duplicate"]["payment_id"] == "P456"


def test_find_duplicate_payments_no_duplicate():
    result = find_duplicate_payments.invoke({"customer_id": "C124"})
    assert result["duplicate_found"] is False


def test_refund_eligibility_under_threshold_no_approval():
    payment = {"payment_id": "P456", "amount": 49.99, "status": "succeeded"}
    result = refund_eligibility.invoke({"payment": payment, "policy_text": "duplicate charges refundable"})
    assert result["eligible"] is True
    assert result["requires_human_approval"] is False


def test_refund_eligibility_over_threshold_requires_approval():
    payment = {"payment_id": "P902", "amount": 499.0, "status": "succeeded"}
    result = refund_eligibility.invoke({"payment": payment, "policy_text": "duplicate charges refundable"})
    assert result["eligible"] is True
    assert result["requires_human_approval"] is True


def test_refund_eligibility_failed_payment_not_eligible():
    payment = {"payment_id": "P900", "amount": 599.0, "status": "failed"}
    result = refund_eligibility.invoke({"payment": payment, "policy_text": "n/a"})
    assert result["eligible"] is False


def test_policy_lookup_returns_category_matches():
    policies = policy_lookup.invoke({"category": "billing"})
    assert all(p["category"] == "billing" for p in policies)
    assert len(policies) == 2


def test_policy_lookup_unknown_category():
    result = policy_lookup.invoke({"category": "unknown"})
    assert "error" in result[0]
