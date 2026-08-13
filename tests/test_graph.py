"""Graph tests monkeypatch the agent functions imported into app.graph.nodes,
so they exercise the graph's wiring - fan-out/fan-in, the bounded reflection
loop, and human-in-the-loop pause/resume - with no LLM calls at all.
"""
import uuid

from langgraph.types import Command

from app.graph import nodes as nodes_module
from app.graph.support_graph import build_graph


def _new_config():
    return {"configurable": {"thread_id": str(uuid.uuid4())}}


def _base_ticket(ticket_id="1", customer_id="C123", message="test"):
    return {"ticket_id": ticket_id, "customer_id": customer_id, "message": message}


def test_happy_path_no_approval_needed(monkeypatch):
    monkeypatch.setattr(
        nodes_module,
        "analyze_ticket",
        lambda ticket: {
            "category": "billing",
            "urgency": "low",
            "customer_intent": "refund",
            "requires_investigation": True,
        },
    )
    monkeypatch.setattr(
        nodes_module,
        "investigate_billing",
        lambda ticket: {
            "duplicate_payment_detected": True,
            "original_payment_id": "P455",
            "duplicate_payment_id": "P456",
            "amount": 49.99,
            "refund_eligible": True,
            "requires_human_approval": False,
            "summary": "duplicate found",
        },
    )
    monkeypatch.setattr(
        nodes_module,
        "investigate_account",
        lambda ticket: {
            "account_status": "active",
            "subscription": "Pro Monthly",
            "cancellation_requested": False,
            "issues_found": [],
            "summary": "ok",
        },
    )
    monkeypatch.setattr(
        nodes_module,
        "evaluate_policy",
        lambda category, billing, account: {
            "allowed_actions": ["refund"],
            "policy_citations": ["POL-001"],
            "requires_human_approval": False,
            "rationale": "duplicate under threshold",
        },
    )
    monkeypatch.setattr(
        nodes_module,
        "propose_resolution",
        lambda *a, **k: {
            "resolution_type": "refund",
            "action": "refund P456",
            "reason": "duplicate charge",
            "requires_human_approval": False,
            "confidence": 0.9,
            "customer_response_summary": "refunded",
        },
    )
    monkeypatch.setattr(
        nodes_module, "reflect_on_resolution", lambda *a, **k: {"approved": True, "issues": [], "recommendation": []}
    )
    monkeypatch.setattr(
        nodes_module,
        "summarize_response",
        lambda ticket, resolution, human_decision=None: "Your duplicate charge has been refunded.",
    )

    graph = build_graph()
    result = graph.invoke({"ticket": _base_ticket()}, config=_new_config())

    assert "__interrupt__" not in result
    assert result["final_response"] == "Your duplicate charge has been refunded."
    assert result["billing_investigation"]["duplicate_payment_detected"] is True
    assert result["account_investigation"]["account_status"] == "active"


def test_human_review_pauses_and_resumes(monkeypatch):
    monkeypatch.setattr(
        nodes_module,
        "analyze_ticket",
        lambda ticket: {"category": "account", "urgency": "medium", "customer_intent": "cancel", "requires_investigation": True},
    )
    monkeypatch.setattr(
        nodes_module,
        "investigate_billing",
        lambda ticket: {
            "duplicate_payment_detected": False,
            "original_payment_id": None,
            "duplicate_payment_id": None,
            "amount": None,
            "refund_eligible": False,
            "requires_human_approval": False,
            "summary": "n/a",
        },
    )
    monkeypatch.setattr(
        nodes_module,
        "investigate_account",
        lambda ticket: {
            "account_status": "active",
            "subscription": "Basic Monthly",
            "cancellation_requested": True,
            "issues_found": [],
            "summary": "wants to cancel",
        },
    )
    monkeypatch.setattr(
        nodes_module,
        "evaluate_policy",
        lambda category, billing, account: {
            "allowed_actions": ["cancellation"],
            "policy_citations": ["POL-003"],
            "requires_human_approval": True,
            "rationale": "cancellation always needs approval",
        },
    )
    monkeypatch.setattr(
        nodes_module,
        "propose_resolution",
        lambda *a, **k: {
            "resolution_type": "cancellation",
            "action": "cancel account",
            "reason": "customer request",
            "requires_human_approval": True,
            "confidence": 0.95,
            "customer_response_summary": "cancelled",
        },
    )
    monkeypatch.setattr(
        nodes_module, "reflect_on_resolution", lambda *a, **k: {"approved": True, "issues": [], "recommendation": []}
    )
    monkeypatch.setattr(
        nodes_module, "summarize_response", lambda ticket, resolution, human_decision=None: f"decision={human_decision}"
    )

    graph = build_graph()
    config = _new_config()
    result = graph.invoke({"ticket": _base_ticket(customer_id="C127", message="cancel please")}, config=config)

    assert "__interrupt__" in result
    payload = result["__interrupt__"][0].value
    assert payload["options"] == ["approve", "reject", "request_more_investigation"]

    result = graph.invoke(Command(resume="approve"), config=config)
    assert "__interrupt__" not in result
    assert result["final_response"] == "decision=approve"
    assert result["human_decision"] == "approve"


def test_reflection_loop_is_bounded_then_proceeds(monkeypatch):
    resolution_calls = {"count": 0}
    reflection_calls = {"count": 0}

    def fake_propose_resolution(*a, **k):
        resolution_calls["count"] += 1
        return {
            "resolution_type": "information",
            "action": "explain",
            "reason": "n/a",
            "requires_human_approval": False,
            "confidence": 0.5,
            "customer_response_summary": "info",
        }

    def fake_reflect(*a, **k):
        reflection_calls["count"] += 1
        return {"approved": False, "issues": ["not good enough"], "recommendation": ["try again"]}

    monkeypatch.setattr(
        nodes_module,
        "analyze_ticket",
        lambda ticket: {"category": "general", "urgency": "low", "customer_intent": "info", "requires_investigation": False},
    )
    monkeypatch.setattr(
        nodes_module,
        "investigate_billing",
        lambda ticket: {
            "duplicate_payment_detected": False,
            "original_payment_id": None,
            "duplicate_payment_id": None,
            "amount": None,
            "refund_eligible": False,
            "requires_human_approval": False,
            "summary": "n/a",
        },
    )
    monkeypatch.setattr(
        nodes_module,
        "investigate_account",
        lambda ticket: {
            "account_status": "active",
            "subscription": "Basic Monthly",
            "cancellation_requested": False,
            "issues_found": [],
            "summary": "ok",
        },
    )
    monkeypatch.setattr(
        nodes_module,
        "evaluate_policy",
        lambda category, billing, account: {
            "allowed_actions": ["information"],
            "policy_citations": [],
            "requires_human_approval": False,
            "rationale": "n/a",
        },
    )
    monkeypatch.setattr(nodes_module, "propose_resolution", fake_propose_resolution)
    monkeypatch.setattr(nodes_module, "reflect_on_resolution", fake_reflect)
    monkeypatch.setattr(nodes_module, "summarize_response", lambda ticket, resolution, human_decision=None: "final")

    graph = build_graph()
    result = graph.invoke({"ticket": _base_ticket(customer_id="C124", message="info?")}, config=_new_config())

    assert "__interrupt__" not in result
    assert result["final_response"] == "final"
    # initial resolution attempt + one bounded retry, capped by MAX_REFLECTION_CYCLES=2
    assert resolution_calls["count"] == 2
    assert reflection_calls["count"] == 2
    assert result["reflection_count"] == 2


def test_human_requests_more_investigation_reruns_specialists(monkeypatch):
    billing_calls = {"count": 0}

    def fake_billing(ticket):
        billing_calls["count"] += 1
        return {
            "duplicate_payment_detected": False,
            "original_payment_id": None,
            "duplicate_payment_id": None,
            "amount": None,
            "refund_eligible": False,
            "requires_human_approval": False,
            "summary": f"call {billing_calls['count']}",
        }

    monkeypatch.setattr(
        nodes_module,
        "analyze_ticket",
        lambda ticket: {"category": "billing", "urgency": "medium", "customer_intent": "refund", "requires_investigation": True},
    )
    monkeypatch.setattr(nodes_module, "investigate_billing", fake_billing)
    monkeypatch.setattr(
        nodes_module,
        "investigate_account",
        lambda ticket: {
            "account_status": "active",
            "subscription": "Pro Monthly",
            "cancellation_requested": False,
            "issues_found": [],
            "summary": "ok",
        },
    )
    monkeypatch.setattr(
        nodes_module,
        "evaluate_policy",
        lambda category, billing, account: {
            "allowed_actions": ["refund"],
            "policy_citations": [],
            "requires_human_approval": True,
            "rationale": "n/a",
        },
    )
    monkeypatch.setattr(
        nodes_module,
        "propose_resolution",
        lambda *a, **k: {
            "resolution_type": "refund",
            "action": "refund",
            "reason": "n/a",
            "requires_human_approval": True,
            "confidence": 0.6,
            "customer_response_summary": "refund",
        },
    )
    monkeypatch.setattr(
        nodes_module, "reflect_on_resolution", lambda *a, **k: {"approved": True, "issues": [], "recommendation": []}
    )
    monkeypatch.setattr(
        nodes_module, "summarize_response", lambda ticket, resolution, human_decision=None: f"decision={human_decision}"
    )

    graph = build_graph()
    config = _new_config()
    result = graph.invoke({"ticket": _base_ticket(message="refund?")}, config=config)
    assert "__interrupt__" in result

    result = graph.invoke(Command(resume="request_more_investigation"), config=config)
    assert "__interrupt__" in result  # loops back through the whole pipeline to human_review again
    assert billing_calls["count"] == 2

    result = graph.invoke(Command(resume="approve"), config=config)
    assert result["final_response"] == "decision=approve"
