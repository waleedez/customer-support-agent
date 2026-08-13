"""Agent tests use FakeLLM (see conftest.py) - no real LLM call, no API key needed."""
from langchain_core.messages import AIMessage

from app.agents.account_agent import investigate_account
from app.agents.billing_agent import investigate_billing
from app.agents.policy_agent import evaluate_policy
from app.agents.reflection_agent import reflect_on_resolution
from app.agents.resolution_agent import propose_resolution
from app.agents.summarizer_agent import summarize_response
from app.agents.ticket_analyzer import analyze_ticket
from app.state.support_state import (
    AccountInvestigation,
    BillingInvestigation,
    PolicyDecision,
    ReflectionResult,
    Resolution,
    TicketAnalysis,
)

TICKET = {
    "ticket_id": "1001",
    "customer_id": "C123",
    "message": "I was charged twice for my subscription. Please refund the duplicate payment.",
}


def test_analyze_ticket_returns_structured_classification(fake_llm_factory):
    expected = TicketAnalysis(
        category="billing",
        urgency="medium",
        customer_intent="Refund a duplicate charge",
        requires_investigation=True,
    )
    llm = fake_llm_factory(structured_responses={TicketAnalysis: expected})

    result = analyze_ticket(TICKET, llm=llm)

    assert result["category"] == "billing"
    assert result["requires_investigation"] is True


def test_investigate_billing_runs_tools_then_synthesizes(fake_llm_factory):
    tool_ai_messages = [
        AIMessage(
            content="",
            tool_calls=[
                {"name": "find_duplicate_payments", "args": {"customer_id": "C123"}, "id": "call_1"}
            ],
        ),
        AIMessage(content="Found a duplicate payment.", tool_calls=[]),
    ]
    expected = BillingInvestigation(
        duplicate_payment_detected=True,
        original_payment_id="P455",
        duplicate_payment_id="P456",
        amount=49.99,
        refund_eligible=True,
        requires_human_approval=False,
        summary="Duplicate $49.99 charge found.",
    )
    llm = fake_llm_factory(
        tool_ai_messages=tool_ai_messages, structured_responses={BillingInvestigation: expected}
    )

    result = investigate_billing(TICKET, llm=llm)

    assert result["duplicate_payment_detected"] is True
    assert result["duplicate_payment_id"] == "P456"


def test_investigate_account_runs_tools_then_synthesizes(fake_llm_factory):
    tool_ai_messages = [
        AIMessage(
            content="",
            tool_calls=[{"name": "customer_lookup", "args": {"customer_id": "C123"}, "id": "call_1"}],
        ),
        AIMessage(content="Account looks fine.", tool_calls=[]),
    ]
    expected = AccountInvestigation(
        account_status="active",
        subscription="Pro Monthly",
        cancellation_requested=False,
        issues_found=[],
        summary="Account in good standing.",
    )
    llm = fake_llm_factory(
        tool_ai_messages=tool_ai_messages, structured_responses={AccountInvestigation: expected}
    )

    result = investigate_account(TICKET, llm=llm)

    assert result["account_status"] == "active"
    assert result["cancellation_requested"] is False


def test_evaluate_policy_no_tool_calls_needed(fake_llm_factory):
    tool_ai_messages = [AIMessage(content="Policy applied.", tool_calls=[])]
    expected = PolicyDecision(
        allowed_actions=["refund"],
        policy_citations=["POL-001"],
        requires_human_approval=False,
        rationale="Duplicate payment under threshold.",
    )
    llm = fake_llm_factory(
        tool_ai_messages=tool_ai_messages, structured_responses={PolicyDecision: expected}
    )

    result = evaluate_policy(
        "billing",
        {"duplicate_payment_detected": True, "amount": 49.99},
        {"account_status": "active"},
        llm=llm,
    )

    assert result["allowed_actions"] == ["refund"]
    assert result["requires_human_approval"] is False


def test_propose_resolution_enforces_refund_threshold(fake_llm_factory):
    llm_response = Resolution(
        resolution_type="refund",
        action="Refund payment P902",
        reason="Duplicate charge",
        requires_human_approval=False,  # model under-called this; the agent must correct it
        confidence=0.9,
        customer_response_summary="Refund issued",
    )
    llm = fake_llm_factory(structured_responses={Resolution: llm_response})

    result = propose_resolution(
        TICKET,
        {"category": "billing"},
        {"amount": 499.0},
        {"account_status": "active"},
        {"allowed_actions": ["refund"], "requires_human_approval": False},
        llm=llm,
    )

    assert result["requires_human_approval"] is True


def test_propose_resolution_enforces_cancellation_approval(fake_llm_factory):
    llm_response = Resolution(
        resolution_type="cancellation",
        action="Cancel subscription",
        reason="Customer request",
        requires_human_approval=False,
        confidence=0.9,
        customer_response_summary="Cancelled",
    )
    llm = fake_llm_factory(structured_responses={Resolution: llm_response})

    result = propose_resolution(
        TICKET,
        {"category": "account"},
        {},
        {"cancellation_requested": True},
        {"allowed_actions": ["cancellation"], "requires_human_approval": True},
        llm=llm,
    )

    assert result["requires_human_approval"] is True


def test_reflect_on_resolution_returns_structured_review(fake_llm_factory):
    expected = ReflectionResult(
        approved=False, issues=["No amount verified"], recommendation=["Re-check payment"]
    )
    llm = fake_llm_factory(structured_responses={ReflectionResult: expected})

    result = reflect_on_resolution(
        TICKET, {"category": "billing"}, {}, {}, {}, {"resolution_type": "refund"}, llm=llm
    )

    assert result["approved"] is False
    assert result["issues"] == ["No amount verified"]


def test_summarize_response_produces_customer_text(fake_llm_factory):
    llm = fake_llm_factory(plain_response="Your duplicate charge has been refunded.")

    text = summarize_response(TICKET, {"resolution_type": "refund"}, human_decision=None, llm=llm)

    assert text == "Your duplicate charge has been refunded."
