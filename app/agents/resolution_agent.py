"""Resolution Agent: decides the concrete action to take, given every prior result."""
from langchain_core.messages import HumanMessage, SystemMessage

from app import config
from app.models.llm import get_llm
from app.state.support_state import Resolution, Ticket

SYSTEM_PROMPT = """You are the Resolution agent for AcmeCloud support. You do not \
investigate anything yourself. Given the ticket, its classification, the billing \
and account investigation results, and the policy decision, decide the single \
concrete action to take. Be conservative: only propose a refund or account change \
that the policy decision explicitly allows. If evidence is missing or ambiguous, \
prefer resolution_type='escalation' over guessing."""


def _enforce_business_rules(resolution: dict, billing_investigation: dict) -> dict:
    """Deterministic safety net: some approval rules are non-negotiable business
    rules, not judgment calls, so we double check them rather than trusting the
    LLM alone (the Reflection Agent also checks this, but belt-and-suspenders)."""
    if resolution["resolution_type"] == "cancellation":
        resolution["requires_human_approval"] = True
    if resolution["resolution_type"] == "refund":
        amount = billing_investigation.get("amount") or 0
        if amount > config.AUTO_REFUND_THRESHOLD:
            resolution["requires_human_approval"] = True
    return resolution


def propose_resolution(
    ticket: Ticket,
    ticket_analysis: dict,
    billing_investigation: dict,
    account_investigation: dict,
    policy_decision: dict,
    llm=None,
) -> dict:
    llm = llm or get_llm()
    structured_llm = llm.with_structured_output(Resolution)
    user_message = (
        f"Original ticket: {ticket['message']}\n"
        f"Ticket classification: {ticket_analysis}\n"
        f"Billing investigation: {billing_investigation}\n"
        f"Account investigation: {account_investigation}\n"
        f"Policy decision: {policy_decision}\n"
        "Decide the resolution."
    )
    result = structured_llm.invoke(
        [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_message)]
    )
    resolution = result.model_dump()
    return _enforce_business_rules(resolution, billing_investigation)
