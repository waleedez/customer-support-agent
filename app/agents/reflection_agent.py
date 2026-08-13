"""Reflection Agent: generate -> critique -> revise. Reviews the proposed resolution."""
from langchain_core.messages import HumanMessage, SystemMessage

from app.models.llm import get_llm
from app.state.support_state import ReflectionResult, Ticket

SYSTEM_PROMPT = """You are the Reflection agent for AcmeCloud support. Critically \
review the proposed resolution before it is acted on. Check:
1. Did the agents correctly understand the ticket?
2. Is the proposed action supported by the investigation evidence?
3. Does it comply with the policy decision?
4. Is there missing information that should have been gathered first?
5. Is requires_human_approval set correctly given the policy decision?
6. Is the confidence score reasonable given the evidence?

Set approved=false if any check fails, and explain exactly what is wrong in \
`issues`, with a concrete fix in `recommendation`."""


def reflect_on_resolution(
    ticket: Ticket,
    ticket_analysis: dict,
    billing_investigation: dict,
    account_investigation: dict,
    policy_decision: dict,
    resolution: dict,
    llm=None,
) -> dict:
    llm = llm or get_llm()
    structured_llm = llm.with_structured_output(ReflectionResult)
    user_message = (
        f"Original ticket: {ticket['message']}\n"
        f"Ticket classification: {ticket_analysis}\n"
        f"Billing investigation: {billing_investigation}\n"
        f"Account investigation: {account_investigation}\n"
        f"Policy decision: {policy_decision}\n"
        f"Proposed resolution: {resolution}\n"
        "Review this resolution."
    )
    result = structured_llm.invoke(
        [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_message)]
    )
    return result.model_dump()
