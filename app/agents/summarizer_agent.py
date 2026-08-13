"""Summarizer/Response Agent: produces only the customer-facing text.

This agent never investigates or decides anything - it renders whatever the
Resolution Agent (and, if applicable, the human reviewer) already decided into a
concise, friendly response. Keeping this separate from decision-making means the
reasoning/workflow state never leaks into what the customer sees.
"""
from langchain_core.messages import HumanMessage, SystemMessage

from app.models.llm import get_llm
from app.state.support_state import Ticket

SYSTEM_PROMPT = """You are writing the final customer-facing reply for AcmeCloud \
support. You are NOT investigating or deciding anything - that has already \
happened. Write a concise, friendly, plain-language response that tells the \
customer what was found and what will happen. Do not mention internal agent \
names, policy IDs, or confidence scores. If a human decision is provided, reflect \
it accurately (e.g. a rejected or pending request should not be described as done)."""


def summarize_response(
    ticket: Ticket, resolution: dict, human_decision: str | None = None, llm=None
) -> str:
    llm = llm or get_llm()
    human_decision_note = (
        f"A human reviewer's decision on this resolution was: {human_decision}."
        if human_decision
        else "No human review was required for this resolution."
    )
    user_message = (
        f"Customer's original message: {ticket['message']}\n"
        f"Resolution decided: {resolution}\n"
        f"{human_decision_note}\n"
        "Write the final customer-facing response."
    )
    response = llm.invoke([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_message)])
    return response.content
