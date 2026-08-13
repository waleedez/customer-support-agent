"""Ticket Analyzer: classifies an incoming ticket via structured output."""
from langchain_core.messages import HumanMessage, SystemMessage

from app.models.llm import get_llm
from app.state.support_state import Ticket, TicketAnalysis

SYSTEM_PROMPT = """You are a support ticket triage specialist for AcmeCloud, a SaaS \
subscription company. Classify the incoming ticket. Choose the single best category:
- billing: payments, invoices, charges, refunds
- account: subscription, account status, cancellation, login/profile info
- technical: product bugs, outages, technical troubleshooting
- general: anything else, including plain informational questions

Set requires_investigation to false only for requests that need no lookup at all \
(e.g. a question you can answer from policy alone)."""


def analyze_ticket(ticket: Ticket, llm=None) -> dict:
    llm = llm or get_llm()
    structured_llm = llm.with_structured_output(TicketAnalysis)
    user_message = (
        f"Ticket ID: {ticket['ticket_id']}\n"
        f"Customer: {ticket['customer_id']}\n"
        f"Message: {ticket['message']}"
    )
    result = structured_llm.invoke(
        [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_message)]
    )
    return result.model_dump()
