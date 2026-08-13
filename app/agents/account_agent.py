"""Account Agent: investigates subscription, account status, and cancellation requests."""
from app.agents._common import run_tool_calling_agent, structured_synthesis
from app.models.llm import get_llm
from app.state.support_state import AccountInvestigation, Ticket
from app.tools.customer_tools import customer_lookup

TOOLS = [customer_lookup]

SYSTEM_PROMPT = """You are the Account specialist agent for AcmeCloud support. \
Investigate the customer's subscription and account status using the customer_lookup \
tool. Note whether the ticket is asking for a cancellation, and flag any account-level \
issues (e.g. inactive subscription, past_due status). Only investigate - do not \
promise a resolution to the customer."""

SYNTHESIS_PROMPT = """Summarize the account investigation above as structured output."""


def investigate_account(ticket: Ticket, llm=None) -> dict:
    llm = llm or get_llm()
    user_message = (
        f"Customer: {ticket['customer_id']}\n"
        f"Ticket message: {ticket['message']}\n"
        "Investigate this customer's account situation."
    )
    messages = run_tool_calling_agent(llm, TOOLS, SYSTEM_PROMPT, user_message)
    result = structured_synthesis(llm, AccountInvestigation, SYNTHESIS_PROMPT, messages)
    return result.model_dump()
