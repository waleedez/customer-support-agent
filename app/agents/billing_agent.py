"""Billing Agent: investigates payments, invoices, and refund eligibility."""
from app.agents._common import run_tool_calling_agent, structured_synthesis
from app.models.llm import get_llm
from app.state.support_state import BillingInvestigation, Ticket
from app.tools.billing_tools import find_duplicate_payments, payment_lookup, refund_eligibility
from app.tools.customer_tools import customer_lookup
from app.tools.policy_tools import policy_lookup

TOOLS = [customer_lookup, payment_lookup, find_duplicate_payments, refund_eligibility, policy_lookup]

SYSTEM_PROMPT = """You are the Billing specialist agent for AcmeCloud support. \
Investigate the customer's payment history using the available tools: look up the \
customer, look up their payments, check for duplicate charges, look up billing \
policy, and check refund eligibility where relevant. Only investigate - do not \
promise a resolution to the customer."""

SYNTHESIS_PROMPT = """Summarize the billing investigation above as structured output. \
If no duplicate payment or billing issue exists, say so plainly (duplicate_payment_detected=false, \
refund_eligible=false)."""


def investigate_billing(ticket: Ticket, llm=None) -> dict:
    llm = llm or get_llm()
    user_message = (
        f"Customer: {ticket['customer_id']}\n"
        f"Ticket message: {ticket['message']}\n"
        "Investigate this customer's billing situation."
    )
    messages = run_tool_calling_agent(llm, TOOLS, SYSTEM_PROMPT, user_message)
    result = structured_synthesis(llm, BillingInvestigation, SYNTHESIS_PROMPT, messages)
    return result.model_dump()
