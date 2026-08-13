"""Policy Agent: decides what actions company policy *allows* given the investigations.

This agent does not investigate the customer and does not decide what action to
actually take - that is the Resolution Agent's job. It only answers: given the
investigation results and company policies, what is allowed?
"""
from app.agents._common import run_tool_calling_agent, structured_synthesis
from app.models.llm import get_llm
from app.state.support_state import PolicyDecision
from app.tools.policy_tools import policy_lookup

TOOLS = [policy_lookup]

SYSTEM_PROMPT = """You are the Policy specialist agent for AcmeCloud support. You do \
not investigate the customer - that has already been done. Given the ticket \
category, the billing/account investigation results, and the applicable policies \
(use policy_lookup), decide what actions are ALLOWED and whether human approval is \
required. Cite the specific policy text you relied on."""

SYNTHESIS_PROMPT = """Summarize the policy decision above as structured output."""


def evaluate_policy(
    category: str, billing_investigation: dict, account_investigation: dict, llm=None
) -> dict:
    llm = llm or get_llm()
    user_message = (
        f"Ticket category: {category}\n"
        f"Billing investigation: {billing_investigation}\n"
        f"Account investigation: {account_investigation}\n"
        "What actions does policy allow here, and is human approval required?"
    )
    messages = run_tool_calling_agent(llm, TOOLS, SYSTEM_PROMPT, user_message)
    result = structured_synthesis(llm, PolicyDecision, SYNTHESIS_PROMPT, messages)
    return result.model_dump()
