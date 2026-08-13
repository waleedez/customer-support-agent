"""Node functions: each wraps one agent and translates its result into a state update."""
from langgraph.types import interrupt

from app.agents.account_agent import investigate_account
from app.agents.billing_agent import investigate_billing
from app.agents.policy_agent import evaluate_policy
from app.agents.reflection_agent import reflect_on_resolution
from app.agents.resolution_agent import propose_resolution
from app.agents.summarizer_agent import summarize_response
from app.agents.ticket_analyzer import analyze_ticket
from app.state.support_state import SupportState


def ticket_analyzer_node(state: SupportState) -> dict:
    analysis = analyze_ticket(state["ticket"])
    return {"ticket_analysis": analysis}


def billing_agent_node(state: SupportState) -> dict:
    investigation = investigate_billing(state["ticket"])
    return {"billing_investigation": investigation}


def account_agent_node(state: SupportState) -> dict:
    investigation = investigate_account(state["ticket"])
    return {"account_investigation": investigation}


def policy_agent_node(state: SupportState) -> dict:
    decision = evaluate_policy(
        state["ticket_analysis"]["category"],
        state["billing_investigation"],
        state["account_investigation"],
    )
    return {"policy_decision": decision}


def resolution_agent_node(state: SupportState) -> dict:
    resolution = propose_resolution(
        state["ticket"],
        state["ticket_analysis"],
        state["billing_investigation"],
        state["account_investigation"],
        state["policy_decision"],
    )
    return {"resolution": resolution}


def reflection_agent_node(state: SupportState) -> dict:
    reflection = reflect_on_resolution(
        state["ticket"],
        state["ticket_analysis"],
        state["billing_investigation"],
        state["account_investigation"],
        state["policy_decision"],
        state["resolution"],
    )
    return {
        "reflection": reflection,
        "reflection_count": state.get("reflection_count", 0) + 1,
    }


def human_review_node(state: SupportState) -> dict:
    """Pause the graph and wait for a human decision on the proposed resolution.

    interrupt() suspends execution here (persisted via the graph's checkpointer)
    until the caller resumes with Command(resume=<decision>). See app/main.py for
    the resume loop.
    """
    decision = interrupt(
        {
            "type": "approval_request",
            "ticket": state["ticket"],
            "resolution": state["resolution"],
            "options": ["approve", "reject", "request_more_investigation"],
        }
    )
    update = {"human_decision": decision}
    if decision == "request_more_investigation":
        update["reinvestigation_count"] = state.get("reinvestigation_count", 0) + 1
    return update


def summarizer_node(state: SupportState) -> dict:
    final_response = summarize_response(
        state["ticket"], state["resolution"], state.get("human_decision")
    )
    return {"final_response": final_response}
