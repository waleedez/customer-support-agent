"""Conditional routing functions used by the graph's decision points."""
from app import config
from app.state.support_state import SupportState


def route_after_reflection(state: SupportState) -> str:
    """Reflection -> retry resolution (bounded), or proceed to approval routing."""
    reflection = state.get("reflection", {})
    approved = reflection.get("approved", True)
    reflection_count = state.get("reflection_count", 0)

    if not approved and reflection_count < config.MAX_REFLECTION_CYCLES:
        return "resolution_agent"

    resolution = state.get("resolution", {})
    if resolution.get("requires_human_approval"):
        return "human_review"
    return "summarizer"


def route_after_human_review(state: SupportState) -> list[str] | str:
    """Human review -> re-investigate (bounded, fans out to both specialists), or finish."""
    decision = state.get("human_decision")
    reinvestigation_count = state.get("reinvestigation_count", 0)

    if decision == "request_more_investigation" and reinvestigation_count <= config.MAX_REINVESTIGATION_CYCLES:
        return ["billing_agent", "account_agent"]
    return "summarizer"
