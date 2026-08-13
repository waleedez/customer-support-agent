"""Assembles the support ticket graph.

    START -> ticket_analyzer -> {billing_agent, account_agent} (parallel)
          -> policy_agent -> resolution_agent -> reflection_agent
          -> [loop back to resolution_agent, bounded by MAX_REFLECTION_CYCLES]
          -> human_review (only if the resolution requires approval)
          -> [loop back to {billing_agent, account_agent} on "request more
             investigation", bounded by MAX_REINVESTIGATION_CYCLES]
          -> summarizer -> END
"""
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app.graph.edges import route_after_human_review, route_after_reflection
from app.graph.nodes import (
    account_agent_node,
    billing_agent_node,
    human_review_node,
    policy_agent_node,
    reflection_agent_node,
    resolution_agent_node,
    summarizer_node,
    ticket_analyzer_node,
)
from app.state.support_state import SupportState


def _assemble() -> StateGraph:
    graph = StateGraph(SupportState)

    graph.add_node("ticket_analyzer", ticket_analyzer_node)
    graph.add_node("billing_agent", billing_agent_node)
    graph.add_node("account_agent", account_agent_node)
    graph.add_node("policy_agent", policy_agent_node)
    graph.add_node("resolution_agent", resolution_agent_node)
    graph.add_node("reflection_agent", reflection_agent_node)
    graph.add_node("human_review", human_review_node)
    graph.add_node("summarizer", summarizer_node)

    graph.add_edge(START, "ticket_analyzer")

    # Fan out: both specialists investigate independently and in parallel.
    graph.add_edge("ticket_analyzer", "billing_agent")
    graph.add_edge("ticket_analyzer", "account_agent")

    # Fan in: policy_agent runs once both specialists have finished.
    graph.add_edge("billing_agent", "policy_agent")
    graph.add_edge("account_agent", "policy_agent")

    graph.add_edge("policy_agent", "resolution_agent")
    graph.add_edge("resolution_agent", "reflection_agent")

    graph.add_conditional_edges(
        "reflection_agent",
        route_after_reflection,
        ["resolution_agent", "human_review", "summarizer"],
    )
    graph.add_conditional_edges(
        "human_review",
        route_after_human_review,
        ["billing_agent", "account_agent", "summarizer"],
    )

    graph.add_edge("summarizer", END)
    return graph


def build_graph(checkpointer=None):
    """Compile with a checkpointer, for running outside the LangGraph API
    server (tests, app/main.py, evaluation/evaluate.py) - interrupt()/resume
    needs one, and nothing else provides it in those contexts."""
    return _assemble().compile(checkpointer=checkpointer or MemorySaver())


# Exposed for the LangGraph CLI (`langgraph dev`) / LangGraph Studio - see
# langgraph.json's "graphs" entry. Deliberately compiled WITHOUT a
# checkpointer: the LangGraph API server manages persistence itself and
# raises an error at load time if the graph already has a custom one.
graph = _assemble().compile()
