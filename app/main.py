"""CLI entry point: run a support ticket through the graph, prompting for human
approval when the graph pauses on a sensitive action."""
import sys
import uuid

from langgraph.types import Command

from app.graph.support_graph import build_graph

SAMPLE_TICKET = {
    "ticket_id": "1001",
    "customer_id": "C123",
    "message": "I was charged twice for my subscription. Please refund the duplicate payment.",
}


def _prompt_for_decision(interrupt_payload: dict) -> str:
    print("\n--- Human approval requested ---")
    print(f"Ticket: {interrupt_payload['ticket']['message']}")
    print(f"Proposed resolution: {interrupt_payload['resolution']}")
    options = interrupt_payload["options"]
    while True:
        choice = input(f"Decision ({'/'.join(options)}): ").strip().lower()
        if choice in options:
            return choice
        print(f"Please enter one of: {', '.join(options)}")


def run_ticket(ticket: dict) -> str:
    graph = build_graph()
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    result = graph.invoke({"ticket": ticket}, config=config)
    while "__interrupt__" in result:
        interrupt_payload = result["__interrupt__"][0].value
        decision = _prompt_for_decision(interrupt_payload)
        result = graph.invoke(Command(resume=decision), config=config)

    return result["final_response"]


def main() -> None:
    message = " ".join(sys.argv[1:]) or SAMPLE_TICKET["message"]
    ticket = {**SAMPLE_TICKET, "message": message}
    final_response = run_ticket(ticket)
    print("\n--- Final response ---")
    print(final_response)


if __name__ == "__main__":
    main()
