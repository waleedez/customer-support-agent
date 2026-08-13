"""Runs evaluation/dataset.json end-to-end through the graph and scores results.

This exercises the real graph and therefore requires a configured LLM provider
(see README "Setup"). If LANGCHAIN_TRACING_V2=true is set (see app/config.py),
each run is also traced to LangSmith automatically - no extra code needed here.
"""
import json
import uuid
from pathlib import Path

from langgraph.types import Command

from app.graph.support_graph import build_graph

DATASET_PATH = Path(__file__).resolve().parent / "dataset.json"


def _run_ticket(graph, ticket: dict) -> dict:
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    result = graph.invoke({"ticket": ticket}, config=config)
    while "__interrupt__" in result:
        # Auto-approve so the batch can run unattended. Policy compliance is
        # scored by checking whether approval *was requested*, not by
        # emulating what a human reviewer would decide.
        result = graph.invoke(Command(resume="approve"), config=config)
    return result


def evaluate() -> None:
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    graph = build_graph()

    correct = {"category": 0, "resolution_type": 0, "requires_human_approval": 0}
    rows = []

    for case in dataset:
        ticket = {
            "ticket_id": case["ticket_id"],
            "customer_id": case["customer_id"],
            "message": case["message"],
        }
        result = _run_ticket(graph, ticket)

        predicted = {
            "category": result["ticket_analysis"]["category"],
            "resolution_type": result["resolution"]["resolution_type"],
            "requires_human_approval": result["resolution"]["requires_human_approval"],
        }
        expected = {
            "category": case["expected_category"],
            "resolution_type": case["expected_resolution_type"],
            "requires_human_approval": case["expected_requires_human_approval"],
        }
        matches = {key: predicted[key] == expected[key] for key in predicted}
        for key, is_match in matches.items():
            correct[key] += int(is_match)

        rows.append(
            {
                "ticket_id": case["ticket_id"],
                "predicted": predicted,
                "expected": expected,
                "matches": matches,
                "final_response": result["final_response"],
            }
        )

    total = len(dataset)
    print(f"Evaluated {total} tickets\n")
    for row in rows:
        status = "OK" if all(row["matches"].values()) else "MISMATCH"
        print(f"[{status}] {row['ticket_id']}")
        for key in ("category", "resolution_type", "requires_human_approval"):
            print(f"  {key}: predicted={row['predicted'][key]!r} expected={row['expected'][key]!r}")
        print(f"  response: {row['final_response']!r}\n")

    print("--- Summary ---")
    for metric, count in correct.items():
        print(f"{metric}: {count}/{total} ({count / total:.0%})")


if __name__ == "__main__":
    evaluate()
