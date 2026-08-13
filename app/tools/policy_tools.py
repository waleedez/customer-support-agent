"""Support-policy lookup tool, backed by app/data/policies.json."""
import json
from pathlib import Path

from langchain_core.tools import tool

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _load_policies() -> list[dict]:
    with open(DATA_DIR / "policies.json", "r", encoding="utf-8") as f:
        return json.load(f)


@tool
def policy_lookup(category: str) -> list[dict]:
    """Look up applicable support policies for a category (billing, account, or general)."""
    matches = [p for p in _load_policies() if p["category"] == category]
    return matches or [{"error": f"No policies found for category {category}"}]
