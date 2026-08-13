"""Customer/account lookup tool, backed by app/data/customers.json."""
import json
from pathlib import Path

from langchain_core.tools import tool

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _load_customers() -> list[dict]:
    with open(DATA_DIR / "customers.json", "r", encoding="utf-8") as f:
        return json.load(f)


@tool
def customer_lookup(customer_id: str) -> dict:
    """Look up a customer's name, subscription plan, and account status by customer_id."""
    for customer in _load_customers():
        if customer["customer_id"] == customer_id:
            return customer
    return {"error": f"No customer found with id {customer_id}"}
