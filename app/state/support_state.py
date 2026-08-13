"""Shared graph state and the structured-output schemas agents populate it with."""
from typing import List, Literal, Optional

from pydantic import BaseModel, Field
from typing_extensions import TypedDict


class Ticket(TypedDict):
    ticket_id: str
    customer_id: str
    message: str


class TicketAnalysis(BaseModel):
    """Structured output of the Ticket Analyzer."""

    category: Literal["billing", "account", "technical", "general"]
    urgency: Literal["low", "medium", "high"]
    customer_intent: str = Field(description="One sentence summary of what the customer wants.")
    requires_investigation: bool


class BillingInvestigation(BaseModel):
    """Structured output of the Billing Agent."""

    duplicate_payment_detected: bool
    original_payment_id: Optional[str] = None
    duplicate_payment_id: Optional[str] = None
    amount: Optional[float] = None
    refund_eligible: bool
    requires_human_approval: bool
    summary: str


class AccountInvestigation(BaseModel):
    """Structured output of the Account Agent."""

    account_status: str
    subscription: Optional[str] = None
    cancellation_requested: bool
    issues_found: List[str] = Field(default_factory=list)
    summary: str


class PolicyDecision(BaseModel):
    """Structured output of the Policy Agent. Decides what's *allowed*, not what to do."""

    allowed_actions: List[str]
    policy_citations: List[str] = Field(default_factory=list)
    requires_human_approval: bool
    rationale: str


class Resolution(BaseModel):
    """Structured output of the Resolution Agent. Decides what action to take."""

    resolution_type: Literal[
        "refund", "account_update", "cancellation", "information", "escalation", "no_action"
    ]
    action: str
    reason: str
    requires_human_approval: bool
    confidence: float = Field(ge=0.0, le=1.0)
    customer_response_summary: str


class ReflectionResult(BaseModel):
    """Structured output of the Reflection Agent's generate-critique-revise check."""

    approved: bool
    issues: List[str] = Field(default_factory=list)
    recommendation: List[str] = Field(default_factory=list)


class SupportState(TypedDict, total=False):
    """The shared state threaded through every node in the graph."""

    ticket: Ticket
    ticket_analysis: dict
    billing_investigation: dict
    account_investigation: dict
    policy_decision: dict
    resolution: dict
    reflection: dict
    reflection_count: int
    reinvestigation_count: int
    human_decision: str
    final_response: str
