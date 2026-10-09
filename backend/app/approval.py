"""Human-In-The-Loop (HITL) safety decision and approval module for BrowserPilot AI.

Defines:
1. Three distinct safety classifications: ALLOW, REQUIRE_APPROVAL, BLOCK.
2. Structured approval request lifecycle: PENDING, APPROVED, REJECTED, TIMED_OUT, CANCELLED, EXPIRED.
3. Thread-safe / async ApprovalManager with anti-stale and anti-duplicate execution guards.
4. Semantic classification separating routine actions from high-consequence operations.

Owned by: AI Engineer (Member A)
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from enum import Enum
import logging
from typing import Any, Dict, List, Optional
import uuid
from pydantic import BaseModel, Field

from .schemas import (
    ActionType,
    BrowserAction,
    PageObservation,
    RiskLevel,
    SafetyCheckResult,
)
from .safety import SafetyGuard, safety_guard

logger = logging.getLogger("browserpilot.approval")


class SafetyDecisionType(str, Enum):
    """Three-tier safety classification for proposed browser actions."""
    ALLOW = "allow"
    REQUIRE_APPROVAL = "require_approval"
    BLOCK = "block"


class ApprovalStatus(str, Enum):
    """Lifecycle status of a human approval request."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class SafetyDecision(BaseModel):
    """Categorized safety decision for a proposed action."""
    decision_type: SafetyDecisionType
    is_safe: bool = True
    reason: str
    risk_level: RiskLevel = RiskLevel.LOW
    flagged_pattern: Optional[str] = None
    requires_human_confirmation: bool = False


class ApprovalRequest(BaseModel):
    """A formal approval request for an action requiring human authorization."""
    approval_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    run_id: str
    step_number: int
    action: BrowserAction
    reason: str
    risk_level: RiskLevel = RiskLevel.MEDIUM
    context: Dict[str, Any] = Field(default_factory=dict)
    status: ApprovalStatus = ApprovalStatus.PENDING
    decision_reason: Optional[str] = None
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    resolved_at: Optional[str] = None
    timeout_seconds: float = 60.0


# Hard-prohibited destructive keywords that are blocked under all circumstances (BLOCK)
PROHIBITED_KEYWORDS = [
    "purge",
    "drop database",
    "drop table",
    "destroy all",
    "wipe disk",
    "format disk",
    "rm -rf",
    "delete all records",
    "clear all database",
    "truncate table",
    "reset database",
]

# Consequential keywords that require explicit human approval (REQUIRE_APPROVAL)
CONSEQUENTIAL_KEYWORDS = [
    # Purchases, financial & transactions
    "buy",
    "purchase",
    "order",
    "checkout",
    "pay",
    "charge",
    "billing",
    "credit card",
    "payment",
    # Communications & external publishing
    "send email",
    "send message",
    "post comment",
    "publish",
    "submit application",
    "broadcast",
    # Sensitive resource removal or account modifications
    "delete",
    "remove",
    "deactivate",
    "cancel subscription",
    "revoke",
    "uninstall",
    # Credentials & security modifications
    "change password",
    "reset password",
    "api key",
    "auth token",
    "security settings",
    # Irreversible workflow steps
    "confirm transfer",
    "finalize payment",
    "execute trade",
    "transfer funds",
]


def classify_action_safety(
    action: BrowserAction,
    observation: Optional[PageObservation] = None,
    guard: Optional[SafetyGuard] = None,
) -> SafetyDecision:
    """Classify a proposed action into ALLOW, REQUIRE_APPROVAL, or BLOCK using semantic analysis.

    Classification Hierarchy:
    1. BLOCK: Prohibited actions violating core safety boundaries (restricted protocols,
       destructive keywords like 'purge' or 'drop table'). Cannot be approved by a prompt.
    2. REQUIRE_APPROVAL: Consequential operations (financial transactions, external messaging,
       irreversible data deletion, credential modification). May proceed only after human consent.
    3. ALLOW: Routine actions (reading text, non-destructive modal opening, search queries,
       navigating ordinary HTTP/HTTPS pages, scrolling). Safe to execute under current policy.
    """
    # 1. Check target URL schemes and input protocols
    restricted_schemes = ("file://", "javascript:", "data:", "vbscript:")
    if action.url and any(action.url.lower().startswith(scheme) for scheme in restricted_schemes):
        return SafetyDecision(
            decision_type=SafetyDecisionType.BLOCK,
            is_safe=False,
            reason=f"Restricted URL scheme detected in target: {action.url}",
            risk_level=RiskLevel.CRITICAL,
            flagged_pattern=action.url,
            requires_human_confirmation=False,
        )

    if action.text and any(action.text.lower().startswith(scheme) for scheme in ("javascript:", "data:", "vbscript:")):
        return SafetyDecision(
            decision_type=SafetyDecisionType.BLOCK,
            is_safe=False,
            reason=f"Restricted protocol detected in input text: {action.text}",
            risk_level=RiskLevel.CRITICAL,
            flagged_pattern=action.text,
            requires_human_confirmation=False,
        )

    # 2. Extract textual corpus for semantic inspection
    corpus_parts = [
        action.selector,
        action.text,
        action.description,
        action.target_element_description,
    ]
    action_text = " ".join(filter(None, corpus_parts)).lower()

    # 3. Check hard-prohibited destructive patterns (BLOCK)
    for kw in PROHIBITED_KEYWORDS:
        if kw in action_text:
            return SafetyDecision(
                decision_type=SafetyDecisionType.BLOCK,
                is_safe=False,
                reason=f"Prohibited destructive operation matching keyword '{kw}'. Execution is barred.",
                risk_level=RiskLevel.CRITICAL,
                flagged_pattern=kw,
                requires_human_confirmation=False,
            )

    # Check observed target element content if available
    if observation and action.selector:
        for elem in observation.interactive_elements:
            if elem.selector == action.selector:
                elem_text = f"{elem.text or ''} {elem.aria_label or ''}".lower()
                for kw in PROHIBITED_KEYWORDS:
                    if kw in elem_text:
                        return SafetyDecision(
                            decision_type=SafetyDecisionType.BLOCK,
                            is_safe=False,
                            reason=f"Target element ({elem.selector}) contains prohibited destructive label '{kw}'.",
                            risk_level=RiskLevel.CRITICAL,
                            flagged_pattern=kw,
                            requires_human_confirmation=False,
                        )

    # 4. Consult SafetyGuard baseline policy if provided
    active_guard = guard or safety_guard
    guard_result: SafetyCheckResult = active_guard.evaluate_action(action, observation)
    if guard_result.requires_human_confirmation and not any(kw in action_text for kw in PROHIBITED_KEYWORDS):
        return SafetyDecision(
            decision_type=SafetyDecisionType.REQUIRE_APPROVAL,
            is_safe=True,
            reason=guard_result.reason or "Safety guardrail requested human confirmation.",
            risk_level=guard_result.risk_level,
            flagged_pattern=guard_result.flagged_pattern,
            requires_human_confirmation=True,
        )
    elif not guard_result.is_safe:
        return SafetyDecision(
            decision_type=SafetyDecisionType.BLOCK,
            is_safe=False,
            reason=guard_result.reason or "Safety guardrail policy violation.",
            risk_level=guard_result.risk_level or RiskLevel.HIGH,
            flagged_pattern=guard_result.flagged_pattern,
            requires_human_confirmation=False,
        )

    # 5. Check consequential keywords for operations requiring human approval (REQUIRE_APPROVAL)
    if action.action_type in (ActionType.CLICK, ActionType.TYPE, ActionType.PRESS_KEY):
        # Exclude routine search inputs from requiring approval
        is_search_context = any(
            search_token in action_text
            for search_token in ("search", "filter", "query", "find")
        ) and not any(
            crit_token in action_text
            for crit_token in ("delete", "pay", "order", "card", "purge", "send")
        )

        if not is_search_context:
            for kw in CONSEQUENTIAL_KEYWORDS:
                if kw in action_text:
                    return SafetyDecision(
                        decision_type=SafetyDecisionType.REQUIRE_APPROVAL,
                        is_safe=True,
                        reason=f"Action involves consequential operation matching '{kw}'. Human confirmation required.",
                        risk_level=RiskLevel.HIGH,
                        flagged_pattern=kw,
                        requires_human_confirmation=True,
                    )

            if observation and action.selector:
                for elem in observation.interactive_elements:
                    if elem.selector == action.selector:
                        elem_text = f"{elem.text or ''} {elem.aria_label or ''}".lower()
                        for kw in CONSEQUENTIAL_KEYWORDS:
                            if kw in elem_text:
                                return SafetyDecision(
                                    decision_type=SafetyDecisionType.REQUIRE_APPROVAL,
                                    is_safe=True,
                                    reason=f"Target element ({elem.selector}) matches consequential label '{kw}'.",
                                    risk_level=RiskLevel.HIGH,
                                    flagged_pattern=kw,
                                    requires_human_confirmation=True,
                                )

    # If SafetyGuard flagged human confirmation, respect it
    if guard_result.requires_human_confirmation:
        return SafetyDecision(
            decision_type=SafetyDecisionType.REQUIRE_APPROVAL,
            is_safe=guard_result.is_safe,
            reason=guard_result.reason or "Safety guardrail requested human confirmation.",
            risk_level=guard_result.risk_level,
            flagged_pattern=guard_result.flagged_pattern,
            requires_human_confirmation=True,
        )

    # 6. Default: Action is safe to proceed under policy (ALLOW)
    return SafetyDecision(
        decision_type=SafetyDecisionType.ALLOW,
        is_safe=True,
        reason=guard_result.reason or "Action passed safety classification.",
        risk_level=RiskLevel.LOW,
        requires_human_confirmation=False,
    )


class ApprovalManager:
    """Manages pending human approval requests, timeouts, and anti-stale execution controls."""

    def __init__(self, default_timeout_seconds: float = 60.0) -> None:
        self.default_timeout_seconds = default_timeout_seconds
        self._requests: Dict[str, ApprovalRequest] = {}
        self._events: Dict[str, asyncio.Event] = {}
        self._lock = asyncio.Lock()

    def create_request(
        self,
        run_id: str,
        step_number: int,
        action: BrowserAction,
        reason: str,
        risk_level: RiskLevel = RiskLevel.MEDIUM,
        context: Optional[Dict[str, Any]] = None,
        timeout_seconds: Optional[float] = None,
    ) -> ApprovalRequest:
        """Create and register a new pending approval request."""
        req_id = str(uuid.uuid4())
        req = ApprovalRequest(
            approval_id=req_id,
            run_id=run_id,
            step_number=step_number,
            action=action,
            reason=reason,
            risk_level=risk_level,
            context=context or {},
            timeout_seconds=timeout_seconds or self.default_timeout_seconds,
        )
        self._requests[req_id] = req
        self._events[req_id] = asyncio.Event()
        return req

    def get_request(self, approval_id: str) -> Optional[ApprovalRequest]:
        """Retrieve an approval request by its unique identifier."""
        return self._requests.get(approval_id)

    def list_pending_requests(self, run_id: Optional[str] = None) -> List[ApprovalRequest]:
        """List all pending approval requests, optionally filtered by run_id."""
        requests = [r for r in self._requests.values() if r.status == ApprovalStatus.PENDING]
        if run_id:
            requests = [r for r in requests if r.run_id == run_id]
        return requests

    def resolve_request(
        self,
        approval_id: str,
        approved: bool,
        reason: Optional[str] = None,
    ) -> bool:
        """Submit a human approval or rejection decision.

        Returns True if the decision was successfully recorded.
        Returns False if the request does not exist, has timed out, was cancelled, or was already resolved.
        """
        req = self._requests.get(approval_id)
        if not req:
            logger.warning("Attempted to resolve non-existent approval request %s", approval_id)
            return False

        if req.status != ApprovalStatus.PENDING:
            logger.warning(
                "Attempted to resolve approval request %s which is in terminal status '%s'",
                approval_id,
                req.status.value,
            )
            return False

        req.status = ApprovalStatus.APPROVED if approved else ApprovalStatus.REJECTED
        req.decision_reason = reason or ("Approved by operator" if approved else "Rejected by operator")
        req.resolved_at = datetime.now(timezone.utc).isoformat()

        # Wake up any waiting async task
        event = self._events.get(approval_id)
        if event:
            event.set()

        return True

    async def wait_for_decision(
        self,
        approval_id: str,
        timeout_seconds: Optional[float] = None,
    ) -> ApprovalStatus:
        """Asynchronously await a human decision for the given approval request.

        Times out safely if no decision is submitted within the timeout window.
        """
        req = self._requests.get(approval_id)
        if not req:
            return ApprovalStatus.REJECTED

        if req.status != ApprovalStatus.PENDING:
            return req.status

        event = self._events.get(approval_id)
        if not event:
            return ApprovalStatus.REJECTED

        effective_timeout = timeout_seconds if timeout_seconds is not None else req.timeout_seconds

        try:
            await asyncio.wait_for(event.wait(), timeout=effective_timeout)
        except (asyncio.TimeoutError, TimeoutError):
            if req.status == ApprovalStatus.PENDING:
                req.status = ApprovalStatus.TIMED_OUT
                req.decision_reason = f"Timed out after {effective_timeout}s waiting for operator authorization"
                req.resolved_at = datetime.now(timezone.utc).isoformat()

        return req.status

    def cancel_run(self, run_id: str) -> List[str]:
        """Cancel all pending approval requests associated with a run."""
        cancelled_ids = []
        for req_id, req in self._requests.items():
            if req.run_id == run_id and req.status == ApprovalStatus.PENDING:
                req.status = ApprovalStatus.CANCELLED
                req.decision_reason = "Run was cancelled by user request"
                req.resolved_at = datetime.now(timezone.utc).isoformat()
                event = self._events.get(req_id)
                if event:
                    event.set()
                cancelled_ids.append(req_id)
        return cancelled_ids

    def is_valid_for_execution(
        self,
        approval_id: str,
        expected_action: BrowserAction,
        is_run_active: bool,
    ) -> bool:
        """Validate that an approval is active, bound to the exact proposed action, and not stale.

        Guarantees:
        1. Run must be actively executing (is_run_active is True).
        2. Approval request must exist and have status APPROVED.
        3. Action parameters must match the approved action exactly.
        """
        if not is_run_active:
            return False

        req = self._requests.get(approval_id)
        if not req or req.status != ApprovalStatus.APPROVED:
            return False

        # Verify action equivalence (exact match)
        if req.action.action_type != expected_action.action_type:
            return False
        if req.action.selector != expected_action.selector:
            return False
        if req.action.text != expected_action.text:
            return False
        if req.action.url != expected_action.url:
            return False
        return True

    def consume_request(self, approval_id: str) -> bool:
        """Mark an approved request as consumed/expired to prevent replay.

        Returns True if the request was successfully transitioned to EXPIRED.
        """
        req = self._requests.get(approval_id)
        if req and req.status == ApprovalStatus.APPROVED:
            req.status = ApprovalStatus.EXPIRED
            return True
        return False
