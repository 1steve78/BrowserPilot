"""Controlled Agent Runner module for BrowserPilot AI.

Implements Member A's Step 3 controlled decision loop connecting the model client,
page observer, and browser executor with strict safeguards:
1. Configurable, bounded action limit (default: 10 actions).
2. Fresh observation captured before every decision step.
3. Element grounding validation against the current observation.
4. Repeated-action / stuck detection on unchanged pages.
5. Independent completion verification (never blindly trusting model 'finish').
6. Controlled error handling for model timeouts, parse failures, observer/executor errors.
7. Structured event telemetry and explicit stop reasons.

Owned by: AI Engineer (Member A)
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from enum import Enum
import hashlib
from typing import Any, Dict, List, Optional
import uuid

from .events import EventType, event_manager as default_event_manager, EventManager
from .executor import PlaywrightExecutor
from .model_client import (
    ModelClientError,
    ModelConnectionError,
    ModelResponseParseError,
    OllamaModelClient,
)
from .observer import PlaywrightObserver
from .safety import SafetyGuard, safety_guard as default_safety_guard
from .schemas import (
    ActionType,
    AgentResponse,
    AgentRunRequest,
    AgentRunState,
    BrowserAction,
    ExecutionResult,
    PageObservation,
    RiskLevel,
    RunStatus,
    SafetyCheckResult,
    StepRecord,
)

DEFAULT_MAX_ACTIONS = 10
MAX_ALLOWED_ACTIONS = 30
STUCK_ACTION_THRESHOLD = 3


class StopReason(str, Enum):
    """Explicit stop reasons for controlled agent loop termination."""
    COMPLETED = "completed"
    UNVERIFIED_COMPLETION = "unverified_completion"
    MAX_STEPS_REACHED = "max_steps_reached"
    STUCK_REPEATED_ACTION = "stuck_repeated_action"
    UNGROUNDED_SELECTOR = "ungrounded_selector"
    SAFETY_BLOCKED = "safety_blocked"
    EXECUTION_FAILED = "execution_failed"
    MODEL_ERROR = "model_error"
    OBSERVER_ERROR = "observer_error"
    STOP_REQUESTED = "stop_requested"


def compute_observation_fingerprint(observation: PageObservation) -> str:
    """Generate a hash representing page state to detect identical consecutive states."""
    data = f"{observation.url}|{observation.title}|{observation.dom_summary}"
    return hashlib.sha256(data.encode("utf-8")).hexdigest()[:16]


def is_action_grounded(action: BrowserAction, observation: PageObservation) -> bool:
    """Verify that an action targeting an element has a selector grounded in the observation."""
    if action.action_type not in (ActionType.CLICK, ActionType.TYPE, ActionType.EXTRACT):
        return True

    if not action.selector:
        return False

    sel = action.selector.strip()
    raw_sel_id = sel.lstrip("#")

    # 1. Check interactive elements list
    for el in observation.interactive_elements:
        if el.selector == sel:
            return True
        if el.id and (el.id == raw_sel_id or el.id == sel):
            return True

    # 2. Check DOM summary presence
    if sel in observation.dom_summary:
        return True
    if raw_sel_id in observation.dom_summary:
        return True

    return False


def verify_task_completion(
    goal: str,
    finish_description: str,
    observation: PageObservation,
) -> bool:
    """Verify whether a model's finish claim is corroborated by page evidence.
    
    Never trusts model claim alone; looks for textual or state indicators
    on the current page matching completion signals or goal keywords.
    """
    text_corpus = (
        f"{observation.title} "
        f"{observation.dom_summary} "
        f"{observation.page_text_snippet or ''} "
        f"{observation.url}"
    ).lower()

    # Universal positive outcome indicators
    completion_indicators = [
        "success", "succeeded", "completed", "complete", "confirmed",
        "confirmation", "thank you", "paid", "submitted", "done",
        "created", "saved", "verified"
    ]

    has_indicator = any(ind in text_corpus for ind in completion_indicators)

    # Goal keyword presence
    goal_words = [w.lower() for w in goal.split() if len(w) > 3]
    goal_words_present = any(w in text_corpus for w in goal_words) if goal_words else True

    return has_indicator or goal_words_present


class ControlledAgentRunner:
    """Controlled agent orchestration runner with safety boundaries and verification."""

    def __init__(
        self,
        model_client: Optional[OllamaModelClient] = None,
        observer: Optional[PlaywrightObserver] = None,
        executor: Optional[PlaywrightExecutor] = None,
        safety: Optional[SafetyGuard] = None,
        events: Optional[EventManager] = None,
        default_max_steps: int = DEFAULT_MAX_ACTIONS,
        stuck_threshold: int = STUCK_ACTION_THRESHOLD,
    ) -> None:
        self.model_client = model_client or OllamaModelClient()
        self.observer = observer or PlaywrightObserver()
        self.executor = executor or PlaywrightExecutor()
        self.safety = safety or default_safety_guard
        self.events = events or default_event_manager
        self.default_max_steps = default_max_steps
        self.stuck_threshold = stuck_threshold
        self._stop_requested = False
        self._current_state: Optional[AgentRunState] = None

    @property
    def current_state(self) -> Optional[AgentRunState]:
        return self._current_state

    def request_stop(self) -> None:
        """Signal the runner to safely halt at the next iteration."""
        self._stop_requested = True

    async def run(
        self,
        request: AgentRunRequest,
        page: Optional[Any] = None,
    ) -> AgentRunState:
        """Run the controlled cognitive loop until verification, limit, or failure."""
        run_id = str(uuid.uuid4())

        requested_steps = request.max_steps if request.max_steps is not None else self.default_max_steps
        max_steps = max(1, min(requested_steps, MAX_ALLOWED_ACTIONS))

        if self._stop_requested:
            state = AgentRunState(
                run_id=run_id,
                goal=request.goal,
                status=RunStatus.STOPPED,
                max_steps=max_steps,
                start_time=datetime.now(timezone.utc),
                end_time=datetime.now(timezone.utc),
                final_output="Run stopped by user request.",
            )
            self._current_state = state
            await self.events.emit(
                EventType.RUN_FINISHED,
                "Run stopped by user request before start.",
                {
                    "status": RunStatus.STOPPED.value,
                    "steps": 0,
                    "stop_reason": StopReason.STOP_REQUESTED.value,
                    "completion_verified": False,
                },
                run_id=run_id,
            )
            return state

        self._stop_requested = False

        state = AgentRunState(
            run_id=run_id,
            goal=request.goal,
            status=RunStatus.INITIALIZING,
            max_steps=max_steps,
            start_time=datetime.now(timezone.utc),
        )
        self._current_state = state

        await self.events.emit(
            EventType.STATUS_CHANGE,
            f"Run initialized: '{request.goal}' with limit {max_steps} actions",
            {"status": RunStatus.INITIALIZING, "run_id": run_id, "max_steps": max_steps},
            run_id=run_id,
        )

        # Track consecutive repeated actions on unchanged pages
        last_action_signature: Optional[str] = None
        last_page_fingerprint: Optional[str] = None
        consecutive_identical_count = 0
        stop_reason: Optional[StopReason] = None
        completion_verified = False

        try:
            # 1. Initialize browser environment
            active_page = page or await self.executor.initialize()
            state.status = RunStatus.RUNNING

            # Initial navigation if requested
            if request.start_url:
                nav_action = BrowserAction(
                    action_type=ActionType.NAVIGATE,
                    url=request.start_url,
                    description=f"Initial navigation to {request.start_url}",
                )
                nav_res = await self.executor.execute(nav_action)
                if not nav_res.success:
                    stop_reason = StopReason.EXECUTION_FAILED
                    state.status = RunStatus.FAILED
                    state.error = f"Failed initial navigation: {nav_res.error}"
                    return state

            # 2. Controlled Execution Loop
            while state.current_step < state.max_steps and not self._stop_requested:
                state.current_step += 1
                step_idx = state.current_step

                # A. Fresh Observation (always re-observe after action)
                try:
                    observation = await self.observer.observe(active_page)
                except Exception as exc:
                    stop_reason = StopReason.OBSERVER_ERROR
                    state.status = RunStatus.FAILED
                    state.error = f"Observation capture failed at step {step_idx}: {exc}"
                    await self.events.emit(
                        EventType.ERROR,
                        state.error,
                        {"step": step_idx, "error": str(exc)},
                        run_id=run_id,
                    )
                    break

                state.current_url = observation.url
                page_fingerprint = compute_observation_fingerprint(observation)

                await self.events.emit(
                    EventType.OBSERVATION_CAPTURED,
                    f"Observed: {observation.title} ({observation.url})",
                    {
                        "step": step_idx,
                        "url": observation.url,
                        "title": observation.title,
                        "fingerprint": page_fingerprint,
                    },
                    run_id=run_id,
                )

                # B. Injection Defense Check
                injections = self.safety.inspect_observation(observation)
                if injections:
                    for inj in injections:
                        await self.events.emit(
                            EventType.SAFETY_ALERT,
                            inj.reason or "Adversarial injection detected in DOM",
                            {"risk": inj.risk_level.value, "pattern": inj.flagged_pattern},
                            run_id=run_id,
                        )

                # C. Query Model Client for Next Action
                await self.events.emit(
                    EventType.AGENT_THINKING,
                    f"Agent thinking for step {step_idx} of {state.max_steps}...",
                    {"step": step_idx},
                    run_id=run_id,
                )

                try:
                    response = await self.model_client.get_next_action(
                        goal=request.goal,
                        observation=observation,
                        step_number=step_idx,
                        max_steps=state.max_steps,
                    )
                except (ModelConnectionError, ModelResponseParseError, ModelClientError, Exception) as exc:
                    stop_reason = StopReason.MODEL_ERROR
                    state.status = RunStatus.FAILED
                    state.error = f"Model client failure at step {step_idx}: {exc}"
                    await self.events.emit(
                        EventType.ERROR,
                        state.error,
                        {"step": step_idx, "error": str(exc)},
                        run_id=run_id,
                    )
                    break

                proposed_action = response.action

                # D. Grounding Validation
                if not is_action_grounded(proposed_action, observation):
                    stop_reason = StopReason.UNGROUNDED_SELECTOR
                    state.status = RunStatus.FAILED
                    state.error = (
                        f"Action selector '{proposed_action.selector}' is not grounded "
                        "in current page observation elements or DOM summary."
                    )
                    await self.events.emit(
                        EventType.ERROR,
                        state.error,
                        {"step": step_idx, "action": proposed_action.model_dump()},
                        run_id=run_id,
                    )
                    break

                # E. Repeated-Action Stuck Detection
                action_signature = f"{proposed_action.action_type.value}|{proposed_action.selector}|{proposed_action.text}|{proposed_action.url}"
                if action_signature == last_action_signature and page_fingerprint == last_page_fingerprint:
                    consecutive_identical_count += 1
                else:
                    consecutive_identical_count = 1

                last_action_signature = action_signature
                last_page_fingerprint = page_fingerprint

                if consecutive_identical_count >= self.stuck_threshold:
                    stop_reason = StopReason.STUCK_REPEATED_ACTION
                    state.status = RunStatus.FAILED
                    state.error = (
                        f"Stuck loop detected: Action '{proposed_action.action_type.value}' "
                        f"repeated {consecutive_identical_count} times without meaningful page change."
                    )
                    await self.events.emit(
                        EventType.ERROR,
                        state.error,
                        {"step": step_idx, "signature": action_signature},
                        run_id=run_id,
                    )
                    break

                # F. Safety Guardrail Evaluation
                safety_check = self.safety.evaluate_action(proposed_action, observation)
                if not safety_check.is_safe:
                    stop_reason = StopReason.SAFETY_BLOCKED
                    state.status = RunStatus.AWAITING_CONFIRMATION
                    state.error = f"Blocked unsafe action: {safety_check.reason}"
                    await self.events.emit(
                        EventType.SAFETY_ALERT,
                        state.error,
                        {"action": proposed_action.model_dump(), "safety": safety_check.model_dump()},
                        run_id=run_id,
                    )
                    break

                await self.events.emit(
                    EventType.ACTION_PROPOSED,
                    f"Proposed {proposed_action.action_type.value}: {proposed_action.description}",
                    {"action": proposed_action.model_dump(), "thought": response.thought.model_dump()},
                    run_id=run_id,
                )

                # G. Completion Verification on FINISH
                if proposed_action.action_type == ActionType.FINISH:
                    is_verified = verify_task_completion(
                        request.goal,
                        proposed_action.description,
                        observation,
                    )
                    if is_verified:
                        stop_reason = StopReason.COMPLETED
                        state.status = RunStatus.COMPLETED
                        state.final_output = f"Verified: {proposed_action.description}"
                        completion_verified = True
                    else:
                        stop_reason = StopReason.UNVERIFIED_COMPLETION
                        state.status = RunStatus.FAILED
                        state.error = (
                            "Model claimed task completion ('finish'), but page state "
                            "does not corroborate completion."
                        )
                    break

                # H. Controlled Failure on FAIL
                if proposed_action.action_type == ActionType.FAIL:
                    stop_reason = StopReason.EXECUTION_FAILED
                    state.status = RunStatus.FAILED
                    state.error = proposed_action.description
                    break

                # I. Execute Action
                exec_result = await self.executor.execute(proposed_action)
                await self.events.emit(
                    EventType.ACTION_EXECUTED,
                    exec_result.message,
                    {"result": exec_result.model_dump(), "step": step_idx},
                    run_id=run_id,
                )

                step_record = StepRecord(
                    step_number=step_idx,
                    observation=observation,
                    thought=response.thought,
                    action=proposed_action,
                    safety_check=safety_check,
                    execution_result=exec_result,
                )
                state.history.append(step_record)

                if not exec_result.success:
                    stop_reason = StopReason.EXECUTION_FAILED
                    state.status = RunStatus.FAILED
                    state.error = f"Execution failed at step {step_idx}: {exec_result.error or exec_result.message}"
                    break

                await asyncio.sleep(0.1)

            # 3. Post-loop resolution
            if self._stop_requested:
                stop_reason = StopReason.STOP_REQUESTED
                state.status = RunStatus.STOPPED
                state.final_output = "Run stopped by user request."
            elif state.status == RunStatus.RUNNING and state.current_step >= state.max_steps:
                stop_reason = StopReason.MAX_STEPS_REACHED
                state.status = RunStatus.COMPLETED
                state.final_output = f"Reached configured limit of {state.max_steps} actions."

        except Exception as exc:
            stop_reason = StopReason.EXECUTION_FAILED
            state.status = RunStatus.FAILED
            state.error = f"Unhandled runner exception: {exc}"
            await self.events.emit(
                EventType.ERROR,
                state.error,
                {"error": str(exc)},
                run_id=run_id,
            )

        finally:
            state.end_time = datetime.now(timezone.utc)
            effective_reason = stop_reason.value if stop_reason else state.status.value
            await self.events.emit(
                EventType.RUN_FINISHED,
                f"Run finished with status {state.status.value} (Reason: {effective_reason})",
                {
                    "status": state.status.value,
                    "steps": state.current_step,
                    "stop_reason": effective_reason,
                    "completion_verified": completion_verified,
                },
                run_id=run_id,
            )

        return state
