"""Agent execution loop coordinator.

Orchestrates the cognitive cycle:
1. Observe page state via Observer
2. Check observation for adversarial injections
3. Request next action from ModelClient
4. Evaluate action safety via SafetyGuard
5. Execute action via Executor
6. Broadcast telemetry via EventManager
7. Evaluate termination conditions

Jointly owned by: AI Engineer & Browser Automation Engineer
"""

import asyncio
from datetime import datetime, timezone
import uuid
from typing import Optional

from .events import EventType, event_manager
from .executor import PlaywrightExecutor
from .model_client import OllamaModelClient
from .observer import PlaywrightObserver
from .safety import SafetyGuard, safety_guard
from .schemas import (
    ActionType,
    AgentRunRequest,
    AgentRunState,
    BrowserAction,
    RunStatus,
    StepRecord,
)


class AgentLoop:
    """Orchestrates an autonomous browser agent session."""

    def __init__(
        self,
        model_client: Optional[OllamaModelClient] = None,
        observer: Optional[PlaywrightObserver] = None,
        executor: Optional[PlaywrightExecutor] = None,
        safety: Optional[SafetyGuard] = None,
    ) -> None:
        self.model_client = model_client or OllamaModelClient()
        self.observer = observer or PlaywrightObserver()
        self.executor = executor or PlaywrightExecutor()
        self.safety = safety or safety_guard
        self._is_running = False
        self._stop_requested = False
        self._current_state: Optional[AgentRunState] = None

    @property
    def current_state(self) -> Optional[AgentRunState]:
        return self._current_state

    def request_stop(self) -> None:
        """Flag the running loop to safely stop at next iteration."""
        self._stop_requested = True

    async def run(self, request: AgentRunRequest) -> AgentRunState:
        """Execute the agent loop for the provided goal."""
        run_id = str(uuid.uuid4())
        self._is_running = True
        self._stop_requested = False

        state = AgentRunState(
            run_id=run_id,
            goal=request.goal,
            status=RunStatus.INITIALIZING,
            max_steps=request.max_steps or 15,
            start_time=datetime.now(timezone.utc),
        )
        self._current_state = state

        await event_manager.emit(
            EventType.STATUS_CHANGE,
            f"Run initialized: '{request.goal}'",
            {"status": RunStatus.INITIALIZING, "run_id": run_id},
            run_id=run_id,
        )

        try:
            # 1. Initialize browser
            page = await self.executor.initialize()
            state.status = RunStatus.RUNNING

            # Initial navigation if requested
            if request.start_url:
                await self.executor.execute(
                    BrowserAction(
                        action_type=ActionType.NAVIGATE,
                        url=request.start_url,
                        description=f"Initial navigation to {request.start_url}",
                    )
                )

            # 2. Main agent loop
            while state.current_step < state.max_steps and not self._stop_requested:
                state.current_step += 1
                step_idx = state.current_step

                # A. Observe Page
                observation = await self.observer.observe(page)
                state.current_url = observation.url

                await event_manager.emit(
                    EventType.OBSERVATION_CAPTURED,
                    f"Observed: {observation.title} ({observation.url})",
                    {"step": step_idx, "url": observation.url, "title": observation.title},
                    run_id=run_id,
                )

                # B. Injection Defense Check
                injections = self.safety.inspect_observation(observation)
                if injections:
                    for inj in injections:
                        await event_manager.emit(
                            EventType.SAFETY_ALERT,
                            inj.reason or "Adversarial injection detected",
                            {"risk": inj.risk_level, "pattern": inj.flagged_pattern},
                            run_id=run_id,
                        )

                # C. Query LLM for next action
                await event_manager.emit(
                    EventType.AGENT_THINKING,
                    f"Agent thinking for step {step_idx}...",
                    {"step": step_idx},
                    run_id=run_id,
                )

                response = await self.model_client.get_next_action(
                    goal=request.goal,
                    observation=observation,
                    step_number=step_idx,
                    max_steps=state.max_steps,
                )

                # D. Safety check on proposed action
                safety_check = self.safety.evaluate_action(response.action, observation)
                if not safety_check.is_safe:
                    await event_manager.emit(
                        EventType.SAFETY_ALERT,
                        f"Blocked unsafe action: {safety_check.reason}",
                        {"action": response.action.model_dump(), "safety": safety_check.model_dump()},
                        run_id=run_id,
                    )
                    state.status = RunStatus.AWAITING_CONFIRMATION
                    break

                await event_manager.emit(
                    EventType.ACTION_PROPOSED,
                    f"Proposed {response.action.action_type.value}: {response.action.description}",
                    {"action": response.action.model_dump(), "thought": response.thought.model_dump()},
                    run_id=run_id,
                )

                # E. Execute action
                exec_result = await self.executor.execute(response.action)
                await event_manager.emit(
                    EventType.ACTION_EXECUTED,
                    exec_result.message,
                    {"result": exec_result.model_dump()},
                    run_id=run_id,
                )

                # Record step history
                record = StepRecord(
                    step_number=step_idx,
                    observation=observation,
                    thought=response.thought,
                    action=response.action,
                    safety_check=safety_check,
                    execution_result=exec_result,
                )
                state.history.append(record)

                # F. Check termination
                if response.action.action_type == ActionType.FINISH:
                    state.status = RunStatus.COMPLETED
                    state.final_output = response.action.description
                    break
                elif response.action.action_type == ActionType.FAIL:
                    state.status = RunStatus.FAILED
                    state.error = response.action.description
                    break

                await asyncio.sleep(0.5)

            if self._stop_requested:
                state.status = RunStatus.STOPPED
            elif state.status == RunStatus.RUNNING and state.current_step >= state.max_steps:
                state.status = RunStatus.COMPLETED
                state.final_output = "Reached maximum step limit."

        except Exception as exc:
            state.status = RunStatus.FAILED
            state.error = str(exc)
            await event_manager.emit(
                EventType.ERROR,
                f"Agent run failed: {str(exc)}",
                {"error": str(exc)},
                run_id=run_id,
            )

        finally:
            self._is_running = False
            state.end_time = datetime.now(timezone.utc)
            await event_manager.emit(
                EventType.RUN_FINISHED,
                f"Run finished with status {state.status.value}",
                {"status": state.status, "steps": state.current_step},
                run_id=run_id,
            )
            # Browser cleanup is deferred or handled explicitly

        return state
