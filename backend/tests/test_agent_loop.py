"""Unit tests for Member A's ControlledAgentRunner.

Tests cognitive loop coordination, action limits, grounding validation,
repeated-action loop detection, completion verification, safety guardrails,
and structured event emission using mock dependencies.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
import pytest

from backend.app.agent_runner import (
    ControlledAgentRunner,
    StopReason,
    is_action_grounded,
    verify_task_completion,
)
from backend.app.model_client import (
    ModelResponseParseError,
    OllamaModelClient,
)
from backend.app.schemas import (
    ActionType,
    AgentResponse,
    AgentRunRequest,
    AgentThought,
    BrowserAction,
    ElementDescriptor,
    ExecutionResult,
    PageObservation,
    RiskLevel,
    RunStatus,
    SafetyCheckResult,
)


def make_sample_thought(reasoning: str = "Test reasoning") -> AgentThought:
    return AgentThought(
        reflection="Observing page",
        reasoning=reasoning,
        plan=["Step 1"],
    )


def make_sample_observation(
    url: str = "http://localhost:8080/test.html",
    title: str = "Test Page",
    dom_summary: str = "button#submit-btn 'Submit'",
    interactive_elements: list[ElementDescriptor] | None = None,
    snippet: str = "Test page text content",
) -> PageObservation:
    elements = interactive_elements or [
        ElementDescriptor(
            tag_name="button",
            selector="#submit-btn",
            id="submit-btn",
            text="Submit",
            role="button",
        )
    ]
    return PageObservation(
        url=url,
        title=title,
        dom_summary=dom_summary,
        interactive_elements=elements,
        page_text_snippet=snippet,
    )


@pytest.mark.asyncio
async def test_successful_multi_action_verified_completion():
    """Verify multi-action run succeeds when page evidence corroborates completion."""
    mock_model = MagicMock(spec=OllamaModelClient)
    mock_observer = MagicMock()
    mock_executor = MagicMock()
    mock_safety = MagicMock()
    mock_events = MagicMock()
    mock_events.emit = AsyncMock()

    obs_step1 = make_sample_observation(title="Form Page", snippet="Please click submit")
    obs_step2 = make_sample_observation(
        title="Confirmation Page",
        dom_summary="div 'Submission Confirmed'",
        snippet="Thank you! Your submission was confirmed and completed.",
    )
    mock_observer.observe = AsyncMock(side_effect=[obs_step1, obs_step2])

    resp1 = AgentResponse(
        thought=make_sample_thought("Click submit"),
        action=BrowserAction(
            action_type=ActionType.CLICK,
            selector="#submit-btn",
            description="Click submit button",
        ),
    )
    resp2 = AgentResponse(
        thought=make_sample_thought("Task complete"),
        action=BrowserAction(
            action_type=ActionType.FINISH,
            description="Form submitted and confirmed",
        ),
    )
    mock_model.get_next_action = AsyncMock(side_effect=[resp1, resp2])

    mock_executor.initialize = AsyncMock(return_value=MagicMock())
    mock_executor.execute = AsyncMock(
        return_value=ExecutionResult(
            success=True,
            action_type=ActionType.CLICK,
            message="Clicked #submit-btn",
        )
    )
    mock_safety.inspect_observation = MagicMock(return_value=[])
    mock_safety.evaluate_action = MagicMock(
        return_value=SafetyCheckResult(is_safe=True)
    )

    runner = ControlledAgentRunner(
        model_client=mock_model,
        observer=mock_observer,
        executor=mock_executor,
        safety=mock_safety,
        events=mock_events,
    )

    req = AgentRunRequest(goal="Submit the form", max_steps=5)
    state = await runner.run(req)

    assert state.status == RunStatus.COMPLETED
    assert "Verified" in (state.final_output or "")
    assert len(state.history) == 1
    assert mock_executor.execute.call_count == 1
    assert mock_observer.observe.call_count == 2


@pytest.mark.asyncio
async def test_repeated_action_stuck_detection():
    """Verify runner stops when same action is proposed 3 times on unchanged page."""
    mock_model = MagicMock(spec=OllamaModelClient)
    mock_observer = MagicMock()
    mock_executor = MagicMock()
    mock_safety = MagicMock()
    mock_events = MagicMock()
    mock_events.emit = AsyncMock()

    obs = make_sample_observation(url="http://localhost:8080/stuck", title="Stuck Page")
    mock_observer.observe = AsyncMock(return_value=obs)

    stuck_resp = AgentResponse(
        thought=make_sample_thought("Clicking again"),
        action=BrowserAction(
            action_type=ActionType.CLICK,
            selector="#submit-btn",
            description="Repeated click",
        ),
    )
    mock_model.get_next_action = AsyncMock(return_value=stuck_resp)
    mock_executor.initialize = AsyncMock(return_value=MagicMock())
    mock_executor.execute = AsyncMock(
        return_value=ExecutionResult(success=True, action_type=ActionType.CLICK, message="Clicked")
    )
    mock_safety.inspect_observation = MagicMock(return_value=[])
    mock_safety.evaluate_action = MagicMock(return_value=SafetyCheckResult(is_safe=True))

    runner = ControlledAgentRunner(
        model_client=mock_model,
        observer=mock_observer,
        executor=mock_executor,
        safety=mock_safety,
        events=mock_events,
        stuck_threshold=3,
    )

    state = await runner.run(AgentRunRequest(goal="Click button", max_steps=10))

    assert state.status == RunStatus.FAILED
    assert "Stuck loop detected" in (state.error or "")
    # Should stop at step 3 before indefinite loop
    assert state.current_step == 3


@pytest.mark.asyncio
async def test_malformed_model_output_error_handling():
    """Verify runner handles model parse failures gracefully without crashing."""
    mock_model = MagicMock(spec=OllamaModelClient)
    mock_observer = MagicMock()
    mock_executor = MagicMock()
    mock_safety = MagicMock()
    mock_events = MagicMock()
    mock_events.emit = AsyncMock()

    mock_observer.observe = AsyncMock(return_value=make_sample_observation())
    mock_model.get_next_action = AsyncMock(
        side_effect=ModelResponseParseError("Model produced invalid JSON syntax")
    )
    mock_executor.initialize = AsyncMock(return_value=MagicMock())
    mock_safety.inspect_observation = MagicMock(return_value=[])

    runner = ControlledAgentRunner(
        model_client=mock_model,
        observer=mock_observer,
        executor=mock_executor,
        safety=mock_safety,
        events=mock_events,
    )

    state = await runner.run(AgentRunRequest(goal="Any goal"))

    assert state.status == RunStatus.FAILED
    assert "Model client failure" in (state.error or "")


@pytest.mark.asyncio
async def test_executor_failure_handling():
    """Verify execution failures halt the run safely and record the error."""
    mock_model = MagicMock(spec=OllamaModelClient)
    mock_observer = MagicMock()
    mock_executor = MagicMock()
    mock_safety = MagicMock()
    mock_events = MagicMock()
    mock_events.emit = AsyncMock()

    mock_observer.observe = AsyncMock(return_value=make_sample_observation())
    mock_model.get_next_action = AsyncMock(
        return_value=AgentResponse(
            thought=make_sample_thought("Click missing"),
            action=BrowserAction(
                action_type=ActionType.CLICK,
                selector="#submit-btn",
                description="Click button",
            ),
        )
    )
    mock_executor.initialize = AsyncMock(return_value=MagicMock())
    mock_executor.execute = AsyncMock(
        return_value=ExecutionResult(
            success=False,
            action_type=ActionType.CLICK,
            message="Click failed",
            error="Node detached from document tree",
        )
    )
    mock_safety.inspect_observation = MagicMock(return_value=[])
    mock_safety.evaluate_action = MagicMock(return_value=SafetyCheckResult(is_safe=True))

    runner = ControlledAgentRunner(
        model_client=mock_model,
        observer=mock_observer,
        executor=mock_executor,
        safety=mock_safety,
        events=mock_events,
    )

    state = await runner.run(AgentRunRequest(goal="Click button"))

    assert state.status == RunStatus.FAILED
    assert "Execution failed" in (state.error or "")


@pytest.mark.asyncio
async def test_max_steps_action_limit():
    """Verify runner enforces the configured maximum step limit."""
    mock_model = MagicMock(spec=OllamaModelClient)
    mock_observer = MagicMock()
    mock_executor = MagicMock()
    mock_safety = MagicMock()
    mock_events = MagicMock()
    mock_events.emit = AsyncMock()

    # Changing observation URL so stuck detection doesn't trigger
    obs1 = make_sample_observation(url="http://localhost:8080/page1")
    obs2 = make_sample_observation(url="http://localhost:8080/page2")
    obs3 = make_sample_observation(url="http://localhost:8080/page3")
    mock_observer.observe = AsyncMock(side_effect=[obs1, obs2, obs3])

    click_action = AgentResponse(
        thought=make_sample_thought("Browse"),
        action=BrowserAction(
            action_type=ActionType.CLICK,
            selector="#submit-btn",
            description="Navigate page",
        ),
    )
    mock_model.get_next_action = AsyncMock(return_value=click_action)
    mock_executor.initialize = AsyncMock(return_value=MagicMock())
    mock_executor.execute = AsyncMock(
        return_value=ExecutionResult(success=True, action_type=ActionType.CLICK, message="OK")
    )
    mock_safety.inspect_observation = MagicMock(return_value=[])
    mock_safety.evaluate_action = MagicMock(return_value=SafetyCheckResult(is_safe=True))

    runner = ControlledAgentRunner(
        model_client=mock_model,
        observer=mock_observer,
        executor=mock_executor,
        safety=mock_safety,
        events=mock_events,
        default_max_steps=3,
    )

    state = await runner.run(AgentRunRequest(goal="Browse pages", max_steps=3))

    assert state.status == RunStatus.COMPLETED
    assert "limit of 3 actions" in (state.final_output or "")
    assert state.current_step == 3


@pytest.mark.asyncio
async def test_unverified_model_finish_rejected():
    """Verify uncorroborated finish claim is marked unverified/failed."""
    mock_model = MagicMock(spec=OllamaModelClient)
    mock_observer = MagicMock()
    mock_executor = MagicMock()
    mock_safety = MagicMock()
    mock_events = MagicMock()
    mock_events.emit = AsyncMock()

    # Page has no completion indicator
    unverified_obs = make_sample_observation(
        title="Error 404",
        dom_summary="h1 'Not Found'",
        snippet="The requested page could not be located on the server.",
    )
    mock_observer.observe = AsyncMock(return_value=unverified_obs)

    finish_resp = AgentResponse(
        thought=make_sample_thought("Done"),
        action=BrowserAction(
            action_type=ActionType.FINISH,
            description="I finished buying the item",
        ),
    )
    mock_model.get_next_action = AsyncMock(return_value=finish_resp)
    mock_executor.initialize = AsyncMock(return_value=MagicMock())
    mock_safety.inspect_observation = MagicMock(return_value=[])
    mock_safety.evaluate_action = MagicMock(return_value=SafetyCheckResult(is_safe=True))

    runner = ControlledAgentRunner(
        model_client=mock_model,
        observer=mock_observer,
        executor=mock_executor,
        safety=mock_safety,
        events=mock_events,
    )

    state = await runner.run(AgentRunRequest(goal="Purchase product"))

    assert state.status == RunStatus.FAILED
    assert "does not corroborate completion" in (state.error or "")


@pytest.mark.asyncio
async def test_ungrounded_selector_rejected():
    """Verify actions referencing non-existent selectors are rejected before execution."""
    mock_model = MagicMock(spec=OllamaModelClient)
    mock_observer = MagicMock()
    mock_executor = MagicMock()
    mock_safety = MagicMock()
    mock_events = MagicMock()
    mock_events.emit = AsyncMock()

    mock_observer.observe = AsyncMock(return_value=make_sample_observation())

    ungrounded_resp = AgentResponse(
        thought=make_sample_thought("Click ghost"),
        action=BrowserAction(
            action_type=ActionType.CLICK,
            selector="#phantom-element-never-observed",
            description="Click ghost button",
        ),
    )
    mock_model.get_next_action = AsyncMock(return_value=ungrounded_resp)
    mock_executor.initialize = AsyncMock(return_value=MagicMock())
    mock_safety.inspect_observation = MagicMock(return_value=[])

    runner = ControlledAgentRunner(
        model_client=mock_model,
        observer=mock_observer,
        executor=mock_executor,
        safety=mock_safety,
        events=mock_events,
    )

    state = await runner.run(AgentRunRequest(goal="Click button"))

    assert state.status == RunStatus.FAILED
    assert "not grounded" in (state.error or "")
    assert mock_executor.execute.call_count == 0


@pytest.mark.asyncio
async def test_observer_failure_handling():
    """Verify observer capture exception is caught safely."""
    mock_model = MagicMock(spec=OllamaModelClient)
    mock_observer = MagicMock()
    mock_executor = MagicMock()
    mock_safety = MagicMock()
    mock_events = MagicMock()
    mock_events.emit = AsyncMock()

    mock_observer.observe = AsyncMock(side_effect=RuntimeError("Browser page disconnected"))
    mock_executor.initialize = AsyncMock(return_value=MagicMock())

    runner = ControlledAgentRunner(
        model_client=mock_model,
        observer=mock_observer,
        executor=mock_executor,
        safety=mock_safety,
        events=mock_events,
    )

    state = await runner.run(AgentRunRequest(goal="Observe"))

    assert state.status == RunStatus.FAILED
    assert "Observation capture failed" in (state.error or "")


@pytest.mark.asyncio
async def test_safety_guard_blocks_action():
    """Verify unsafe action transitions run to AWAITING_CONFIRMATION."""
    mock_model = MagicMock(spec=OllamaModelClient)
    mock_observer = MagicMock()
    mock_executor = MagicMock()
    mock_safety = MagicMock()
    mock_events = MagicMock()
    mock_events.emit = AsyncMock()

    obs = make_sample_observation(
        dom_summary="button#purge-btn 'Purge Database'",
        interactive_elements=[
            ElementDescriptor(tag_name="button", selector="#purge-btn", text="Purge Database")
        ],
    )
    mock_observer.observe = AsyncMock(return_value=obs)

    destructive_resp = AgentResponse(
        thought=make_sample_thought("Delete all"),
        action=BrowserAction(
            action_type=ActionType.CLICK,
            selector="#purge-btn",
            description="Purge all user database records",
        ),
    )
    mock_model.get_next_action = AsyncMock(return_value=destructive_resp)
    mock_executor.initialize = AsyncMock(return_value=MagicMock())
    mock_safety.inspect_observation = MagicMock(return_value=[])
    mock_safety.evaluate_action = MagicMock(
        return_value=SafetyCheckResult(
            is_safe=False,
            risk_level=RiskLevel.CRITICAL,
            reason="Destructive keyword 'purge' detected",
            requires_human_confirmation=True,
        )
    )

    runner = ControlledAgentRunner(
        model_client=mock_model,
        observer=mock_observer,
        executor=mock_executor,
        safety=mock_safety,
        events=mock_events,
    )

    state = await runner.run(AgentRunRequest(goal="Manage data"))

    assert state.status == RunStatus.AWAITING_CONFIRMATION
    assert "Blocked unsafe action" in (state.error or "")
    assert mock_executor.execute.call_count == 0


@pytest.mark.asyncio
async def test_cancellation_request_stop():
    """Verify request_stop halts the loop at the next step."""
    mock_model = MagicMock(spec=OllamaModelClient)
    mock_observer = MagicMock()
    mock_executor = MagicMock()
    mock_safety = MagicMock()
    mock_events = MagicMock()
    mock_events.emit = AsyncMock()

    mock_observer.observe = AsyncMock(return_value=make_sample_observation())
    mock_executor.initialize = AsyncMock(return_value=MagicMock())

    runner = ControlledAgentRunner(
        model_client=mock_model,
        observer=mock_observer,
        executor=mock_executor,
        safety=mock_safety,
        events=mock_events,
    )

    # Trigger stop immediately
    runner.request_stop()
    state = await runner.run(AgentRunRequest(goal="Cancelled goal"))

    assert state.status == RunStatus.STOPPED
    assert "stopped by user request" in (state.final_output or "")
    assert mock_model.get_next_action.call_count == 0
