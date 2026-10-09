"""FastAPI application entrypoint for BrowserPilot AI.

Provides REST and WebSocket endpoints for:
1. Health verification: GET /health
2. Browser observation: GET /observe
3. Browser action execution: POST /execute
4. Execution event telemetry: GET /events
5. Session flow control: POST /stop and POST /resume
6. Dashboard communication and static mock-site serving

Owned by: Browser Automation Engineer (Member B)
"""

import asyncio
from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from .agent_loop import AgentLoop
from .events import EventType, event_manager
from .executor import MOCK_SITE_DEFAULT_URL, PlaywrightExecutor
from .observer import PlaywrightObserver
from .safety import safety_guard
from .schemas import (
    ActionType,
    AgentRunRequest,
    AgentRunState,
    BrowserAction,
    ExecuteActionRequest,
    ExecutionResult,
    ObservationResponse,
    RunStatus,
    StopResponse,
)

# Concurrency lock to serialize all active browser operations
_browser_lock: Optional[asyncio.Lock] = None
_lock_loop: Optional[asyncio.AbstractEventLoop] = None


def get_browser_lock() -> asyncio.Lock:
    """Return an asyncio.Lock bound to the current running event loop."""
    global _browser_lock, _lock_loop
    try:
        current_loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.Lock()
    if _browser_lock is None or _lock_loop != current_loop:
        _browser_lock = asyncio.Lock()
        _lock_loop = current_loop
    return _browser_lock


# Headless mode: defaults to True for backend service, configurable via env var
HEADLESS_MODE = os.getenv("BROWSERPILOT_HEADLESS", "true").lower() in ("true", "1", "yes")

# Core browser observer and executor singletons
executor = PlaywrightExecutor(headless=HEADLESS_MODE)
observer = PlaywrightObserver()

# Global agent loop instance (Member A coordinator)
agent_loop = AgentLoop(observer=observer, executor=executor)
_active_task: Optional[asyncio.Task] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for startup and graceful shutdown."""
    yield
    # Cleanup browser resources on shutdown
    async with get_browser_lock():
        if executor:
            await executor.close()
        if observer:
            await observer.close()


app = FastAPI(
    title="BrowserPilot AI",
    description="Local-first autonomous browser agent backend",
    version="0.1.0",
    lifespan=lifespan,
)

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =====================================================================
# STEP 4 REQUIRED ENDPOINTS
# =====================================================================

@app.get("/", include_in_schema=False)
async def root():
    """Redirect root path to interactive Swagger API documentation."""
    return RedirectResponse(url="/docs")


@app.get("/health")
async def health_check():
    """Confirm the API server is running."""
    return {"status": "ok"}


@app.get("/observe", response_model=ObservationResponse)
async def observe_browser():
    """Return the current page URL, title, and visible interactive elements using observer.py."""
    async with get_browser_lock():
        try:
            # Ensure browser page is open and active
            if not executor.current_page or executor.current_page.is_closed():
                await executor.initialize()
                await executor.execute({"action": "navigate", "url": MOCK_SITE_DEFAULT_URL})
            elif executor.current_page.url in ("", "about:blank"):
                await executor.execute({"action": "navigate", "url": MOCK_SITE_DEFAULT_URL})

            page = executor.current_page
            obs = await observer.observe_structured(page)

            return ObservationResponse(
                url=obs.get("url", ""),
                title=obs.get("title", ""),
                elements=obs.get("elements", []),
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Observation failed: {str(exc)}")


@app.post("/execute", response_model=ExecutionResult)
async def execute_action(request: ExecuteActionRequest):
    """Accept one validated browser action, execute it through executor.py, and return actual result."""
    events_to_emit: List[Dict[str, Any]] = []

    async with get_browser_lock():
        # Check stop state before executing
        if executor.is_stopped:
            result = ExecutionResult(
                success=False,
                action_type=ActionType.FAIL,
                action=request.action,
                target=request.target,
                message="Execution is stopped. Action blocked.",
                error="Execution stopped",
            )
        else:
            # Action-level safety guard policy check at shared execution boundary
            act_type = request.action
            if isinstance(act_type, str):
                try:
                    act_type = ActionType(act_type.lower())
                except ValueError:
                    act_type = ActionType.FAIL

            browser_act = BrowserAction(
                action_type=act_type,
                selector=f'[data-agent-id="{request.target}"]' if request.target else None,
                text=request.text,
                url=request.url,
                description=f"Direct action {act_type.value if hasattr(act_type, 'value') else act_type}" + (f" on {request.target}" if request.target else ""),
            )
            safety_check = safety_guard.evaluate_action(browser_act)
            if not safety_check.is_safe:
                events_to_emit.append({
                    "type": EventType.SAFETY_ALERT,
                    "message": f"Action blocked by SafetyGuard: {safety_check.reason}",
                    "data": {
                        "action": act_type.value if hasattr(act_type, "value") else str(act_type),
                        "target": request.target,
                        "reason": safety_check.reason,
                    },
                })
                result = ExecutionResult(
                    success=False,
                    action_type=ActionType.FAIL,
                    action=act_type.value if hasattr(act_type, "value") else str(act_type),
                    target=request.target,
                    message=f"Action blocked by SafetyGuard: {safety_check.reason}",
                    error=safety_check.reason,
                )
            else:
                result = await executor.execute(request)
                # Check for prompt injections on navigation / page changes
                is_nav = act_type in (ActionType.NAVIGATE, "navigate") or bool(request.url)
                if result.success and is_nav and executor.current_page and not executor.current_page.is_closed():
                    try:
                        current_obs = await asyncio.wait_for(observer.observe(executor.current_page), timeout=2.0)
                        injections = safety_guard.inspect_observation(current_obs)
                        if injections:
                            for inj in injections:
                                events_to_emit.append({
                                    "type": EventType.SAFETY_ALERT,
                                    "message": f"Prompt injection detected on page: {inj.reason}",
                                    "data": {
                                        "reason": inj.reason,
                                        "pattern": inj.flagged_pattern,
                                        "url": executor.current_page.url,
                                    },
                                })
                            result.message = f"{result.message} [SAFETY ALERT: {injections[0].reason}]"
                    except Exception as exc:
                        logger.warning(f"Safety inspection on page observation failed or timed out: {exc}")

    # Outside browser lock: emit collected events
    for ev in events_to_emit:
        await event_manager.emit(
            ev["type"],
            message=ev["message"],
            data=ev["data"],
        )

    # Record event telemetry in event manager
    await event_manager.emit(
        EventType.ACTION_EXECUTED,
        message=result.message,
        data={
            "action": result.action or (result.action_type.value if hasattr(result.action_type, "value") else str(result.action_type)),
            "target": result.target,
            "success": result.success,
            "status": "success" if result.success else "failure",
            "error": result.error,
            "duration_ms": result.duration_ms,
        },
    )

    return result


@app.get("/events")
async def get_action_events(limit: int = 50):
    """Return recent browser action events, including timestamps, action names, and status."""
    raw_events = event_manager.get_history(limit=limit)
    formatted: List[Dict[str, Any]] = []
    for ev in raw_events:
        item = {
            "event_id": ev.event_id,
            "timestamp": ev.timestamp.isoformat(),
            "type": ev.type.value if hasattr(ev.type, "value") else str(ev.type),
            "message": ev.message,
            "action": ev.data.get("action", ""),
            "target": ev.data.get("target"),
            "status": ev.data.get("status", "success" if ev.data.get("success") else "failure"),
            "success": ev.data.get("success", False),
            "error": ev.data.get("error"),
            "duration_ms": ev.data.get("duration_ms", 0.0),
            "data": ev.data,
        }
        formatted.append(item)
    return formatted


@app.post("/stop", response_model=StopResponse)
async def stop_execution():
    """Set stop flag to prevent subsequent actions in execution layer."""
    executor.stop()
    agent_loop.request_stop()
    return StopResponse(status="stopped", message="Execution stopped")


@app.post("/resume")
async def resume_execution():
    """Reset stop flag to resume allowing actions."""
    executor.resume()
    return {"status": "resumed", "message": "Execution resumed"}


# =====================================================================
# AGENT LIFECYCLE & LEGACY BACKWARD-COMPATIBLE ENDPOINTS
# =====================================================================

@app.get("/api/health")
async def api_health_check():
    """Health status and service readiness check."""
    return {
        "status": "healthy",
        "service": "BrowserPilot AI Backend",
        "version": "0.1.0",
        "agent_running": agent_loop._is_running,
    }


@app.post("/api/agent/start")
async def start_agent(request: AgentRunRequest):
    """Start an autonomous agent run in a background task."""
    global _active_task
    if agent_loop._is_running:
        return {"error": "Agent is already running. Please stop the current run first."}

    _active_task = asyncio.create_task(agent_loop.run(request))
    return {
        "message": "Agent execution initiated",
        "goal": request.goal,
        "max_steps": request.max_steps,
    }


@app.post("/api/agent/stop")
async def stop_agent():
    """Request the active agent run to halt."""
    if not agent_loop._is_running:
        return {"message": "Agent is not currently running"}

    agent_loop.request_stop()
    executor.stop()
    return {"message": "Stop signal sent to agent"}


@app.get("/api/agent/state", response_model=Optional[AgentRunState])
async def get_agent_state():
    """Retrieve the current state snapshot of the agent."""
    return agent_loop.current_state or AgentRunState(
        run_id="idle",
        goal="No active goal",
        status=RunStatus.IDLE,
    )


@app.get("/api/events")
async def get_recent_api_events(limit: int = 50):
    """Retrieve recent event history (raw AgentEvent models)."""
    return [event.model_dump() for event in event_manager.get_history(limit=limit)]


@app.websocket("/ws/events")
async def websocket_events(websocket: WebSocket):
    """WebSocket stream for real-time telemetry and timeline events."""
    await event_manager.connect(websocket)
    try:
        while True:
            # Keep-alive receive loop
            await websocket.receive_text()
    except WebSocketDisconnect:
        await event_manager.disconnect(websocket)


# Mount static assets for Mock Site and Dashboard
BASE_DIR = Path(__file__).resolve().parent.parent.parent
mock_site_path = BASE_DIR / "mock-site"
dashboard_path = BASE_DIR / "dashboard"

if mock_site_path.exists():
    app.mount("/mock", StaticFiles(directory=str(mock_site_path), html=True), name="mock-site")

if dashboard_path.exists():
    app.mount("/dashboard", StaticFiles(directory=str(dashboard_path), html=True), name="dashboard")
