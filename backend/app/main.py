"""FastAPI application entrypoint for BrowserPilot AI.

Provides REST and WebSocket endpoints for dashboard communication,
agent lifecycle management, and static file serving.
"""

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .agent_loop import AgentLoop
from .events import event_manager
from .schemas import AgentRunRequest, AgentRunState, RunStatus

# Global agent loop instance
agent_loop = AgentLoop()
_active_task: Optional[asyncio.Task] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for startup and graceful shutdown."""
    yield
    # Cleanup browser resources on shutdown
    if agent_loop and agent_loop.executor:
        await agent_loop.executor.close()


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


@app.get("/api/health")
async def health_check():
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
async def get_recent_events(limit: int = 50):
    """Retrieve recent event history."""
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
