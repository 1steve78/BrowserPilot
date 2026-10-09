"""Pydantic v2 schemas for BrowserPilot AI.

Defines the shared contracts for:
- Page observations
- Browser actions
- Agent thoughts & responses
- Execution results
- Run lifecycle states
- Safety check results
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, model_validator


class ActionType(str, Enum):
    """Supported browser action types."""
    NAVIGATE = "navigate"
    CLICK = "click"
    TYPE = "type"
    PRESS_KEY = "press_key"
    SCROLL = "scroll"
    WAIT = "wait"
    EXTRACT = "extract"
    FINISH = "finish"
    FAIL = "fail"


class RunStatus(str, Enum):
    """Lifecycle status of an agent run."""
    IDLE = "idle"
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    AWAITING_CONFIRMATION = "awaiting_confirmation"
    COMPLETED = "completed"
    STOPPED = "stopped"
    FAILED = "failed"


class RiskLevel(str, Enum):
    """Risk severity classification for safety guardrails."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ElementCoordinates(BaseModel):
    """Bounding box coordinates of an element."""
    x: float = Field(..., description="X coordinate of top-left corner")
    y: float = Field(..., description="Y coordinate of top-left corner")
    width: float = Field(..., description="Width of the element")
    height: float = Field(..., description="Height of the element")


class ElementDescriptor(BaseModel):
    """Descriptor for an interactive DOM element observed on the page."""
    id: Optional[str] = Field(default=None, description="DOM ID attribute")
    tag_name: str = Field(..., description="HTML tag name (e.g. button, input, a)")
    selector: str = Field(..., description="Recommended CSS or XPath selector")
    text: Optional[str] = Field(default=None, description="Visible inner text or placeholder")
    role: Optional[str] = Field(default=None, description="ARIA role or computed role")
    aria_label: Optional[str] = Field(default=None, description="ARIA label attribute")
    is_interactive: bool = Field(default=True, description="Whether element is clickable/typeable")
    coordinates: Optional[ElementCoordinates] = Field(default=None, description="Screen coordinates")


class PageObservation(BaseModel):
    """Structured observation of the active browser page."""
    url: str = Field(..., description="Current page URL")
    title: str = Field(default="", description="Current page title")
    dom_summary: str = Field(
        default="",
        description="Distilled, token-efficient DOM representation formatted for LLM consumption"
    )
    interactive_elements: List[ElementDescriptor] = Field(
        default_factory=list,
        description="List of detected interactive elements"
    )
    page_text_snippet: Optional[str] = Field(
        default=None,
        description="Truncated text content of the visible viewport"
    )
    screenshot_base64: Optional[str] = Field(
        default=None,
        description="Optional base64-encoded screenshot image for multimodal validation"
    )
    error: Optional[str] = Field(
        default=None,
        description="Observation error description if capture partially failed"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp of observation"
    )


class BrowserAction(BaseModel):
    """Next atomic action to execute on the browser."""
    action_type: ActionType = Field(..., description="Action to perform")
    target: Optional[str] = Field(
        default=None,
        description="Target element data-agent-id or identifier (for CLICK, TYPE, WAIT)"
    )
    selector: Optional[str] = Field(
        default=None,
        description="CSS selector or XPath targeting the element (for CLICK, TYPE, EXTRACT)"
    )
    text: Optional[str] = Field(
        default=None,
        description="Text content to type (for TYPE action)"
    )
    url: Optional[str] = Field(
        default=None,
        description="Target URL to open (for NAVIGATE action)"
    )
    key: Optional[str] = Field(
        default=None,
        description="Keyboard key to press (e.g. 'Enter', 'Tab', 'Escape')"
    )
    scroll_delta_y: Optional[int] = Field(
        default=None,
        description="Vertical scroll amount in pixels (positive down, negative up)"
    )
    wait_seconds: Optional[float] = Field(
        default=1.0,
        description="Duration in seconds to wait (for WAIT action)"
    )
    description: str = Field(
        ...,
        description="Human-readable explanation of why this action was chosen"
    )
    target_element_description: Optional[str] = Field(
        default=None,
        description="Plain text description of the target element (for verification)"
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_action_payload(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "action" in data and "action_type" not in data:
                data["action_type"] = data["action"]
            if "delta_y" in data and "scroll_delta_y" not in data:
                data["scroll_delta_y"] = data["delta_y"]
            if "amount" in data and "scroll_delta_y" not in data:
                data["scroll_delta_y"] = data["amount"]
            if "duration" in data and "wait_seconds" not in data:
                data["wait_seconds"] = data["duration"]
            if "seconds" in data and "wait_seconds" not in data:
                data["wait_seconds"] = data["seconds"]
        return data


class AgentThought(BaseModel):
    """Reasoning traces generated by the AI agent prior to acting."""
    reflection: str = Field(
        ...,
        description="Assessment of previous action result and current situation"
    )
    reasoning: str = Field(
        ...,
        description="Analytical deduction and decision logic for the next step"
    )
    plan: List[str] = Field(
        default_factory=list,
        description="Anticipated remaining high-level steps to achieve the goal"
    )


class AgentResponse(BaseModel):
    """Structured response contract produced by Ollama / Gemma 3."""
    thought: AgentThought = Field(..., description="Cognitive thought process")
    action: BrowserAction = Field(..., description="Structured browser action")
    raw_model_response: Optional[str] = Field(
        default=None,
        description="Raw unparsed model text response for telemetry and auditing"
    )


class ExecutionResult(BaseModel):
    """Result returned by PlaywrightExecutor after attempting an action."""
    success: bool = Field(..., description="Whether action execution succeeded")
    action_type: ActionType = Field(..., description="Action type attempted")
    action: Optional[str] = Field(default=None, description="Action name string alias")
    target: Optional[str] = Field(default=None, description="Target element identifier")
    message: str = Field(..., description="Execution outcome or extracted data summary")
    duration_ms: float = Field(default=0.0, description="Execution duration in milliseconds")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Execution timestamp"
    )
    error: Optional[str] = Field(default=None, description="Detailed error message if failed")
    extracted_data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Data extracted from page if action was EXTRACT"
    )

    @model_validator(mode="before")
    @classmethod
    def sync_action_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "action" in data and "action_type" not in data:
                data["action_type"] = data["action"]
            elif "action_type" in data and "action" not in data:
                act = data["action_type"]
                data["action"] = act.value if hasattr(act, "value") else str(act)
        return data


class SafetyCheckResult(BaseModel):
    """Outcome of safety and prompt-injection guardrails evaluation."""
    is_safe: bool = Field(..., description="Whether the action is deemed safe to execute")
    risk_level: RiskLevel = Field(default=RiskLevel.LOW, description="Assessed risk level")
    reason: Optional[str] = Field(default=None, description="Explanation for safety classification")
    flagged_pattern: Optional[str] = Field(
        default=None,
        description="Triggered regex, keyword, or adversarial marker"
    )
    requires_human_confirmation: bool = Field(
        default=False,
        description="Whether agent must pause for human confirmation before continuing"
    )


class AgentRunRequest(BaseModel):
    """Payload to initiate an autonomous browser session."""
    goal: str = Field(..., min_length=3, description="Natural language goal for the agent")
    start_url: Optional[str] = Field(
        default=None,
        description="Initial URL to navigate to (defaults to MOCK_SITE_URL or configured home)"
    )
    max_steps: Optional[int] = Field(
        default=15,
        ge=1,
        le=50,
        description="Maximum execution steps before automatic timeout"
    )


class StepRecord(BaseModel):
    """Historical trace record for a single step in an agent run."""
    step_number: int
    observation: PageObservation
    thought: AgentThought
    action: BrowserAction
    safety_check: SafetyCheckResult
    execution_result: Optional[ExecutionResult] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AgentRunState(BaseModel):
    """Complete state snapshot of an active or finished agent session."""
    run_id: str = Field(..., description="Unique UUID for this run")
    goal: str = Field(..., description="Goal provided by user")
    status: RunStatus = Field(default=RunStatus.IDLE, description="Current run state")
    current_step: int = Field(default=0, description="Current step index")
    max_steps: int = Field(default=15, description="Configured step threshold")
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    current_url: Optional[str] = None
    history: List[StepRecord] = Field(default_factory=list, description="Step execution history")
    error: Optional[str] = None
    final_output: Optional[str] = None


class ExecuteActionRequest(BaseModel):
    """Payload to execute an atomic browser action via HTTP API."""
    action: str = Field(..., description="Action name: click, type, scroll, navigate, wait, press_key")
    target: Optional[str] = Field(default=None, description="Target data-agent-id or selector")
    text: Optional[str] = Field(default=None, description="Text to type")
    url: Optional[str] = Field(default=None, description="Target URL")
    key: Optional[str] = Field(default=None, description="Keyboard key to press")
    scroll_delta_y: Optional[int] = Field(default=None, description="Vertical scroll delta")
    delta_y: Optional[int] = Field(default=None, description="Alias for scroll_delta_y")
    wait_seconds: Optional[float] = Field(default=None, description="Duration in seconds to wait")
    seconds: Optional[float] = Field(default=None, description="Alias for wait_seconds")
    description: Optional[str] = Field(default="", description="Optional human-readable description")

    @model_validator(mode="before")
    @classmethod
    def normalize_execute_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "delta_y" in data and "scroll_delta_y" not in data:
                data["scroll_delta_y"] = data["delta_y"]
            if "seconds" in data and "wait_seconds" not in data:
                data["wait_seconds"] = data["seconds"]
            if "duration" in data and "wait_seconds" not in data:
                data["wait_seconds"] = data["duration"]
        return data


class ObservationResponse(BaseModel):
    """Structured response for GET /observe endpoint."""
    url: str = Field(..., description="Current page URL")
    title: str = Field(default="", description="Current page title")
    elements: List[Dict[str, Any]] = Field(default_factory=list, description="Visible interactive elements")


class StopResponse(BaseModel):
    """Response returned by POST /stop endpoint."""
    status: str = Field(default="stopped", description="Status confirmation")
    message: str = Field(default="Execution stopped", description="Outcome description")

