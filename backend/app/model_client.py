"""Ollama model client interface for Gemma 3 4B.

Responsible for:
1. Constructing system and user prompts with observation context.
2. Invoking Ollama API (gemma3:4b) via HTTP.
3. Parsing and validating structured JSON into AgentResponse.
4. Providing a mock fallback mode when Ollama is offline.

Owned by: AI Engineer
"""

import json
import re
from typing import Optional
import httpx
from .schemas import ActionType, AgentResponse, AgentThought, BrowserAction, PageObservation


SYSTEM_PROMPT = """You are BrowserPilot AI, an autonomous browser navigation agent.
Your mission is to achieve the user's goal by interacting with web pages step-by-step.

You MUST respond strictly with a valid JSON object matching this schema:
{
  "thought": {
    "reflection": "Brief reflection on previous action and current page state",
    "reasoning": "Deductive reasoning on what action to take next",
    "plan": ["Remaining step 1", "Remaining step 2"]
  },
  "action": {
    "action_type": "navigate" | "click" | "type" | "press_key" | "scroll" | "wait" | "extract" | "finish" | "fail",
    "selector": "CSS selector if applicable",
    "text": "text to type if applicable",
    "url": "url to navigate to if applicable",
    "key": "key to press if applicable",
    "description": "Clear explanation of this action",
    "target_element_description": "Description of target element"
  }
}
Do NOT include markdown formatting or explanations outside the JSON object.
"""


class OllamaModelClient:
    """Interface to local Ollama instance running gemma3:4b."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "gemma3:4b",
        timeout: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    async def get_next_action(
        self,
        goal: str,
        observation: PageObservation,
        step_number: int,
        max_steps: int,
    ) -> AgentResponse:
        """Query Ollama with the current goal and page observation."""
        user_prompt = self._format_user_prompt(goal, observation, step_number, max_steps)

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": user_prompt,
                        "system": SYSTEM_PROMPT,
                        "stream": False,
                        "format": "json",
                    },
                )
                if response.status_code == 200:
                    raw_text = response.json().get("response", "")
                    return self._parse_response(raw_text)
        except Exception:
            # Fall back to development stub if Ollama is unreachable
            pass

        return self._generate_fallback_response(goal, observation, step_number)

    def _format_user_prompt(
        self, goal: str, observation: PageObservation, step: int, max_steps: int
    ) -> str:
        """Construct user prompt combining goal and page observation."""
        return f"""User Goal: {goal}
Current Step: {step} of {max_steps}
Current URL: {observation.url}
Page Title: {observation.title}

Visible Page Elements:
{observation.dom_summary or "(No interactive elements detected)"}

Page Content Snippet:
{observation.page_text_snippet or "(Empty)"}

Select the next best action to accomplish the goal."""

    def _parse_response(self, raw_text: str) -> AgentResponse:
        """Extract and validate JSON from model output."""
        cleaned = raw_text.strip()
        # Remove markdown code fences if present
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\n?", "", cleaned)
            cleaned = re.sub(r"\n?```$", "", cleaned)

        data = json.loads(cleaned)
        response = AgentResponse.model_validate(data)
        response.raw_model_response = raw_text
        return response

    def _generate_fallback_response(
        self, goal: str, observation: PageObservation, step: int
    ) -> AgentResponse:
        """Heuristic stub used when Ollama is offline during scaffold testing."""
        return AgentResponse(
            thought=AgentThought(
                reflection=f"Step {step}: Observational review at {observation.url}",
                reasoning=f"Ollama offline or mocking mode. Progressing towards goal: '{goal}'.",
                plan=["Inspect elements", "Complete task"],
            ),
            action=BrowserAction(
                action_type=ActionType.WAIT,
                wait_seconds=1.0,
                description=f"Scaffold fallback step {step} for goal: {goal}",
                target_element_description="Waiting for next user instruction",
            ),
            raw_model_response="[Mock Response: Ollama offline]",
        )

    # TODO (AI Engineer): Add few-shot in-context demonstration examples
    # TODO (AI Engineer): Integrate vision/screenshot multimodal tokens when supported
