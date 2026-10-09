"""Playwright action executor module.

Responsible for launching Chromium, managing browser pages,
and executing atomic BrowserAction requests against the active page.

Owned by: Browser Automation Engineer
"""

import time
from typing import Optional
from playwright.async_api import Browser, BrowserContext, Page, async_playwright
from .schemas import ActionType, BrowserAction, ExecutionResult


class PlaywrightExecutor:
    """Manages Chromium lifecycle and executes browser actions."""

    def __init__(self, headless: bool = False, slow_mo_ms: int = 300) -> None:
        self.headless = headless
        self.slow_mo_ms = slow_mo_ms
        self._playwright = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None

    async def initialize(self) -> Page:
        """Start Playwright and launch Chromium browser."""
        if not self._playwright:
            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(
                headless=self.headless,
                slow_mo=self.slow_mo_ms,
            )
            self._context = await self._browser.new_context(
                viewport={"width": 1280, "height": 800}
            )
            self._page = await self._context.new_page()
        return self._page

    @property
    def current_page(self) -> Optional[Page]:
        """Return the active page instance."""
        return self._page

    async def execute(self, action: BrowserAction) -> ExecutionResult:
        """Execute a structured BrowserAction on the current page."""
        if not self._page:
            await self.initialize()

        page = self._page
        start_time = time.perf_counter()

        try:
            if action.action_type == ActionType.NAVIGATE:
                target_url = action.url or "about:blank"
                await page.goto(target_url, wait_until="domcontentloaded", timeout=15000)
                msg = f"Navigated to {target_url}"

            elif action.action_type == ActionType.CLICK:
                if not action.selector:
                    raise ValueError("CLICK action requires a valid selector")
                await page.click(action.selector, timeout=5000)
                msg = f"Clicked element: {action.selector}"

            elif action.action_type == ActionType.TYPE:
                if not action.selector:
                    raise ValueError("TYPE action requires a valid selector")
                text_to_type = action.text or ""
                await page.fill(action.selector, text_to_type, timeout=5000)
                msg = f"Typed '{text_to_type}' into {action.selector}"

            elif action.action_type == ActionType.PRESS_KEY:
                key = action.key or "Enter"
                await page.keyboard.press(key)
                msg = f"Pressed key '{key}'"

            elif action.action_type == ActionType.WAIT:
                wait_sec = action.wait_seconds or 1.0
                await page.wait_for_timeout(wait_sec * 1000)
                msg = f"Waited for {wait_sec} seconds"

            elif action.action_type == ActionType.SCROLL:
                delta_y = action.scroll_delta_y or 300
                await page.evaluate(f"window.scrollBy(0, {delta_y})")
                msg = f"Scrolled by {delta_y}px"

            elif action.action_type in (ActionType.FINISH, ActionType.FAIL):
                msg = f"Terminating action: {action.description}"

            else:
                msg = f"Action {action.action_type} executed (unhandled custom type)"

            duration_ms = (time.perf_counter() - start_time) * 1000
            return ExecutionResult(
                success=True,
                action_type=action.action_type,
                message=msg,
                duration_ms=duration_ms,
            )

        except Exception as exc:
            duration_ms = (time.perf_counter() - start_time) * 1000
            # TODO (Browser Automation Engineer): Implement automatic selector fallback / retries
            return ExecutionResult(
                success=False,
                action_type=action.action_type,
                message=f"Execution failed: {str(exc)}",
                duration_ms=duration_ms,
                error=str(exc),
            )

    async def close(self) -> None:
        """Tear down browser session and release resources."""
        if self._browser:
            await self._browser.close()
            self._browser = None
        if self._playwright:
            await self._playwright.stop()
            self._playwright = None
        self._page = None
        self._context = None
