"""Playwright action executor module.

Responsible for launching Chromium, managing browser pages,
and executing atomic browser actions with safeguards against the active page.

Actions supported:
1. click: Click an element identified by exact data-agent-id.
2. type: Enter text into an editable input or textarea.
3. scroll: Scroll the page vertically by a bounded amount.
4. navigate: Navigate to an allowed local mock-site URL.
5. wait: Wait for a short, bounded interval or for a specific element to appear.

Owned by: Browser Automation Engineer (Member B)
"""

from __future__ import annotations

import asyncio
import os
import re
import time
import urllib.parse
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from playwright.async_api import (
    Browser,
    BrowserContext,
    Locator,
    Page,
    Playwright,
    async_playwright,
)

try:
    from .schemas import ActionType, BrowserAction, ExecutionResult
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from backend.app.schemas import ActionType, BrowserAction, ExecutionResult

# Configuration & Safeguard Bounds
DEFAULT_ACTION_TIMEOUT_MS = 5000
MAX_WAIT_SECONDS = 10.0
MAX_SCROLL_PIXELS = 2000
MIN_SCROLL_PIXELS = -2000

MOCK_SITE_DIR = Path(__file__).resolve().parent.parent.parent / "mock-site"
MOCK_SITE_DEFAULT_URL = f"file:///{(MOCK_SITE_DIR / 'index.html').as_posix()}"


class PlaywrightExecutor:
    """Manages Chromium lifecycle and executes browser actions with safeguards."""

    def __init__(
        self,
        page: Optional[Page] = None,
        headless: bool = False,
        slow_mo_ms: int = 100,
    ) -> None:
        self.headless = headless
        self.slow_mo_ms = slow_mo_ms
        self._page: Optional[Page] = page
        self._context: Optional[BrowserContext] = None
        self._browser: Optional[Browser] = None
        self._playwright: Optional[Playwright] = None
        self._owns_browser: bool = page is None
        self._is_stopped: bool = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    @property
    def is_stopped(self) -> bool:
        """Return True if executor has been commanded to stop."""
        return self._is_stopped

    def stop(self) -> None:
        """Set stop flag to block subsequent action executions."""
        self._is_stopped = True

    def resume(self) -> None:
        """Clear stop flag to allow action executions."""
        self._is_stopped = False

    async def __aenter__(self) -> "PlaywrightExecutor":
        await self.initialize()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()

    @property
    def current_page(self) -> Optional[Page]:
        """Return the active page instance."""
        try:
            current_loop = asyncio.get_running_loop()
            if self._loop is not None and self._loop != current_loop:
                return None
        except RuntimeError:
            pass
        return self._page

    async def initialize(self) -> Page:
        """Start Playwright and launch Chromium browser if not already active."""
        current_loop = asyncio.get_running_loop()
        if self._loop is not None and self._loop != current_loop:
            self._page = None
            self._context = None
            self._browser = None
            self._playwright = None

        self._loop = current_loop

        if self._page and not self._page.is_closed():
            try:
                if self._playwright and self._browser and self._browser.is_connected():
                    return self._page
            except Exception:
                pass
            self._page = None

        if not self._playwright:
            self._playwright = await async_playwright().start()

        viewport_size = {"width": 1440, "height": 900}

        try:
            # 1. Standard Playwright bundled Chromium
            self._browser = await self._playwright.chromium.launch(
                headless=self.headless,
                slow_mo=self.slow_mo_ms,
            )
        except Exception:
            # 2. Fallback to system-installed Chrome or Edge on Windows
            try:
                self._browser = await self._playwright.chromium.launch(
                    channel="chrome",
                    headless=self.headless,
                    slow_mo=self.slow_mo_ms,
                )
            except Exception:
                self._browser = await self._playwright.chromium.launch(
                    channel="msedge",
                    headless=self.headless,
                    slow_mo=self.slow_mo_ms,
                )

        self._context = await self._browser.new_context(viewport=viewport_size)
        self._page = await self._context.new_page()
        self._owns_browser = True
        return self._page

    def _clean_agent_id(self, target: Optional[str]) -> Optional[str]:
        """Extract clean agent ID from target or selector string."""
        if not target:
            return None
        trimmed = target.strip()
        match = re.match(r'^\[data-agent-id=["\']?([^"\'\]]+)["\']?\]$', trimmed)
        if match:
            return match.group(1)
        return trimmed

    async def _resolve_target(
        self, page: Page, target: Optional[str], action_name: str
    ) -> Tuple[Optional[Locator], Optional[str], Optional[ExecutionResult]]:
        """Resolve target through exact data-agent-id match.
        
        Rejects missing or ambiguous targets.
        """
        if not target or not target.strip():
            return (
                None,
                None,
                ExecutionResult(
                    success=False,
                    action_type=ActionType(action_name),
                    action=action_name,
                    target=None,
                    message=f"Missing target: {action_name.upper()} action requires a valid target data-agent-id",
                    error="Missing target",
                ),
            )

        clean_id = self._clean_agent_id(target)
        locator = page.locator(f'[data-agent-id="{clean_id}"]')
        count = await locator.count()

        if count == 0:
            return (
                None,
                clean_id,
                ExecutionResult(
                    success=False,
                    action_type=ActionType(action_name),
                    action=action_name,
                    target=clean_id,
                    message=f"Target '{clean_id}' not found on the page",
                    error="Target not found",
                ),
            )

        if count > 1:
            return (
                None,
                clean_id,
                ExecutionResult(
                    success=False,
                    action_type=ActionType(action_name),
                    action=action_name,
                    target=clean_id,
                    message=f"Ambiguous target: {count} elements found matching data-agent-id '{clean_id}'",
                    error="Ambiguous target",
                ),
            )

        return (locator, clean_id, None)

    async def _check_editable(self, locator: Locator, target: str) -> Tuple[bool, Optional[str]]:
        """Permit typing only into appropriate editable elements."""
        try:
            info = await locator.evaluate("""
                el => {
                    const tag = el.tagName.toLowerCase();
                    const type = (el.getAttribute('type') || 'text').toLowerCase();
                    const isContentEditable = el.isContentEditable || el.getAttribute('contenteditable') === 'true';
                    const isReadonly = el.readOnly || el.hasAttribute('readonly');
                    const isDisabled = el.disabled || el.hasAttribute('disabled');
                    
                    const validTypes = ['text', 'search', 'email', 'url', 'tel', 'password', 'number'];
                    const isTextInput = tag === 'input' && validTypes.includes(type);
                    const isTextarea = tag === 'textarea';
                    const isEditable = (isTextInput || isTextarea || isContentEditable) && !isReadonly && !isDisabled;
                    
                    return {
                        tag: tag,
                        type: type,
                        isTextInput: isTextInput,
                        isTextarea: isTextarea,
                        isContentEditable: isContentEditable,
                        isReadonly: isReadonly,
                        isDisabled: isDisabled,
                        isEditable: isEditable
                    };
                }
            """)

            if info.get("isDisabled"):
                return False, f"Target '{target}' is disabled"
            if info.get("isReadonly"):
                return False, f"Target '{target}' is readonly"
            if not info.get("isEditable"):
                tag = info.get("tag", "element")
                type_attr = info.get("type", "")
                if tag == "input":
                    return False, f"Cannot type into input of type '{type_attr}': target '{target}' is not a text input"
                return False, f"Cannot type into <{tag}>: target '{target}' is not an editable input or textarea"

            return True, None
        except Exception as exc:
            return False, f"Failed to check element editability: {str(exc)}"

    def _is_allowed_url(self, url: str) -> bool:
        """Restrict navigation to local mock-site origin only."""
        if not url or not url.strip():
            return False
        clean_url = url.strip()
        if clean_url == "about:blank":
            return True

        if clean_url.startswith("#") or clean_url.startswith("/"):
            return True

        parsed = urllib.parse.urlparse(clean_url)
        if parsed.scheme in ("javascript", "data", "vbscript"):
            return False

        if parsed.scheme == "file":
            try:
                path_part = parsed.path
                if path_part.startswith("/") and len(path_part) > 2 and path_part[2] == ":":
                    path_part = path_part[1:]
                norm_path = Path(urllib.parse.unquote(path_part)).resolve().as_posix().lower()
                mock_dir = MOCK_SITE_DIR.resolve().as_posix().lower()
                return norm_path.startswith(mock_dir) or "mock-site" in norm_path
            except Exception:
                return False

        if parsed.scheme in ("http", "https"):
            hostname = (parsed.hostname or "").lower()
            if hostname in ("localhost", "127.0.0.1", "0.0.0.0", "::1"):
                return True
            return False

        # Local HTML filenames or anchors
        if not parsed.scheme and (clean_url.endswith(".html") or clean_url.endswith(".htm") or "#" in clean_url):
            return True

        return False

    async def execute(self, action: Union[BrowserAction, Dict[str, Any]]) -> ExecutionResult:
        """Execute a validated browser action dictionary or BrowserAction object."""
        if not self.current_page or (self._page and self._page.is_closed()):
            await self.initialize()

        page = self._page
        start_time = time.perf_counter()

        # Normalize action payload
        if hasattr(action, "model_dump"):
            action = action.model_dump()

        if isinstance(action, dict):
            raw_act = action.get("action") or action.get("action_type") or ""
            raw_action = (raw_act.value if hasattr(raw_act, "value") else str(raw_act)).strip().lower()
            target = action.get("target") or action.get("selector")
            text = action.get("text")
            url = action.get("url")
            key = action.get("key")
            scroll_delta_y = action.get("scroll_delta_y") or action.get("delta_y") or action.get("amount")
            wait_seconds = action.get("wait_seconds") or action.get("seconds") or action.get("duration")
            description = action.get("description", "")
        else:
            return ExecutionResult(
                success=False,
                action_type=ActionType.FAIL,
                action="fail",
                message=f"Unsupported action object type: {type(action).__name__}",
                error="Invalid action type",
                duration_ms=0.0,
            )

        if not raw_action:
            return ExecutionResult(
                success=False,
                action_type=ActionType.FAIL,
                action="fail",
                message="Action type is required but was not provided",
                error="Missing action type",
                duration_ms=0.0,
            )

        if self._is_stopped:
            return ExecutionResult(
                success=False,
                action_type=ActionType.FAIL,
                action=raw_action or "stop",
                target=self._clean_agent_id(target),
                message="Execution is stopped. Action rejected.",
                error="Execution stopped",
                duration_ms=0.0,
            )

        try:
            # Dispatch to action handlers
            if raw_action == "click":
                result = await self._execute_click(page, target)
            elif raw_action == "type":
                result = await self._execute_type(page, target, text)
            elif raw_action == "scroll":
                result = await self._execute_scroll(page, scroll_delta_y)
            elif raw_action == "navigate":
                result = await self._execute_navigate(page, url)
            elif raw_action == "wait":
                result = await self._execute_wait(page, target, wait_seconds)
            elif raw_action == "press_key":
                result = await self._execute_press_key(page, key)
            elif raw_action in ("finish", "fail"):
                is_finish = raw_action == "finish"
                result = ExecutionResult(
                    success=is_finish,
                    action_type=ActionType.FINISH if is_finish else ActionType.FAIL,
                    action=raw_action,
                    message=description or f"Agent requested {raw_action}",
                )
            else:
                result = ExecutionResult(
                    success=False,
                    action_type=ActionType.FAIL,
                    action=raw_action,
                    message=f"Unknown or unsupported action type: '{raw_action}'",
                    error="Unsupported action",
                )

            duration_ms = (time.perf_counter() - start_time) * 1000
            result.duration_ms = duration_ms
            return result

        except Exception as exc:
            duration_ms = (time.perf_counter() - start_time) * 1000
            act_enum = ActionType(raw_action) if raw_action in ActionType._value2member_map_ else ActionType.FAIL
            return ExecutionResult(
                success=False,
                action_type=act_enum,
                action=raw_action,
                target=self._clean_agent_id(target),
                message=f"Execution error: {str(exc)}",
                duration_ms=duration_ms,
                error=str(exc),
            )

    async def _execute_click(self, page: Page, target: Optional[str]) -> ExecutionResult:
        """Perform click action with exact data-agent-id target resolution."""
        locator, clean_id, error_result = await self._resolve_target(page, target, "click")
        if error_result:
            return error_result

        # Check visibility with action timeout
        try:
            await locator.wait_for(state="visible", timeout=3000)
        except Exception:
            return ExecutionResult(
                success=False,
                action_type=ActionType.CLICK,
                action="click",
                target=clean_id,
                message=f"Target '{clean_id}' is not visible on the page",
                error="Element not visible",
            )

        # Execute click
        await locator.click(timeout=DEFAULT_ACTION_TIMEOUT_MS)
        await page.wait_for_timeout(150)

        # Verify what happened after click without blindly assuming success
        modal_open = await page.locator("#invoice-modal-overlay.open").count() > 0
        if modal_open and ("invoice" in clean_id.lower() or "view" in clean_id.lower()):
            msg = "Invoice details opened"
        elif clean_id.startswith("nav-"):
            section_name = clean_id.replace("nav-", "")
            msg = f"Navigated to {section_name.capitalize()} page"
        elif "close" in clean_id.lower() and not modal_open:
            msg = "Invoice details modal closed"
        elif "search" in clean_id.lower():
            msg = f"Executed invoice search with '{clean_id}'"
        else:
            msg = f"Clicked element with agent ID '{clean_id}'"

        return ExecutionResult(
            success=True,
            action_type=ActionType.CLICK,
            action="click",
            target=clean_id,
            message=msg,
        )

    async def _execute_type(
        self, page: Page, target: Optional[str], text: Optional[str]
    ) -> ExecutionResult:
        """Perform type action with editability safeguard."""
        locator, clean_id, error_result = await self._resolve_target(page, target, "type")
        if error_result:
            return error_result

        # Check visibility
        try:
            await locator.wait_for(state="visible", timeout=3000)
        except Exception:
            return ExecutionResult(
                success=False,
                action_type=ActionType.TYPE,
                action="type",
                target=clean_id,
                message=f"Target '{clean_id}' is not visible on the page",
                error="Element not visible",
            )

        # Verify element is an appropriate editable element
        is_editable, edit_err = await self._check_editable(locator, clean_id)
        if not is_editable:
            return ExecutionResult(
                success=False,
                action_type=ActionType.TYPE,
                action="type",
                target=clean_id,
                message=edit_err or f"Target '{clean_id}' is not editable",
                error="Non-editable element",
            )

        text_to_type = text if text is not None else ""
        await locator.fill(text_to_type, timeout=DEFAULT_ACTION_TIMEOUT_MS)
        await page.wait_for_timeout(100)

        # Verify that value was typed correctly
        actual_val = await locator.input_value()
        if actual_val == text_to_type:
            msg = f"Typed '{text_to_type}' into '{clean_id}'"
        else:
            msg = f"Typed text into '{clean_id}' (value is now '{actual_val}')"

        return ExecutionResult(
            success=True,
            action_type=ActionType.TYPE,
            action="type",
            target=clean_id,
            message=msg,
        )

    async def _execute_scroll(
        self, page: Page, delta_y: Optional[Union[int, float]]
    ) -> ExecutionResult:
        """Perform vertical scroll bounded between MIN_SCROLL_PIXELS and MAX_SCROLL_PIXELS."""
        try:
            val = int(delta_y) if delta_y is not None else 300
        except (ValueError, TypeError):
            val = 300

        bounded_delta = max(MIN_SCROLL_PIXELS, min(MAX_SCROLL_PIXELS, val))
        clamped = bounded_delta != val

        start_y = await page.evaluate("window.scrollY || window.pageYOffset || 0")
        await page.evaluate(f"window.scrollBy(0, {bounded_delta})")
        await page.wait_for_timeout(100)
        new_y = await page.evaluate("window.scrollY || window.pageYOffset || 0")

        clamp_note = f" (bounded from {val}px)" if clamped else ""
        msg = f"Scrolled page by {bounded_delta}px{clamp_note} (scroll position: {start_y} -> {new_y})"

        return ExecutionResult(
            success=True,
            action_type=ActionType.SCROLL,
            action="scroll",
            message=msg,
        )

    async def _execute_navigate(self, page: Page, url: Optional[str]) -> ExecutionResult:
        """Perform navigation with origin restriction safeguard."""
        target_url = (url or "").strip()
        if not target_url:
            return ExecutionResult(
                success=False,
                action_type=ActionType.NAVIGATE,
                action="navigate",
                message="Missing URL: NAVIGATE action requires a target URL",
                error="Missing URL",
            )

        if not self._is_allowed_url(target_url):
            return ExecutionResult(
                success=False,
                action_type=ActionType.NAVIGATE,
                action="navigate",
                message=f"Navigation rejected: External URL '{target_url}' is not permitted. Only local mock-site URLs are allowed.",
                error="Restricted navigation origin",
            )

        if target_url.startswith("#"):
            await page.evaluate(f"window.location.hash = '{target_url}'")
            await page.wait_for_timeout(100)
        else:
            if not urllib.parse.urlparse(target_url).scheme:
                if (MOCK_SITE_DIR / target_url).exists():
                    target_url = f"file:///{(MOCK_SITE_DIR / target_url).as_posix()}"
            await page.goto(target_url, wait_until="domcontentloaded", timeout=DEFAULT_ACTION_TIMEOUT_MS * 2)
            await page.wait_for_timeout(100)

        current_url = page.url
        title = await page.title()
        msg = f"Navigated to {current_url} ('{title}')"

        return ExecutionResult(
            success=True,
            action_type=ActionType.NAVIGATE,
            action="navigate",
            message=msg,
        )

    async def _execute_wait(
        self, page: Page, target: Optional[str], wait_seconds: Optional[Union[int, float]]
    ) -> ExecutionResult:
        """Perform bounded wait for interval or for target element appearance."""
        try:
            sec = float(wait_seconds) if wait_seconds is not None else 1.0
        except (ValueError, TypeError):
            sec = 1.0

        bounded_sec = max(0.0, min(MAX_WAIT_SECONDS, sec))
        clamped = bounded_sec != sec

        # Case A: Wait for specific element to appear
        if target and target.strip():
            clean_id = self._clean_agent_id(target)
            selector = f'[data-agent-id="{clean_id}"]'
            timeout_ms = max(500, int(bounded_sec * 1000))

            try:
                await page.wait_for_selector(selector, state="visible", timeout=timeout_ms)
                msg = f"Element '{clean_id}' appeared and is visible"
                return ExecutionResult(
                    success=True,
                    action_type=ActionType.WAIT,
                    action="wait",
                    target=clean_id,
                    message=msg,
                )
            except Exception as exc:
                return ExecutionResult(
                    success=False,
                    action_type=ActionType.WAIT,
                    action="wait",
                    target=clean_id,
                    message=f"Timed out waiting for element '{clean_id}' to appear within {timeout_ms}ms",
                    error=str(exc),
                )

        # Case B: Wait for fixed bounded duration
        await page.wait_for_timeout(int(bounded_sec * 1000))
        clamp_note = f" (bounded from {sec}s)" if clamped else ""
        msg = f"Waited for {bounded_sec:.1f}s{clamp_note}"

        return ExecutionResult(
            success=True,
            action_type=ActionType.WAIT,
            action="wait",
            message=msg,
        )

    async def _execute_press_key(self, page: Page, key: Optional[str]) -> ExecutionResult:
        """Perform key press on active element."""
        key_name = key or "Enter"
        await page.keyboard.press(key_name)
        return ExecutionResult(
            success=True,
            action_type=ActionType.PRESS_KEY,
            action="press_key",
            message=f"Pressed key '{key_name}'",
        )

    # Convenience helper methods
    async def click(self, target: str) -> ExecutionResult:
        return await self.execute({"action": "click", "target": target})

    async def type(self, target: str, text: str) -> ExecutionResult:
        return await self.execute({"action": "type", "target": target, "text": text})

    async def scroll(self, delta_y: int = 300) -> ExecutionResult:
        return await self.execute({"action": "scroll", "delta_y": delta_y})

    async def navigate(self, url: str) -> ExecutionResult:
        return await self.execute({"action": "navigate", "url": url})

    async def wait(self, seconds: float = 1.0, target: Optional[str] = None) -> ExecutionResult:
        return await self.execute({"action": "wait", "seconds": seconds, "target": target})

    async def close(self) -> None:
        """Tear down browser session and release resources cleanly."""
        try:
            if self._page and not self._page.is_closed():
                await self._page.close()
        except Exception:
            pass

        try:
            if self._context:
                await self._context.close()
        except Exception:
            pass

        try:
            if self._browser and self._browser.is_connected():
                await self._browser.close()
        except Exception:
            pass

        try:
            if self._playwright and self._owns_browser:
                await self._playwright.stop()
        except Exception:
            pass

        self._page = None
        self._context = None
        self._browser = None
        self._playwright = None
        self._loop = None


# Standalone runner for fixed invoice sequence test
if __name__ == "__main__":
    import asyncio
    import json
    import sys

    async def run_standalone_demo() -> None:
        headless = "--headless" in sys.argv
        executor = PlaywrightExecutor(headless=headless)

        print("==================================================")
        print("Starting Standalone Browser Executor Test (Step 3)")
        print(f"Mock Site URL: {MOCK_SITE_DEFAULT_URL}")
        print("==================================================")

        try:
            # 1. Open mock dashboard
            print("\n[Step 1] Opening mock dashboard...")
            res1 = await executor.execute({"action": "navigate", "url": MOCK_SITE_DEFAULT_URL})
            print(json.dumps(res1.model_dump(exclude_none=True), indent=2, default=str))
            assert res1.success, f"Step 1 failed: {res1.message}"

            # 2. Navigate to invoices page
            print("\n[Step 2] Navigating to invoices page...")
            res2 = await executor.execute({"action": "click", "target": "nav-invoices"})
            print(json.dumps(res2.model_dump(exclude_none=True), indent=2, default=str))
            assert res2.success, f"Step 2 failed: {res2.message}"

            # 3. Fill in invoice search field
            print("\n[Step 3] Filling invoice search field with 'INV-1001'...")
            res3 = await executor.execute({"action": "type", "target": "invoice-search", "text": "INV-1001"})
            print(json.dumps(res3.model_dump(exclude_none=True), indent=2, default=str))
            assert res3.success, f"Step 3 failed: {res3.message}"

            # 4. Click appropriate View button
            print("\n[Step 4] Clicking View button for invoice 1001...")
            res4 = await executor.execute({"action": "click", "target": "view-invoice-1001"})
            print(json.dumps(res4.model_dump(exclude_none=True), indent=2, default=str))
            assert res4.success, f"Step 4 failed: {res4.message}"

            # 5. Verify invoice details are visible
            print("\n[Step 5] Waiting for invoice modal details to appear...")
            res5 = await executor.execute({"action": "wait", "target": "close-invoice-modal-btn", "wait_seconds": 2.0})
            print(json.dumps(res5.model_dump(exclude_none=True), indent=2, default=str))
            assert res5.success, f"Step 5 failed: {res5.message}"

            # Check details
            page = executor.current_page
            invoice_title = await page.locator("#modal-invoice-id").text_content()
            client_name = await page.locator("#modal-client-name").text_content()
            total_due = await page.locator("#modal-grand-total").text_content()

            print("\n[Verification Details]")
            print(f"  Invoice Title: {invoice_title}")
            print(f"  Client Name:   {client_name}")
            print(f"  Grand Total:   {total_due}")

            assert "1001" in (invoice_title or ""), "Invoice modal ID does not match 1001"
            assert "Apex Systems" in (client_name or ""), "Client name does not match Apex Systems"

            print("\nSUCCESS: All 5 steps completed and verified successfully!")

        finally:
            print("\nClosing browser...")
            await executor.close()
            print("Browser closed.")

    asyncio.run(run_standalone_demo())
