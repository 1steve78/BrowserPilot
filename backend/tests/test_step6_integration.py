"""Step 6 End-to-End Integration & Stabilization Test Suite.

Validates the full system architecture under Member B's scope:
1. Test A: Browser connection (Observer returns title, URL, interactive elements)
2. Test B: Action execution (Search INV-1001, view details, verified in DOM)
3. Test C: Dashboard synchronization (Real backend events, URL/title updates, status)
4. Test D: Stop & failure handling (Stop blocks actions, invalid targets produce errors, timeouts don't crash API)
5. Test E: Repeated runs idempotence (No stale browser state poisoning subsequent runs)
"""

import asyncio
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, List

import httpx
from playwright.async_api import async_playwright
import pytest


@pytest.fixture(scope="module")
def uvicorn_server():
    """Start uvicorn server in a subprocess on port 8000 for Step 6 integration tests."""
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.app.main:app", "--port", "8000"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    ready = False
    for _ in range(40):
        try:
            r = httpx.get("http://127.0.0.1:8000/health", timeout=2.0)
            if r.status_code == 200:
                ready = True
                break
        except Exception:
            time.sleep(0.25)

    if not ready:
        proc.kill()
        raise RuntimeError("Uvicorn server failed to start within timeout.")

    yield "http://127.0.0.1:8000"

    # Teardown
    try:
        httpx.post("http://127.0.0.1:8000/resume", timeout=2.0)
    except Exception:
        pass
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


@pytest.mark.asyncio
async def test_a_browser_connection_and_observation(uvicorn_server):
    """Test A: Confirm browser connects and observer returns correct page title, URL, and elements."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        # 1. Health check
        health_resp = await client.get(f"{uvicorn_server}/health")
        assert health_resp.status_code == 200
        assert health_resp.json() == {"status": "ok"}

        # 2. Reset stop flag
        await client.post(f"{uvicorn_server}/resume")

        # 3. Navigate to mock dashboard
        nav_resp = await client.post(
            f"{uvicorn_server}/execute",
            json={"action": "navigate", "url": "index.html"},
        )
        assert nav_resp.status_code == 200
        assert nav_resp.json()["success"] is True

        # 4. Observe page state
        obs_resp = await client.get(f"{uvicorn_server}/observe")
        assert obs_resp.status_code == 200
        data = obs_resp.json()

        assert "index.html" in data["url"]
        assert len(data["title"]) > 0
        assert "ApexFlow" in data["title"] or "Dashboard" in data["title"]
        assert isinstance(data["elements"], list)
        assert len(data["elements"]) > 0

        # Verify observer extracts data-agent-id elements
        agent_ids = [el.get("agent_id") for el in data["elements"] if el.get("agent_id")]
        assert "nav-invoices" in agent_ids or "global-search-input" in agent_ids


@pytest.mark.asyncio
async def test_b_action_execution_invoice_workflow(uvicorn_server):
    """Test B: Search for invoice INV-1001, open details, confirm details are visible."""
    async with httpx.AsyncClient(timeout=15.0) as client:
        await client.post(f"{uvicorn_server}/resume")

        # 1. Navigate to invoices
        r1 = await client.post(
            f"{uvicorn_server}/execute",
            json={"action": "navigate", "url": "index.html#invoices"},
        )
        assert r1.status_code == 200
        assert r1.json()["success"] is True

        # 2. Type search INV-1001
        r2 = await client.post(
            f"{uvicorn_server}/execute",
            json={"action": "type", "target": "invoice-search", "text": "INV-1001"},
        )
        assert r2.status_code == 200
        assert r2.json()["success"] is True
        assert "INV-1001" in r2.json()["message"]

        # 3. Click View button
        r3 = await client.post(
            f"{uvicorn_server}/execute",
            json={"action": "click", "target": "view-invoice-1001"},
        )
        assert r3.status_code == 200
        assert r3.json()["success"] is True
        assert "Invoice details opened" in r3.json()["message"]

        # 4. Wait for invoice details modal to appear
        r4 = await client.post(
            f"{uvicorn_server}/execute",
            json={"action": "wait", "target": "close-invoice-modal-btn", "seconds": 2.0},
        )
        assert r4.status_code == 200
        assert r4.json()["success"] is True
        assert "appeared and is visible" in r4.json()["message"]

        # 5. Observe that modal details are in the active DOM
        obs = await client.get(f"{uvicorn_server}/observe")
        assert obs.status_code == 200
        obs_data = obs.json()
        agent_ids = [el.get("agent_id") for el in obs_data["elements"] if el.get("agent_id")]
        assert "close-invoice-modal-btn" in agent_ids


@pytest.mark.asyncio
async def test_c_dashboard_synchronization(uvicorn_server):
    """Test C: Verify dashboard displays real events, updates URL/title, and reflects execution state."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        dashboard_url = f"{uvicorn_server}/dashboard/index.html"
        await page.goto(dashboard_url, wait_until="domcontentloaded")
        await page.wait_for_selector("#backend-status-text:has-text('ONLINE')", timeout=8000)

        # Trigger action via API directly to simulate agent/backend activity
        async with httpx.AsyncClient(timeout=10.0) as client:
            await client.post(
                f"{uvicorn_server}/execute",
                json={"action": "scroll", "delta_y": 250},
            )

        # Dashboard activity feed must render the real event
        await page.wait_for_timeout(1500)
        feed_items = await page.locator(".feed-item").all()
        assert len(feed_items) > 0, "Dashboard must synchronize events from backend"

        # Check browser state indicators
        url_text = await page.inner_text("#browser-url")
        title_text = await page.inner_text("#browser-title")
        assert len(url_text) > 0 and url_text != "about:blank"
        assert len(title_text) > 0 and title_text != "Checking page..."

        await browser.close()


@pytest.mark.asyncio
async def test_d_stop_and_failure_handling(uvicorn_server):
    """Test D: Stop blocks subsequent actions, invalid target errors cleanly, timeouts don't crash server."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        # Part 1: Stop prevents next action
        stop_resp = await client.post(f"{uvicorn_server}/stop")
        assert stop_resp.status_code == 200
        assert stop_resp.json()["status"] == "stopped"

        blocked_resp = await client.post(
            f"{uvicorn_server}/execute",
            json={"action": "scroll", "delta_y": 100},
        )
        assert blocked_resp.status_code == 200
        assert blocked_resp.json()["success"] is False
        assert "stopped" in blocked_resp.json()["message"].lower()

        # Part 2: Resume and test nonexistent target error
        await client.post(f"{uvicorn_server}/resume")
        missing_resp = await client.post(
            f"{uvicorn_server}/execute",
            json={"action": "click", "target": "target-that-does-not-exist-xyz-999"},
        )
        assert missing_resp.status_code == 200
        assert missing_resp.json()["success"] is False
        assert "not found" in missing_resp.json()["message"].lower()

        # Part 3: Element wait timeout does not crash the server
        timeout_wait_resp = await client.post(
            f"{uvicorn_server}/execute",
            json={"action": "wait", "target": "phantom-element-never-appears", "seconds": 0.5},
        )
        assert timeout_wait_resp.status_code == 200
        assert timeout_wait_resp.json()["success"] is False
        assert "timed out" in timeout_wait_resp.json()["message"].lower()

        # Server must still be fully responsive
        health_check = await client.get(f"{uvicorn_server}/health")
        assert health_check.status_code == 200
        assert health_check.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_e_repeated_runs_do_not_leave_stale_state(uvicorn_server):
    """Test E: Repeated workflow runs succeed cleanly without stale browser state poisoning."""
    async with httpx.AsyncClient(timeout=20.0) as client:
        await client.post(f"{uvicorn_server}/resume")

        for run_idx in range(1, 3):
            # 1. Reset / Navigate to fresh invoices listing
            nav = await client.post(
                f"{uvicorn_server}/execute",
                json={"action": "navigate", "url": "index.html#invoices"},
            )
            assert nav.json()["success"] is True

            # 2. Type search filter
            search = await client.post(
                f"{uvicorn_server}/execute",
                json={"action": "type", "target": "invoice-search", "text": "INV-1001"},
            )
            assert search.json()["success"] is True

            # 3. View invoice
            view = await client.post(
                f"{uvicorn_server}/execute",
                json={"action": "click", "target": "view-invoice-1001"},
            )
            assert view.json()["success"] is True

            # 4. Wait for modal
            wait_modal = await client.post(
                f"{uvicorn_server}/execute",
                json={"action": "wait", "target": "close-invoice-modal-btn", "seconds": 2.0},
            )
            assert wait_modal.json()["success"] is True

            # 5. Close modal to cleanly finish cycle
            close_modal = await client.post(
                f"{uvicorn_server}/execute",
                json={"action": "click", "target": "close-invoice-modal-btn"},
            )
            assert close_modal.json()["success"] is True
