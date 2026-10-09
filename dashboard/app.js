/**
 * BrowserPilot AI - Live Dashboard Client Logic
 * Step 5: Visual Control Panel for Browser Automation
 * 
 * Owned by: Member B (Browser Automation Engineer)
 */

// Dynamic API Origin Resolution
const API_BASE = (window.location.origin.includes('8000') || window.location.origin.includes('localhost') || window.location.origin.includes('127.0.0.1'))
  ? window.location.origin
  : 'http://127.0.0.1:8000';

const WS_URL = API_BASE.replace(/^http/, 'ws') + '/ws/events';

// Internal State
let isBackendConnected = false;
let currentExecutionState = 'IDLE'; // IDLE | RUNNING | COMPLETED | FAILED | STOPPED
let activePreset = 'full-workflow';
let abortRequested = false;
let processedEventIds = new Set();
let socket = null;
let healthPollTimer = null;
let eventsPollTimer = null;

// DOM Elements Cache
const elBackendStatusDot = document.getElementById('backend-status-dot');
const elBackendStatusText = document.getElementById('backend-status-text');
const elAgentStatusBadge = document.getElementById('agent-status-badge');
const elAgentStatusText = document.getElementById('agent-status-text');
const elGoalInput = document.getElementById('goal-input');
const elBtnRun = document.getElementById('btn-run');
const elBtnRunText = document.getElementById('btn-run-text');
const elBtnStop = document.getElementById('btn-stop');
const elBtnResume = document.getElementById('btn-resume');
const elBrowserTitle = document.getElementById('browser-title');
const elBrowserUrl = document.getElementById('browser-url');
const elActiveTargetBadge = document.getElementById('active-target-badge');
const elActiveTargetText = document.getElementById('active-target-text');
const elElementsCount = document.getElementById('elements-count');
const elElementsSummary = document.getElementById('elements-summary');
const elAgentTagsContainer = document.getElementById('agent-tags-container');
const elEventCountBadge = document.getElementById('event-count-badge');
const elAutoscrollCheckbox = document.getElementById('autoscroll-checkbox');
const elActivityFeedList = document.getElementById('activity-feed-list');
const elFeedEmptyState = document.getElementById('feed-empty-state');

// Custom Single Action Drawer Elements
const elCustomType = document.getElementById('custom-action-type');
const elCustomTarget = document.getElementById('custom-action-target');
const elCustomValue = document.getElementById('custom-action-value');

// ============================================================================
// 1. Backend Connectivity & Health Check
// ============================================================================

async function checkBackendHealth() {
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 2000);
    const res = await fetch(`${API_BASE}/health`, { signal: controller.signal });
    clearTimeout(timeoutId);

    if (res.ok) {
      const data = await res.json();
      if (data.status === 'ok') {
        updateBackendStatus(true);
        return true;
      }
    }
    updateBackendStatus(false);
    return false;
  } catch (err) {
    updateBackendStatus(false);
    return false;
  }
}

function updateBackendStatus(connected) {
  isBackendConnected = connected;
  if (!elBackendStatusDot || !elBackendStatusText) return;

  if (connected) {
    elBackendStatusDot.className = 'status-dot dot-online';
    elBackendStatusText.textContent = 'ONLINE';
    elBackendStatusText.style.color = 'var(--success)';
  } else {
    elBackendStatusDot.className = 'status-dot dot-offline';
    elBackendStatusText.textContent = 'OFFLINE';
    elBackendStatusText.style.color = 'var(--danger)';
  }
}

// ============================================================================
// 2. Browser State & Observation (GET /observe)
// ============================================================================

async function refreshObservation() {
  if (!isBackendConnected) return;

  try {
    const res = await fetch(`${API_BASE}/observe`);
    if (!res.ok) return;

    const data = await res.json();
    if (elBrowserTitle) elBrowserTitle.textContent = data.title || '(No Title)';
    if (elBrowserUrl) elBrowserUrl.textContent = data.url || 'about:blank';

    const elements = data.elements || [];
    if (elElementsCount) elElementsCount.textContent = `${elements.length} Elements`;
    if (elElementsSummary) elElementsSummary.textContent = `via Playwright Observer`;

    renderInteractiveTagCloud(elements);
  } catch (err) {
    console.warn('Failed to refresh browser observation:', err);
  }
}

function renderInteractiveTagCloud(elements) {
  if (!elAgentTagsContainer) return;

  if (!elements || elements.length === 0) {
    elAgentTagsContainer.innerHTML = '<span class="tag-placeholder">No interactive elements with data-agent-id visible</span>';
    return;
  }

  elAgentTagsContainer.innerHTML = '';
  elements.forEach((item) => {
    const agentId = typeof item === 'string' ? item : item.agent_id;
    if (!agentId) return;

    const tag = document.createElement('span');
    tag.className = 'tag-pill';
    tag.textContent = agentId;
    tag.title = `Click to set as target: ${agentId}`;
    tag.onclick = () => {
      if (elCustomTarget) elCustomTarget.value = agentId;
      setActiveTargetHighlight(agentId);
    };
    elAgentTagsContainer.appendChild(tag);
  });
}

function setActiveTargetHighlight(targetId) {
  if (!elActiveTargetText) return;
  elActiveTargetText.textContent = targetId || 'None';
  if (elActiveTargetBadge) {
    elActiveTargetBadge.style.borderColor = targetId ? 'var(--purple)' : 'rgba(139, 92, 246, 0.35)';
  }
}

// ============================================================================
// 3. Execution Lifecycle Status Feedback
// ============================================================================

function setExecutionState(state, details = '') {
  currentExecutionState = state;
  if (!elAgentStatusBadge || !elAgentStatusText) return;

  elAgentStatusText.textContent = state;
  elAgentStatusBadge.className = `badge status-${state.toLowerCase()}`;

  if (state === 'RUNNING') {
    elBtnRun.disabled = true;
    elBtnStop.disabled = false;
    if (elBtnResume) elBtnResume.style.display = 'none';
    if (elBtnRunText) elBtnRunText.textContent = 'Running...';
  } else if (state === 'STOPPED') {
    elBtnRun.disabled = false;
    elBtnStop.disabled = true;
    if (elBtnResume) elBtnResume.style.display = 'inline-flex';
    if (elBtnRunText) elBtnRunText.textContent = 'Run Task';
  } else {
    // IDLE, COMPLETED, FAILED
    elBtnRun.disabled = false;
    elBtnStop.disabled = false;
    if (elBtnResume) elBtnResume.style.display = 'none';
    if (elBtnRunText) elBtnRunText.textContent = 'Run Task';
  }
}

// ============================================================================
// 4. Activity Feed & Event Telemetry (GET /events & WebSocket)
// ============================================================================

async function fetchRecentEvents() {
  if (!isBackendConnected) return;

  try {
    const res = await fetch(`${API_BASE}/events?limit=30`);
    if (!res.ok) return;

    const events = await res.json();
    if (Array.isArray(events)) {
      events.forEach((ev) => ingestEvent(ev));
    }
  } catch (err) {
    // Polling error silently suppressed to avoid console spam
  }
}

function ingestEvent(ev) {
  const evKey = ev.event_id || `${ev.timestamp}_${ev.action}_${ev.target}`;
  if (processedEventIds.has(evKey)) return;
  processedEventIds.add(evKey);

  renderEventItem(ev);
}

function renderEventItem(ev) {
  if (elFeedEmptyState) {
    elFeedEmptyState.style.display = 'none';
  }

  const isSuccess = ev.success === true || ev.status === 'success';
  const actionName = (ev.action || ev.type || 'ACTION').toUpperCase();
  const targetName = ev.target || (ev.data && ev.data.target) || null;
  const message = ev.message || (isSuccess ? 'Action completed successfully' : 'Action failed');
  const errorMsg = ev.error || (ev.data && ev.data.error) || null;
  const timestamp = ev.timestamp ? new Date(ev.timestamp).toLocaleTimeString() : new Date().toLocaleTimeString();

  if (targetName) {
    setActiveTargetHighlight(targetName);
  }

  const item = document.createElement('div');
  item.className = `feed-item ${isSuccess ? 'feed-item-success' : 'feed-item-error'}`;

  let targetHtml = targetName ? `<span class="feed-target-pill">🎯 ${escapeHtml(targetName)}</span>` : '';
  let errorHtml = (!isSuccess && errorMsg) ? `<div class="feed-error-box">⚠️ Error: ${escapeHtml(errorMsg)}</div>` : '';

  item.innerHTML = `
    <div class="feed-item-header">
      <div class="feed-header-left">
        <span class="feed-action-badge ${isSuccess ? 'badge-act-success' : 'badge-act-failure'}">
          ${isSuccess ? '✓ ' : '✗ '}${escapeHtml(actionName)}
        </span>
        ${targetHtml}
      </div>
      <span class="feed-time">${timestamp}</span>
    </div>
    <div class="feed-item-body">
      ${escapeHtml(message)}
      ${errorHtml}
    </div>
  `;

  // Prepend to top of activity feed for immediate visibility
  elActivityFeedList.insertBefore(item, elActivityFeedList.firstChild);

  // Update badge counter
  if (elEventCountBadge) {
    const totalItems = elActivityFeedList.querySelectorAll('.feed-item').length;
    elEventCountBadge.textContent = `${totalItems} events`;
  }

  // Auto-scroll logic if enabled
  if (elAutoscrollCheckbox && elAutoscrollCheckbox.checked) {
    elActivityFeedList.scrollTop = 0;
  }
}

function clearActivityFeed() {
  if (!elActivityFeedList) return;
  elActivityFeedList.innerHTML = `
    <div class="feed-empty-state" id="feed-empty-state">
      <div class="empty-icon">📡</div>
      <p>Awaiting browser events...</p>
      <span style="font-size: 0.8rem; color: var(--text-muted);">Real-time action executions will appear here.</span>
    </div>
  `;
  processedEventIds.clear();
  if (elEventCountBadge) elEventCountBadge.textContent = '0 events';
  setActiveTargetHighlight(null);
}

function connectWebSocket() {
  try {
    socket = new WebSocket(WS_URL);

    socket.onopen = () => {
      console.log('Telemetry WebSocket connected.');
    };

    socket.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data);
        if (payload.event_id || payload.type) {
          ingestEvent({
            event_id: payload.event_id,
            timestamp: payload.timestamp,
            action: payload.data?.action || payload.type,
            target: payload.data?.target,
            status: payload.data?.status || (payload.data?.success ? 'success' : 'failure'),
            success: payload.data?.success ?? true,
            message: payload.message,
            error: payload.data?.error,
            data: payload.data,
          });
        }
      } catch (e) {
        console.warn('Failed to parse WS payload:', e);
      }
    };

    socket.onclose = () => {
      // Reconnect after 3 seconds
      setTimeout(connectWebSocket, 3000);
    };

    socket.onerror = () => {
      // WebSocket fallback handled by periodic HTTP polling
    };
  } catch (err) {
    console.warn('WebSocket connection not supported, relying on HTTP polling.');
  }
}

// ============================================================================
// 5. Presets & Task Execution Engine
// ============================================================================

const PRESETS = {
  'full-workflow': {
    title: 'Verify invoice 1001 details on mock site',
    description: 'Full end-to-end workflow: open invoices, search INV-1001, view details, verify amount',
    steps: [
      { action: 'navigate', url: 'index.html#invoices' },
      { action: 'type', target: 'invoice-search', text: 'INV-1001' },
      { action: 'click', target: 'view-invoice-1001' },
      { action: 'wait', target: 'invoice-detail-amount', wait_seconds: 2 },
    ],
  },
  'nav-invoices': {
    title: 'Navigate to invoices page',
    description: 'Open the mock invoices listing',
    steps: [
      { action: 'navigate', url: 'index.html#invoices' },
    ],
  },
  'type-search': {
    title: 'Search for invoice INV-1001',
    description: 'Filter invoice table with text INV-1001',
    steps: [
      { action: 'type', target: 'invoice-search', text: 'INV-1001' },
    ],
  },
  'click-view': {
    title: 'Click View button for invoice 1001',
    description: 'Expand the invoice details modal/panel',
    steps: [
      { action: 'click', target: 'view-invoice-1001' },
    ],
  },
  'scroll-down': {
    title: 'Scroll invoices table',
    description: 'Scroll vertically to reveal lower invoices',
    steps: [
      { action: 'scroll', scroll_delta_y: 400 },
    ],
  },
  'invalid-target': {
    title: 'Error Demo: Click non-existent target',
    description: 'Demonstrate safeguards rejecting missing target element',
    steps: [
      { action: 'click', target: 'non-existent-button-999' },
    ],
  },
};

function selectPreset(presetKey) {
  activePreset = presetKey;
  const config = PRESETS[presetKey];
  if (!config) return;

  if (elGoalInput) {
    elGoalInput.value = config.title;
  }

  // Update active button state
  document.querySelectorAll('.preset-btn').forEach((btn) => {
    btn.classList.remove('active');
  });

  const matchingBtn = Array.from(document.querySelectorAll('.preset-btn')).find((b) =>
    b.getAttribute('onclick')?.includes(`'${presetKey}'`)
  );
  if (matchingBtn) matchingBtn.classList.add('active');

  // If single action step, pre-populate custom drawer
  if (config.steps.length === 1) {
    const s = config.steps[0];
    if (elCustomType) elCustomType.value = s.action;
    if (elCustomTarget) elCustomTarget.value = s.target || '';
    if (elCustomValue) elCustomValue.value = s.text || s.url || s.scroll_delta_y || '';
  }
}

/**
 * Execute real action payload through POST /execute
 */
async function executeActionRequest(payload) {
  const res = await fetch(`${API_BASE}/execute`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    const errorText = await res.text();
    return {
      success: false,
      action: payload.action,
      target: payload.target,
      message: `HTTP ${res.status}: ${errorText}`,
      error: `Server Error ${res.status}`,
    };
  }

  return await res.json();
}

/**
 * Main Run Task Handler
 * Executes real actions sequentially against the FastAPI backend
 */
async function runCurrentTask() {
  if (!isBackendConnected) {
    const reconnected = await checkBackendHealth();
    if (!reconnected) {
      setExecutionState('FAILED', 'FastAPI backend is OFFLINE. Check port 8000.');
      return;
    }
  }

  abortRequested = false;
  setExecutionState('RUNNING');
  console.log('Task run initiated for preset:', activePreset);

  const presetConfig = PRESETS[activePreset] || PRESETS['full-workflow'];
  const steps = presetConfig.steps;

  let allSuccess = true;
  let finalMessage = '';

  for (let i = 0; i < steps.length; i++) {
    if (abortRequested) {
      setExecutionState('STOPPED');
      console.log('Task halted by stop signal.');
      return;
    }

    const step = steps[i];
    if (step.target) {
      setActiveTargetHighlight(step.target);
    }

    try {
      const result = await executeActionRequest(step);
      console.log('Action executed:', step.action, 'Success:', result.success, result.message);

      // Immediately fetch events & refresh observation in background
      fetchRecentEvents().catch(console.warn);
      refreshObservation().catch(console.warn);

      if (!result.success) {
        allSuccess = false;
        finalMessage = result.message || result.error || 'Action failed';
        setExecutionState('FAILED', finalMessage);
        console.warn('Task step failed:', finalMessage);
        return;
      }
    } catch (err) {
      allSuccess = false;
      setExecutionState('FAILED', err.message);
      console.error('Task execution exception:', err);
      return;
    }

    // Brief realistic pause between actions if multi-step
    if (i < steps.length - 1 && !abortRequested) {
      await new Promise((r) => setTimeout(r, 400));
    }
  }

  if (allSuccess) {
    setExecutionState('COMPLETED');
    console.log('Task successfully completed.');
  }
}

/**
 * Stop Execution Handler
 * Commands POST /stop to set the backend stop flag and halt actions
 */
async function stopCurrentTask() {
  abortRequested = true;
  setExecutionState('STOPPED');

  try {
    const res = await fetch(`${API_BASE}/stop`, { method: 'POST' });
    if (res.ok) {
      console.log('Stop signal confirmed by backend.');
    }
    await fetchRecentEvents();
    await refreshObservation();
  } catch (err) {
    console.error('Failed to dispatch stop request:', err);
  }
}

/**
 * Resume Execution Handler
 * Commands POST /resume to reset backend stop flag
 */
async function resumeAgent() {
  abortRequested = false;
  try {
    const res = await fetch(`${API_BASE}/resume`, { method: 'POST' });
    if (res.ok) {
      console.log('Backend execution resumed.');
      setExecutionState('IDLE');
    }
    await fetchRecentEvents();
  } catch (err) {
    console.error('Failed to dispatch resume request:', err);
  }
}

/**
 * Custom Atomic Action Executor from drawer
 */
async function executeCustomSingleAction() {
  if (!isBackendConnected) {
    const reconnected = await checkBackendHealth();
    if (!reconnected) {
      setExecutionState('FAILED', 'Backend is offline.');
      return;
    }
  }

  const action = elCustomType.value;
  const target = elCustomTarget.value.trim() || undefined;
  const rawVal = elCustomValue.value.trim();

  const payload = { action, target };

  if (action === 'type') {
    payload.text = rawVal || 'INV-1001';
  } else if (action === 'navigate') {
    payload.url = rawVal || 'invoices.html';
  } else if (action === 'scroll') {
    payload.scroll_delta_y = parseInt(rawVal, 10) || 300;
  } else if (action === 'wait') {
    payload.wait_seconds = parseFloat(rawVal) || 2.0;
  }

  setExecutionState('RUNNING');
  if (target) setActiveTargetHighlight(target);

  try {
    const result = await executeActionRequest(payload);
    await fetchRecentEvents();
    await refreshObservation();

    if (result.success) {
      setExecutionState('COMPLETED');
    } else {
      setExecutionState('FAILED', result.message);
    }
  } catch (err) {
    setExecutionState('FAILED', err.message);
  }
}

// ============================================================================
// 6. Utility Functions
// ============================================================================

function escapeHtml(text) {
  if (!text) return '';
  const div = document.createElement('div');
  div.textContent = String(text);
  return div.innerHTML;
}

// ============================================================================
// 7. Initialization & Lifecycle Loop
// ============================================================================

async function initDashboard() {
  // 1. Initial health check
  const connected = await checkBackendHealth();
  if (connected) {
    await refreshObservation();
    await fetchRecentEvents();
  }

  // 2. Setup periodic health & event polling
  healthPollTimer = setInterval(async () => {
    const wasConnected = isBackendConnected;
    const nowConnected = await checkBackendHealth();
    if (!wasConnected && nowConnected) {
      // Reconnection: refresh observation
      refreshObservation();
    }
  }, 2500);

  eventsPollTimer = setInterval(async () => {
    if (isBackendConnected) {
      await fetchRecentEvents();
    }
  }, 1200);

  // 3. Connect WebSocket for live streaming
  connectWebSocket();

  // 4. Default preset selection
  selectPreset('full-workflow');
}

window.addEventListener('DOMContentLoaded', initDashboard);
