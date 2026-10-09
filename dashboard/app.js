// BrowserPilot AI Dashboard Client Logic

const API_BASE = window.location.origin.includes('8000')
  ? window.location.origin
  : 'http://localhost:8000';

const WS_URL = API_BASE.replace(/^http/, 'ws') + '/ws/events';

let socket = null;

// DOM Elements
const goalInput = document.getElementById('goal-input');
const urlInput = document.getElementById('url-input');
const btnStart = document.getElementById('btn-start');
const btnStop = document.getElementById('btn-stop');
const agentStatusBadge = document.getElementById('agent-status-badge');
const agentStatusText = document.getElementById('agent-status-text');
const wsStatusBadge = document.getElementById('ws-status-badge');
const wsStatusText = document.getElementById('ws-status-text');
const thoughtReflection = document.getElementById('thought-reflection');
const thoughtReasoning = document.getElementById('thought-reasoning');
const thoughtAction = document.getElementById('thought-action');
const eventTimeline = document.getElementById('event-timeline');

// Initialize WebSocket
function connectWebSocket() {
  socket = new WebSocket(WS_URL);

  socket.onopen = () => {
    wsStatusBadge.className = 'badge status-completed';
    wsStatusText.textContent = 'CONNECTED';
  };

  socket.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      handleAgentEvent(data);
    } catch (e) {
      console.error('Failed to parse WebSocket message', e);
    }
  };

  socket.onclose = () => {
    wsStatusBadge.className = 'badge status-idle';
    wsStatusText.textContent = 'DISCONNECTED';
    setTimeout(connectWebSocket, 3000);
  };

  socket.onerror = (err) => {
    console.warn('WebSocket error, retrying...', err);
  };
}

function handleAgentEvent(event) {
  addTimelineItem(event.type, event.message, event.timestamp);

  if (event.type === 'status_change') {
    updateRunStatus(event.data?.status || 'RUNNING');
  } else if (event.type === 'action_proposed') {
    if (event.data?.thought) {
      thoughtReflection.textContent = event.data.thought.reflection || '-';
      thoughtReasoning.textContent = event.data.thought.reasoning || '-';
    }
    if (event.data?.action) {
      thoughtAction.textContent = `${event.data.action.action_type.toUpperCase()}: ${event.data.action.description}`;
    }
  } else if (event.type === 'run_finished') {
    updateRunStatus(event.data?.status || 'IDLE');
  }
}

function updateRunStatus(status) {
  const normalized = (status || '').toLowerCase();
  agentStatusText.textContent = normalized.toUpperCase();

  if (normalized === 'running') {
    agentStatusBadge.className = 'badge status-running';
    btnStart.disabled = true;
    btnStop.disabled = false;
  } else if (normalized === 'completed') {
    agentStatusBadge.className = 'badge status-completed';
    btnStart.disabled = false;
    btnStop.disabled = true;
  } else if (normalized === 'failed') {
    agentStatusBadge.className = 'badge status-failed';
    btnStart.disabled = false;
    btnStop.disabled = true;
  } else {
    agentStatusBadge.className = 'badge status-idle';
    btnStart.disabled = false;
    btnStop.disabled = true;
  }
}

function addTimelineItem(type, message, timestamp) {
  const item = document.createElement('div');
  item.className = 'timeline-item';

  let borderColor = 'var(--primary)';
  if (type === 'safety_alert' || type === 'error') borderColor = 'var(--danger)';
  if (type === 'action_executed') borderColor = 'var(--success)';
  if (type === 'agent_thinking') borderColor = 'var(--purple)';
  item.style.borderLeftColor = borderColor;

  const timeStr = timestamp ? new Date(timestamp).toLocaleTimeString() : new Date().toLocaleTimeString();

  item.innerHTML = `
    <div class="timeline-header">
      <span class="timeline-type" style="color: ${borderColor}">${type.replace('_', ' ')}</span>
      <span class="timeline-time">${timeStr}</span>
    </div>
    <div class="timeline-msg">${escapeHtml(message)}</div>
  `;

  eventTimeline.prepend(item);
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text || '';
  return div.innerHTML;
}

function clearTimeline() {
  eventTimeline.innerHTML = '';
}

function setPreset(goal, url) {
  goalInput.value = goal;
  urlInput.value = url;
}

async function startAgent() {
  const goal = goalInput.value.trim();
  const url = urlInput.value.trim();
  if (!goal) {
    alert('Please enter a goal for the agent.');
    return;
  }

  btnStart.disabled = true;
  try {
    const res = await fetch(`${API_BASE}/api/agent/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ goal, start_url: url, max_steps: 15 }),
    });
    const data = await res.json();
    if (data.error) {
      alert(data.error);
      btnStart.disabled = false;
    } else {
      updateRunStatus('RUNNING');
    }
  } catch (e) {
    alert('Failed to connect to backend API: ' + e.message);
    btnStart.disabled = false;
  }
}

async function stopAgent() {
  btnStop.disabled = true;
  try {
    await fetch(`${API_BASE}/api/agent/stop`, { method: 'POST' });
  } catch (e) {
    console.error('Failed to stop agent', e);
  }
}

// Auto-start connection on page load
window.addEventListener('DOMContentLoaded', connectWebSocket);
