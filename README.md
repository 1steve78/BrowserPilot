# BrowserPilot AI

> **A local-first, privacy-respecting autonomous browser agent powered by Ollama (Gemma 3 4B), Playwright, and FastAPI.**

BrowserPilot AI executes end-to-end web navigation tasks autonomously on your machine without routing sensitive browser telemetry or credentials to cloud API providers.

---

## 🌟 Key Capabilities (Milestone 1 Scaffold)

- **Local-First AI Brain**: Runs locally via [Ollama](https://ollama.com/) with Google's `gemma3:4b` model.
- **Robust Browser Automation**: Direct Chromium control and DOM observation powered by Playwright.
- **Structured Action Contracts**: Pydantic v2 schemas validating every cognitive thought and browser action.
- **Proactive Safety Guardrails**: Hardened against destructive operations (`delete`, `purge`) and indirect prompt injection attacks.
- **Live Telemetry Dashboard**: Dark-mode UI with live WebSocket event streaming, thought inspection, and emergency stop controls.
- **Built-in Mock Test Arena**: Includes realistic task tables, invoice documents, destructive targets, and prompt injection sandboxes.

---

## 👥 Four-Person Team Ownership

This project is organized into modular boundaries so engineering, quality, and product streams can build in parallel:

| Team Role | Primary Ownership & Focus Area | Key Modules |
| :--- | :--- | :--- |
| **AI Engineer** | Model prompting, Ollama client integration, structured JSON validation, few-shot examples, and reasoning loops. | `backend/app/model_client.py`<br>`backend/app/agent_loop.py` |
| **Browser Engineer** | Playwright browser lifecycle, DOM inspection, accessibility tree mapping, element interaction heuristics. | `backend/app/observer.py`<br>`backend/app/executor.py` |
| **QA & Security Engineer** | Test harness, unit/integration suites, prompt-injection defense verification, destructive action guardrails. | `backend/app/safety.py`<br>`backend/tests/`<br>`docs/test-plan.md` |
| **Product & Presentation Lead** | Dashboard interface UX, mock site scenarios, demonstration scripts, presentation assets. | `dashboard/`<br>`mock-site/`<br>`docs/demo-script.md` |

---

## 📂 Project Structure

```text
browserpilot-ai/
├── backend/
│   ├── app/
│   │   ├── __init__.py           # Package version
│   │   ├── main.py               # FastAPI server & WebSocket routes
│   │   ├── schemas.py            # Pydantic v2 data models
│   │   ├── model_client.py       # Ollama gemma3:4b client & prompts
│   │   ├── agent_loop.py         # Autonomous coordinator loop
│   │   ├── observer.py           # Playwright DOM inspection
│   │   ├── executor.py           # Playwright browser automation
│   │   ├── safety.py             # Safety & injection guardrails
│   │   └── events.py             # Event pub/sub & WebSocket broadcaster
│   └── tests/
│       ├── __init__.py
│       └── test_schemas.py       # Unit tests for schemas & safety
├── mock-site/
│   ├── index.html                # Overview, invoice data, destructive target
│   ├── tasks.html                # Searchable task table with actions
│   ├── injection.html            # Adversarial prompt-injection test page
│   └── styles.css                # Mock site styling
├── dashboard/
│   ├── index.html                # Dark-mode telemetry dashboard
│   ├── app.js                    # WebSocket client & API caller
│   └── styles.css                # Modern dark dashboard styling
├── extension/
│   └── README.md                 # Optional Chrome extension architecture
├── docs/
│   ├── architecture.md           # System architecture & data flow
│   ├── test-plan.md              # Test matrix and security verification
│   └── demo-script.md            # Live walkthrough presentation script
├── .env.example                  # Environment configuration template
├── .gitignore                    # Git ignore rules
├── requirements.txt              # Python dependencies (Python 3.11+)
└── README.md                     # Project documentation & setup guide
```

---

## 🛠️ Prerequisites

Before getting started, make sure you have:
1. **Windows 10/11** with **PowerShell**
2. **Python 3.11+** installed (`python --version`)
3. **Ollama** installed from [ollama.com](https://ollama.com/)
   - Pull the model: `ollama pull gemma3:4b`

---

## 🚀 Quickstart Guide (Windows PowerShell)

Follow these exact commands to set up the environment and launch all services:

### 1. Create and Activate Virtual Environment

```powershell
# Open PowerShell in the project directory
cd d:\projects\Browser-agent

# Create virtual environment
python -m venv .venv

# Activate virtual environment
.\.venv\Scripts\Activate.ps1
```

### 2. Install Dependencies & Playwright Browser

```powershell
# Upgrade pip
python -m pip install --upgrade pip

# Install project dependencies
pip install -r requirements.txt

# Install Playwright Chromium binary
playwright install chromium
```

### 3. Configure Environment Variables

```powershell
# Copy the example environment file
Copy-Item .env.example .env
```

### 4. Run Unit Tests

Verify that schemas, safety checks, and API structures are sound:

```powershell
pytest backend/tests -v
```

---

## 🖥️ Running the Application

### Step 1: Start the Mock Site (Terminal 1)
Run a local static server for the test target pages:

```powershell
# In PowerShell Terminal 1
python -m http.server 8080 --directory mock-site
```
> The mock site will be available at: **http://localhost:8080/**

### Step 2: Start the FastAPI Backend & Dashboard (Terminal 2)

```powershell
# In PowerShell Terminal 2 (with .venv active)
uvicorn backend.app.main:app --reload --port 8000
```
> - Backend REST API & Docs: **http://localhost:8000/docs**
> - Live Dashboard: **http://localhost:8000/dashboard/**
> - Mock Site (also mounted at): **http://localhost:8000/mock/**

### Step 3: Start Ollama (Terminal 3)

```powershell
# In PowerShell Terminal 3 (if not already running as a Windows service)
ollama run gemma3:4b
```

---

## 🔒 Security & Privacy

- All LLM inference happens **strictly on localhost** via Ollama.
- No network telemetry or page screenshots leave your workstation.
- Destructive browser actions are caught by `SafetyGuard` prior to execution.
- Web content is inspected for prompt injection attempts before model consumption.

---

## 🧭 Milestone Roadmap

- [x] **Milestone 1**: Clean, modular monorepo scaffold with schemas, mock site, dashboard, and test suites.
- [ ] **Milestone 2**: Parallel implementation of core AI brain prompts and Playwright accessibility tree walker.
- [ ] **Milestone 3**: End-to-end integration with live Ollama inference and automated QA test runs.
- [ ] **Milestone 4 (Future)**: Optional Manifest V3 Chrome extension bridge.
