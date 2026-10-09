# Chrome Extension (Optional / Future Milestone)

## Architectural Role

In **Milestone 1**, BrowserPilot AI relies exclusively on **Playwright (Chromium)** for browser control and page observation. Playwright provides:
- Direct CDP (Chrome DevTools Protocol) integration.
- Deterministic, scriptable headless and headful session management.
- Zero client installation or browser-extension store approval friction.

## Future Phase Integration (Manifest V3)

This directory is reserved for an optional Chrome extension designed to allow BrowserPilot AI to interact with the user's primary authenticated browser session.

### Proposed Architecture:
1. **Background Service Worker (`background.js`)**: Connects to the local FastAPI backend via `ws://localhost:8000/ws/extension`.
2. **Content Script (`content.js`)**: Injects DOM observers, extracts accessibility nodes, and receives execution commands (`click`, `fill`, `scroll`) to execute within the active user tab.
3. **Popup Interface (`popup.html`)**: Lightweight quick-launch trigger for user goals directly from the Chrome toolbar.

### TODO Items (Phase 2):
- [ ] Define Manifest V3 `manifest.json` with permissions (`activeTab`, `scripting`, `nativeMessaging`).
- [ ] Implement WebSocket relay between extension background script and `backend/app/main.py`.
- [ ] Implement DOM snapshot serializer compatible with `PageObservation` schema.
