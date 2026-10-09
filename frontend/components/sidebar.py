"""Sidebar navigation, service connectivity monitor, and configuration controls."""

import streamlit as st
from typing import Optional
from ..services.backend_client import BackendClient
from ..services.ollama_client import OllamaService


def render_sidebar(backend_client: BackendClient, ollama_service: OllamaService) -> None:
    """Render the sidebar with connection badges, settings, and stop controls."""
    with st.sidebar:
        # 1. Branding Header
        st.markdown(
            """
            <div style="margin-bottom: 1.25rem;">
                <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.25rem;">
                    <span style="font-size: 1.6rem;">🌐</span>
                    <h2 style="margin: 0; font-size: 1.35rem; font-weight: 700;">BrowserPilot AI</h2>
                </div>
                <div style="font-size: 0.8rem; color: #94a3b8; line-height: 1.3;">
                    Local-first autonomous browser navigation agent powered by Ollama & Playwright.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()

        # 2. Service Connectivity Monitoring
        st.subheader("Service Status")

        # Backend Health
        backend_health = backend_client.check_health()
        if backend_health.get("online"):
            st.markdown(
                """
                <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 6px; padding: 0.5rem 0.75rem; margin-bottom: 0.5rem;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-weight: 600; font-size: 0.85rem; color: #10b981;">● Backend Online</span>
                        <span style="font-size: 0.75rem; color: #94a3b8;">FastAPI</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            err_msg = backend_health.get("error", "Offline")
            st.markdown(
                f"""
                <div style="background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.3); border-radius: 6px; padding: 0.5rem 0.75rem; margin-bottom: 0.5rem;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-weight: 600; font-size: 0.85rem; color: #ef4444;">● Backend Offline</span>
                    </div>
                    <div style="font-size: 0.72rem; color: #fca5a5; margin-top: 0.25rem;">{err_msg}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Ollama Health
        ollama_health = ollama_service.check_health()
        if ollama_health.get("online"):
            model_info = "Ready" if ollama_health.get("model_present") else "Model not found"
            badge_color = "#10b981" if ollama_health.get("model_present") else "#f59e0b"
            st.markdown(
                f"""
                <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 6px; padding: 0.5rem 0.75rem; margin-bottom: 0.75rem;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-weight: 600; font-size: 0.85rem; color: {badge_color};">● Ollama Connected</span>
                        <span style="font-size: 0.75rem; color: #94a3b8;">{len(ollama_health.get('models', []))} models</span>
                    </div>
                    <div style="font-size: 0.72rem; color: #94a3b8; margin-top: 0.2rem;">Model: <code>{ollama_service.default_model}</code> ({model_info})</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div style="background: rgba(245, 158, 11, 0.1); border: 1px solid rgba(245, 158, 11, 0.3); border-radius: 6px; padding: 0.5rem 0.75rem; margin-bottom: 0.75rem;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-weight: 600; font-size: 0.85rem; color: #f59e0b;">● Ollama Unreachable</span>
                    </div>
                    <div style="font-size: 0.72rem; color: #fcd34d; margin-top: 0.25rem;">{ollama_health.get('error', '')}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # 3. Active Run Flow Control
        is_running = st.session_state.get("is_running", False)
        if is_running:
            st.markdown(
                """
                <div style="background: rgba(59, 130, 246, 0.15); border: 1px solid rgba(59, 130, 246, 0.4); border-radius: 6px; padding: 0.6rem; margin-bottom: 0.75rem; text-align: center;">
                    <div style="font-size: 0.85rem; font-weight: 600; color: #60a5fa;">Agent Task Active</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("⏹ Stop Agent Run", key="sidebar_stop_btn", use_container_width=True, type="primary"):
                backend_client.stop_agent()
                st.session_state["is_running"] = False
                st.session_state["stop_requested"] = True
                st.toast("Stop signal dispatched to backend.", icon="⏹️")
                st.rerun()

        st.divider()

        # 4. Settings & Endpoint Overrides
        with st.expander("⚙️ Configuration & Endpoints", expanded=False):
            new_backend_url = st.text_input(
                "Backend URL",
                value=st.session_state.get("backend_url", backend_client.base_url),
                help="Base URL for the FastAPI backend server.",
            )
            if new_backend_url != st.session_state.get("backend_url"):
                st.session_state["backend_url"] = new_backend_url
                backend_client.base_url = new_backend_url.rstrip("/")

            new_ollama_url = st.text_input(
                "Ollama Base URL",
                value=st.session_state.get("ollama_url", ollama_service.base_url),
                help="Base URL for the local Ollama instance.",
            )
            if new_ollama_url != st.session_state.get("ollama_url"):
                st.session_state["ollama_url"] = new_ollama_url
                ollama_service.base_url = new_ollama_url.rstrip("/")

            available_models = ollama_health.get("models", [])
            current_model = st.session_state.get("model_name", ollama_service.default_model)
            if available_models:
                # If current model in list, use selectbox
                default_idx = available_models.index(current_model) if current_model in available_models else 0
                selected_model = st.selectbox(
                    "Model Name",
                    options=available_models,
                    index=default_idx,
                    help="Configured model loaded in local Ollama.",
                )
            else:
                selected_model = st.text_input(
                    "Model Name",
                    value=current_model,
                    help="Model tag to request from Ollama.",
                )

            if selected_model != st.session_state.get("model_name"):
                st.session_state["model_name"] = selected_model
                ollama_service.default_model = selected_model

            st.session_state["max_steps"] = st.slider(
                "Max Steps per Task",
                min_value=3,
                max_value=30,
                value=st.session_state.get("max_steps", 15),
                help="Upper safety limit on cognitive steps before halting.",
            )

            st.session_state["start_url"] = st.text_input(
                "Start URL (Optional)",
                value=st.session_state.get("start_url", "http://localhost:8080/index.html"),
                help="Initial browser destination for the task.",
            )

        # 5. Quick Refresh Button
        if st.button("🔄 Refresh Status", key="sidebar_refresh_btn", use_container_width=True):
            st.rerun()
