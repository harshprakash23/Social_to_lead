"""Streamlit frontend for recording the AutoStream social-to-lead workflow."""

from __future__ import annotations

from typing import Any, Dict, List

import streamlit as st
from dotenv import load_dotenv

from autostream_agent import AgentState, AutoStreamAgent


DEMO_MESSAGES = [
    "Hi, tell me about your pricing.",
    "That sounds good, I want to try the Pro plan for my YouTube channel.",
    "My name is Aisha Mehta.",
    "My email is aisha@example.com.",
]


def main() -> None:
    load_dotenv()
    st.set_page_config(page_title="AutoStream Lead Agent", page_icon="AS", layout="wide")
    _apply_styles()
    _init_session()

    left, right = st.columns([0.64, 0.36], gap="large")

    with left:
        st.markdown("<div class='brand'>AutoStream</div>", unsafe_allow_html=True)
        st.markdown("<h1>Social-to-Lead Agent</h1>", unsafe_allow_html=True)
        st.caption("Conversational AI workflow for pricing answers, high-intent detection, and lead capture.")
        _render_demo_controls()
        _render_chat()
        _render_chat_input()

    with right:
        _render_workflow_panel()
        _render_rag_panel()
        _render_tool_panel()


def _init_session() -> None:
    if "agent" not in st.session_state:
        st.session_state.agent = AutoStreamAgent()
    if "agent_state" not in st.session_state:
        st.session_state.agent_state = st.session_state.agent.initial_state()
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "last_tool_event" not in st.session_state:
        st.session_state.last_tool_event = None


def _render_demo_controls() -> None:
    st.markdown("#### Recording Demo")
    demo_cols = st.columns([0.5, 0.5])
    with demo_cols[0]:
        if st.button("Load expected flow", use_container_width=True):
            _reset_conversation()
            for message in DEMO_MESSAGES:
                _send_message(message)
            st.rerun()
    with demo_cols[1]:
        if st.button("Reset conversation", use_container_width=True):
            _reset_conversation()
            st.rerun()

def _render_chat() -> None:
    st.markdown("#### Conversation")
    if not st.session_state.chat_history:
        with st.chat_message("assistant"):
            st.write(
                "Hi, I am the AutoStream lead agent. Ask me about pricing, features, refunds, "
                "or say you want to try the Pro plan when you are ready."
            )

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])


def _render_chat_input() -> None:
    user_message = st.chat_input("Ask about pricing, then say you want to try the Pro plan")
    if user_message:
        _send_message(user_message)
        st.rerun()


def _render_workflow_panel() -> None:
    state: AgentState = st.session_state.agent_state
    lead = state["lead"]
    missing = state["missing_fields"]

    st.markdown("#### Live Agent State")
    st.markdown(
        f"""
        <div class='status-panel'>
            <div class='metric-label'>Current intent</div>
            <div class='intent'>{_humanize_intent(state["intent"])}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cols = st.columns(3)
    cols[0].markdown(_field_card("Name", lead.get("name", "Missing")), unsafe_allow_html=True)
    cols[1].markdown(_field_card("Email", lead.get("email", "Missing")), unsafe_allow_html=True)
    cols[2].markdown(_field_card("Platform", lead.get("platform", "Missing")), unsafe_allow_html=True)

    if missing:
        st.warning(f"Waiting for: {', '.join(missing)}")
    else:
        st.success("All lead details collected")

    steps = [
        ("Pricing answer from RAG", _has_pricing_answer()),
        ("High-intent detected", state["intent"] == "high_intent_lead"),
        ("Lead details collected", not missing),
        ("Mock lead capture complete", state["lead_capture_done"]),
    ]
    for label, done in steps:
        st.markdown(_step_row(label, done), unsafe_allow_html=True)


def _render_rag_panel() -> None:
    state: AgentState = st.session_state.agent_state
    st.markdown("#### Retrieved Knowledge")
    if not state["retrieved_docs"]:
        st.caption("No retrieval yet.")
        return

    for doc in state["retrieved_docs"]:
        st.markdown(
            f"""
            <div class='kb-row'>
                <div class='kb-title'>{doc["title"]}</div>
                <div>{doc["content"]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_tool_panel() -> None:
    st.markdown("#### Tool Execution")
    event = st.session_state.last_tool_event
    if event:
        st.success("mock_lead_capture() executed")
        st.code(
            f"Lead captured successfully: {event['name']}, {event['email']}, {event['platform']}",
            language="text",
        )
    else:
        st.info("Tool has not run yet. It will run only after name, email, and platform are collected.")


def _send_message(user_message: str) -> None:
    old_state: AgentState = st.session_state.agent_state
    old_capture_done = old_state["lead_capture_done"]

    new_state, response = st.session_state.agent.chat(old_state, user_message)
    st.session_state.agent_state = new_state
    st.session_state.chat_history.append({"role": "user", "content": user_message})
    st.session_state.chat_history.append({"role": "assistant", "content": response})

    if not old_capture_done and new_state["lead_capture_done"]:
        lead = new_state["lead"]
        st.session_state.last_tool_event = {
            "name": lead["name"],
            "email": lead["email"],
            "platform": lead["platform"],
        }


def _reset_conversation() -> None:
    st.session_state.agent = AutoStreamAgent()
    st.session_state.agent_state = st.session_state.agent.initial_state()
    st.session_state.chat_history = []
    st.session_state.last_tool_event = None


def _has_pricing_answer() -> bool:
    history: List[Dict[str, str]] = st.session_state.chat_history
    return any("$29/month" in message["content"] and "$79/month" in message["content"] for message in history)


def _humanize_intent(intent: str) -> str:
    labels = {
        "casual_greeting": "Casual greeting",
        "product_or_pricing_inquiry": "Product or pricing inquiry",
        "high_intent_lead": "High-intent lead",
    }
    return labels.get(intent, intent)


def _field_card(label: str, value: str) -> str:
    status_class = "field-value" if value != "Missing" else "field-missing"
    return (
        "<div class='field-card'>"
        f"<div class='field-label'>{label}</div>"
        f"<div class='{status_class}'>{value}</div>"
        "</div>"
    )


def _step_row(label: str, done: bool) -> str:
    marker = "Done" if done else "Pending"
    status_class = "step-done" if done else "step-pending"
    return (
        f"<div class='step-row {status_class}'>"
        f"<span class='step-dot'></span>"
        f"<span class='step-label'>{label}</span>"
        f"<span class='step-status'>{marker}</span>"
        "</div>"
    )


def _apply_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: #0f172a;
            color: #e5edf8;
        }
        [data-testid="stHeader"] {
            background: rgba(15, 23, 42, 0);
        }
        [data-testid="stAppViewContainer"] {
            background: radial-gradient(circle at 20% 0%, #1e3a5f 0, #0f172a 38%, #111827 100%);
        }
        [data-testid="stSidebar"] {
            background: #111827;
        }
        .block-container {
            padding-top: 2rem;
            max-width: 1280px;
        }
        p, li, label, span, div {
            color: #e5edf8;
        }
        .brand {
            width: fit-content;
            padding: 0.35rem 0.6rem;
            border: 1px solid #38bdf8;
            border-radius: 6px;
            background: rgba(14, 165, 233, 0.14);
            color: #bae6fd;
            font-weight: 700;
            letter-spacing: 0;
        }
        h1 {
            margin: 0.4rem 0 0.1rem;
            font-size: 2.2rem;
            letter-spacing: 0;
            color: #f8fafc;
        }
        h2, h3, h4 {
            color: #f8fafc;
        }
        [data-testid="stCaptionContainer"] p {
            color: #b6c3d6;
        }
        .stButton > button {
            border-radius: 8px;
            border: 1px solid #38bdf8;
            background: #0ea5e9;
            color: #031525;
            font-weight: 800;
        }
        .stButton > button:hover {
            border-color: #7dd3fc;
            background: #38bdf8;
            color: #031525;
        }
        .status-panel {
            padding: 1rem;
            border: 1px solid #334155;
            border-radius: 8px;
            background: rgba(15, 23, 42, 0.82);
            margin-bottom: 0.8rem;
        }
        .metric-label {
            color: #b6c3d6;
            font-size: 0.82rem;
            margin-bottom: 0.25rem;
        }
        .intent {
            color: #7dd3fc;
            font-weight: 800;
            font-size: 1.15rem;
        }
        .field-card {
            border: 1px solid #334155;
            border-radius: 8px;
            background: rgba(15, 23, 42, 0.82);
            padding: 0.75rem;
            min-height: 82px;
            margin-bottom: 0.75rem;
        }
        .field-label {
            color: #93a4b8;
            font-size: 0.78rem;
            margin-bottom: 0.3rem;
        }
        .field-value {
            color: #bbf7d0;
            font-weight: 800;
            overflow-wrap: anywhere;
        }
        .field-missing {
            color: #fcd34d;
            font-weight: 800;
        }
        .step-row {
            display: flex;
            align-items: center;
            gap: 0.6rem;
            border: 1px solid #334155;
            border-radius: 8px;
            background: rgba(15, 23, 42, 0.82);
            padding: 0.65rem 0.75rem;
            margin: 0.45rem 0;
        }
        .step-dot {
            width: 0.7rem;
            height: 0.7rem;
            border-radius: 50%;
            display: inline-block;
            flex: 0 0 auto;
        }
        .step-done .step-dot {
            background: #22c55e;
        }
        .step-pending .step-dot {
            background: #64748b;
        }
        .step-label {
            color: #e5edf8;
            font-weight: 700;
            flex: 1;
        }
        .step-status {
            color: #b6c3d6;
            font-size: 0.82rem;
        }
        .kb-row {
            border: 1px solid #334155;
            border-radius: 8px;
            background: rgba(15, 23, 42, 0.82);
            padding: 0.8rem;
            margin-bottom: 0.6rem;
            line-height: 1.42;
            color: #e5edf8;
        }
        .kb-title {
            color: #7dd3fc;
            font-weight: 800;
            margin-bottom: 0.25rem;
        }
        div[data-testid="stChatMessage"] {
            background: rgba(15, 23, 42, 0.9);
            border: 1px solid #334155;
            border-radius: 8px;
            color: #e5edf8;
        }
        div[data-testid="stChatMessage"] p {
            color: #e5edf8;
            font-size: 1rem;
            line-height: 1.55;
        }
        div[data-testid="stChatMessage"] svg {
            color: #e5edf8;
        }
        div[data-testid="stAlert"] {
            border-radius: 8px;
            color: #0f172a;
        }
        div[data-testid="stChatInput"] textarea {
            color: #ffffff;
        }
        div[data-testid="stChatInput"] textarea::placeholder {
            color: #c7ceda;
        }
        [data-testid="stTextInput"] input {
            background: #111827;
            color: #e5edf8;
            border: 1px solid #334155;
        }
        [data-testid="stExpander"] {
            border: 1px solid #334155;
            background: rgba(15, 23, 42, 0.65);
            border-radius: 8px;
        }
        code {
            color: #e5edf8;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
