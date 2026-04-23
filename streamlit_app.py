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

    st.caption("For the cleanest recording, paste each message without the number at the beginning.")
    with st.expander("Suggested recording script", expanded=False):
        for index, message in enumerate(DEMO_MESSAGES, start=1):
            st.text_input(f"Message {index}", value=message, key=f"demo_message_{index}")


def _render_chat() -> None:
    st.markdown("#### Conversation")
    if not st.session_state.chat_history:
        st.info("Start with: Hi, tell me about your pricing.")

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
    cols[0].metric("Name", lead.get("name", "Missing"))
    cols[1].metric("Email", lead.get("email", "Missing"))
    cols[2].metric("Platform", lead.get("platform", "Missing"))

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
        st.checkbox(label, value=done, disabled=True)


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


def _apply_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: #f6f8fb;
            color: #172033;
        }
        .brand {
            width: fit-content;
            padding: 0.35rem 0.6rem;
            border: 1px solid #bed0e8;
            border-radius: 6px;
            background: #eaf2ff;
            color: #244f85;
            font-weight: 700;
            letter-spacing: 0;
        }
        h1 {
            margin: 0.4rem 0 0.1rem;
            font-size: 2.2rem;
            letter-spacing: 0;
        }
        h4 {
            margin-top: 1rem;
        }
        .status-panel {
            padding: 1rem;
            border: 1px solid #d9e2ef;
            border-radius: 8px;
            background: #ffffff;
            margin-bottom: 0.8rem;
        }
        .metric-label {
            color: #5d6b82;
            font-size: 0.82rem;
            margin-bottom: 0.25rem;
        }
        .intent {
            color: #123766;
            font-weight: 800;
            font-size: 1.15rem;
        }
        .kb-row {
            border: 1px solid #d9e2ef;
            border-radius: 8px;
            background: #ffffff;
            padding: 0.8rem;
            margin-bottom: 0.6rem;
            line-height: 1.42;
        }
        .kb-title {
            color: #244f85;
            font-weight: 800;
            margin-bottom: 0.25rem;
        }
        div[data-testid="stMetricValue"] {
            font-size: 1rem;
            line-height: 1.25;
        }
        div[data-testid="stChatMessage"] {
            background: #ffffff;
            border: 1px solid #d9e2ef;
            border-radius: 8px;
            color: #172033;
        }
        div[data-testid="stChatMessage"] p {
            color: #172033;
            font-size: 1rem;
            line-height: 1.55;
        }
        div[data-testid="stChatMessage"] svg {
            color: #172033;
        }
        div[data-testid="stChatInput"] textarea {
            color: #ffffff;
        }
        div[data-testid="stChatInput"] textarea::placeholder {
            color: #c7ceda;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
