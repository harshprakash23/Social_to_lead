"""LangGraph-powered conversational agent for AutoStream."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, TypedDict

from langgraph.graph import END, StateGraph

from .kb import AutoStreamKnowledgeBase
from .llm import LLMClient
from .tools import mock_lead_capture


class Message(TypedDict):
    role: str
    content: str


class LeadInfo(TypedDict, total=False):
    name: str
    email: str
    platform: str


class AgentState(TypedDict):
    messages: list[Message]
    intent: str
    retrieved_docs: list[dict[str, str]]
    lead: LeadInfo
    missing_fields: list[str]
    lead_capture_done: bool


EMAIL_PATTERN = re.compile(r"[\w\.-]+@[\w\.-]+\.\w+")
PLATFORM_PATTERN = re.compile(
    r"\b(YouTube|Instagram|TikTok|Twitch|LinkedIn|Facebook|X|Twitter|Podcast|Shorts|Reels)\b",
    re.IGNORECASE,
)
PLATFORM_NAMES = {
    "youtube": "YouTube",
    "instagram": "Instagram",
    "tiktok": "TikTok",
    "twitch": "Twitch",
    "linkedin": "LinkedIn",
    "facebook": "Facebook",
    "x": "Twitter/X",
    "twitter": "Twitter/X",
    "podcast": "Podcast",
    "shorts": "YouTube Shorts",
    "reels": "Instagram Reels",
}
HIGH_INTENT_TERMS = (
    "sign up",
    "signup",
    "start",
    "try",
    "buy",
    "purchase",
    "subscribe",
    "get started",
    "book",
    "demo",
    "pro plan",
    "ready",
)
PRICING_TERMS = ("price", "pricing", "cost", "plan", "feature", "support", "refund", "caption", "4k", "videos")
GREETING_TERMS = ("hi", "hello", "hey", "good morning", "good afternoon", "good evening")


class AutoStreamAgent:
    """Conversational workflow that identifies intent, retrieves knowledge, and captures leads."""

    def __init__(self, kb_path: str | Path = "knowledge_base/autostream_kb.json") -> None:
        self.kb = AutoStreamKnowledgeBase(kb_path)
        self.llm = LLMClient()
        self.graph = self._build_graph()

    def initial_state(self) -> AgentState:
        return {
            "messages": [],
            "intent": "casual_greeting",
            "retrieved_docs": [],
            "lead": {},
            "missing_fields": ["name", "email", "platform"],
            "lead_capture_done": False,
        }

    def chat(self, state: AgentState, user_message: str) -> tuple[AgentState, str]:
        next_state = {
            **state,
            "messages": [*state["messages"], {"role": "user", "content": user_message}],
        }
        updated_state = self.graph.invoke(next_state)
        assistant_message = updated_state["messages"][-1]["content"]
        return updated_state, assistant_message

    def _build_graph(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("retrieve_knowledge", self._retrieve_knowledge)
        workflow.add_node("understand_user", self._understand_user)
        workflow.add_node("generate_response", self._generate_response)

        workflow.set_entry_point("retrieve_knowledge")
        workflow.add_edge("retrieve_knowledge", "understand_user")
        workflow.add_edge("understand_user", "generate_response")
        workflow.add_edge("generate_response", END)
        return workflow.compile()

    def _retrieve_knowledge(self, state: AgentState) -> AgentState:
        user_message = self._latest_user_message(state)
        return {**state, "retrieved_docs": self.kb.retrieve(user_message)}

    def _understand_user(self, state: AgentState) -> AgentState:
        user_message = self._latest_user_message(state)
        extracted_lead = self._extract_lead_info(user_message)
        lead = {**state.get("lead", {}), **extracted_lead}
        missing_fields = [field for field in ("name", "email", "platform") if not lead.get(field)]
        intent = self._classify_intent(user_message, lead, missing_fields)

        return {
            **state,
            "intent": intent,
            "lead": lead,
            "missing_fields": missing_fields,
        }

    def _generate_response(self, state: AgentState) -> AgentState:
        intent = state["intent"]
        lead = state["lead"]
        missing_fields = state["missing_fields"]

        if intent == "high_intent_lead":
            if missing_fields:
                response = self._lead_question(missing_fields, lead)
            elif not state["lead_capture_done"]:
                mock_lead_capture(lead["name"], lead["email"], lead["platform"])
                response = (
                    f"Thanks, {lead['name']}. I captured your interest in AutoStream for "
                    f"{lead['platform']}. Our team will contact you at {lead['email']} soon."
                )
                state = {**state, "lead_capture_done": True}
            else:
                response = "Your lead details are already captured. Our team will reach out soon."
        elif intent == "product_or_pricing_inquiry":
            response = self._answer_product_question(state)
        else:
            response = "Hi! I can help with AutoStream pricing, features, refunds, support, or getting started."

        return {**state, "messages": [*state["messages"], {"role": "assistant", "content": response}]}

    def _classify_intent(self, user_message: str, lead: LeadInfo, missing_fields: list[str]) -> str:
        text = user_message.lower()
        if any(term in text for term in HIGH_INTENT_TERMS):
            return "high_intent_lead"
        if len(missing_fields) < 3 and ("email" in lead or "name" in lead or "platform" in lead):
            return "high_intent_lead"

        llm_intent = self.llm.classify_intent(user_message)
        if llm_intent:
            return llm_intent

        if any(term in text for term in PRICING_TERMS):
            return "product_or_pricing_inquiry"
        if any(term in text for term in GREETING_TERMS):
            return "casual_greeting"
        return "product_or_pricing_inquiry"

    def _extract_lead_info(self, user_message: str) -> LeadInfo:
        lead: LeadInfo = {}

        email_match = EMAIL_PATTERN.search(user_message)
        if email_match:
            lead["email"] = email_match.group(0)

        platform_match = PLATFORM_PATTERN.search(user_message)
        if platform_match:
            platform = platform_match.group(1).lower()
            lead["platform"] = PLATFORM_NAMES.get(platform, platform.title())

        name = self._extract_name(user_message)
        if name:
            lead["name"] = name

        return lead

    def _extract_name(self, user_message: str) -> Optional[str]:
        patterns = (
            r"\bmy name is\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})",
            r"\bi am\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})",
            r"\bi'm\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})",
            r"\bname:\s*([A-Za-z]+(?:\s+[A-Za-z]+){0,2})",
        )
        for pattern in patterns:
            match = re.search(pattern, user_message, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None

    def _lead_question(self, missing_fields: list[str], lead: LeadInfo) -> str:
        known_parts = []
        if lead.get("platform"):
            known_parts.append(f"creator platform: {lead['platform']}")
        if lead.get("name"):
            known_parts.append(f"name: {lead['name']}")
        if lead.get("email"):
            known_parts.append(f"email: {lead['email']}")

        if len(missing_fields) == 3:
            return "Great, I can help you get started. Please share your name, email, and creator platform."
        if len(missing_fields) == 1:
            question = f"Please share your {missing_fields[0]} to finish the lead capture."
        else:
            question = f"Please share your {', '.join(missing_fields[:-1])}, and {missing_fields[-1]} to finish the lead capture."

        if known_parts:
            return f"I have your {', '.join(known_parts)}. {question}"
        return question

    def _answer_product_question(self, state: AgentState) -> str:
        context = "\n".join(f"- {doc['content']}" for doc in state["retrieved_docs"])
        llm_answer = self.llm.answer_from_context(self._latest_user_message(state), context)
        if llm_answer:
            return llm_answer

        docs = state["retrieved_docs"]
        plan_docs = [doc["content"] for doc in docs if doc["id"].startswith("plan:")]
        policy_docs = [doc["content"] for doc in docs if doc["id"].startswith("policy:")]

        if plan_docs and policy_docs:
            return " ".join([*plan_docs, *policy_docs])
        if plan_docs:
            return " ".join(plan_docs)
        if policy_docs:
            return " ".join(policy_docs)
        return "AutoStream provides automated video editing tools for content creators."

    def _latest_user_message(self, state: AgentState) -> str:
        for message in reversed(state["messages"]):
            if message["role"] == "user":
                return message["content"]
        return ""
