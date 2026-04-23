"""Optional GPT-4o-mini helpers with deterministic fallbacks."""

from __future__ import annotations

import json
import os
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage


class LLMClient:
    """Small wrapper around GPT-4o-mini that keeps the project runnable offline."""

    def __init__(self) -> None:
        self.enabled = bool(os.getenv("OPENAI_API_KEY"))
        self.chat = None

        if self.enabled:
            try:
                from langchain_openai import ChatOpenAI

                self.chat = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            except Exception:
                self.enabled = False
                self.chat = None

    def classify_intent(self, user_message: str) -> str | None:
        if not self.enabled or self.chat is None:
            return None

        system = (
            "Classify the user's intent for AutoStream. "
            "Return only one JSON object with key intent. "
            "Allowed values: casual_greeting, product_or_pricing_inquiry, high_intent_lead."
        )
        response = self.chat.invoke([SystemMessage(content=system), HumanMessage(content=user_message)])
        try:
            parsed: dict[str, Any] = json.loads(str(response.content))
            intent = parsed.get("intent")
            if intent in {"casual_greeting", "product_or_pricing_inquiry", "high_intent_lead"}:
                return intent
        except json.JSONDecodeError:
            return None
        return None

    def answer_from_context(self, user_message: str, context: str) -> str | None:
        if not self.enabled or self.chat is None:
            return None

        system = (
            "You are AutoStream's helpful sales agent. "
            "Answer only with the provided context. "
            "If the context does not contain the answer, say you do not have that information. "
            "Be concise and accurate."
        )
        human = f"Context:\n{context}\n\nUser question:\n{user_message}"
        response = self.chat.invoke([SystemMessage(content=system), HumanMessage(content=human)])
        return str(response.content).strip()
