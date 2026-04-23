"""Optional FastAPI webhook adapter for deploying the AutoStream agent."""

from __future__ import annotations

import os
from typing import Any, Dict

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request

from autostream_agent import AgentState, AutoStreamAgent

load_dotenv()

app = FastAPI(title="AutoStream Social-to-Lead Webhook")
agent = AutoStreamAgent()
SESSION_STORE: Dict[str, AgentState] = {}


@app.get("/webhook/whatsapp")
def verify_webhook(
    hub_mode: str = "",
    hub_challenge: str = "",
    hub_verify_token: str = "",
) -> str:
    """Verify a WhatsApp Cloud API webhook subscription."""
    expected_token = os.getenv("WHATSAPP_VERIFY_TOKEN", "autostream-demo-token")
    if hub_mode == "subscribe" and hub_verify_token == expected_token:
        return hub_challenge
    raise HTTPException(status_code=403, detail="Invalid webhook verification token")


@app.post("/webhook/whatsapp")
async def receive_whatsapp_message(request: Request) -> dict[str, Any]:
    """Handle a WhatsApp-style inbound message and return the agent reply payload."""
    payload = await request.json()
    sender_id, text = _extract_whatsapp_message(payload)

    state = SESSION_STORE.get(sender_id, agent.initial_state())
    updated_state, reply = agent.chat(state, text)
    SESSION_STORE[sender_id] = updated_state

    return {
        "to": sender_id,
        "text": reply,
        "whatsapp_cloud_api_payload": {
            "messaging_product": "whatsapp",
            "to": sender_id,
            "type": "text",
            "text": {"body": reply},
        },
    }


def _extract_whatsapp_message(payload: dict[str, Any]) -> tuple[str, str]:
    """Extract sender and text from WhatsApp Cloud API payloads or simple test JSON."""
    if "from" in payload and "text" in payload:
        return str(payload["from"]), str(payload["text"])

    try:
        message = payload["entry"][0]["changes"][0]["value"]["messages"][0]
        sender_id = message["from"]
        text = message["text"]["body"]
        return str(sender_id), str(text)
    except (KeyError, IndexError, TypeError) as exc:
        raise HTTPException(status_code=400, detail="Unsupported WhatsApp payload") from exc
