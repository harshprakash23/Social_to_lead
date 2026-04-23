# Social-to-Lead Agentic Workflow

Conversational AI agent for **AutoStream**, a fictional SaaS product that provides automated video editing tools for content creators.

The agent can:

- Identify user intent as casual greeting, product/pricing inquiry, or high-intent lead.
- Retrieve pricing, feature, and policy answers from a local JSON knowledge base.
- Retain conversation state across multiple turns with LangGraph.
- Collect lead details safely and call `mock_lead_capture(name, email, platform)` only after name, email, and creator platform are all available.

## Assignment Checklist

- **Language:** Python 3.9+
- **Framework:** LangGraph from the LangChain ecosystem
- **LLM:** GPT-4o-mini through `langchain-openai`
- **State management:** LangGraph `StateGraph` carries messages, intent, retrieved docs, lead fields, missing fields, and tool status across turns
- **Local RAG source:** `knowledge_base/autostream_kb.json`
- **Intent classes:** casual greeting, product/pricing inquiry, high-intent lead
- **Lead capture gate:** `mock_lead_capture` runs only after name, email, and creator platform are collected

## Project Structure

```text
.
+-- autostream_agent/
|   +-- __init__.py
|   +-- agent.py
|   +-- kb.py
|   +-- llm.py
|   +-- tools.py
+-- knowledge_base/
|   +-- autostream_kb.json
+-- tests/
|   +-- test_agent_workflow.py
+-- main.py
+-- requirements.txt
+-- README.md
```

## How To Run Locally

1. Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Optional but recommended: set your OpenAI API key for GPT-4o-mini responses:

```bash
set OPENAI_API_KEY=your_api_key_here
```

4. Start the chat:

```bash
python main.py
```

Run the smoke test:

```bash
python -m unittest discover -s tests
```

Try this sample conversation:

```text
You: Hi, tell me about your pricing.
Agent: Basic Plan costs $29/month... Pro Plan costs $79/month...
You: That sounds good, I want to try the Pro plan for my YouTube channel.
Agent: I have your creator platform: YouTube. Please share your name, and email...
You: My name is Aisha Mehta.
Agent: I have your creator platform: YouTube, name: Aisha Mehta. Please share your email...
You: My email is aisha@example.com.
Agent: Thanks, Aisha Mehta. I captured your interest in AutoStream for YouTube...
```

When all three lead fields are collected, the terminal prints:

```text
Lead captured successfully: Aisha Mehta, aisha@example.com, YouTube
```

## Architecture Explanation

This project uses **LangGraph** because the assignment requires a real agentic workflow with state, branching behavior, retrieval, and tool execution. A normal chatbot loop can answer messages, but LangGraph makes the agent state explicit: every turn carries conversation history, detected intent, retrieved knowledge, collected lead fields, missing fields, and whether the lead capture tool has already run. The graph has three main nodes. First, `retrieve_knowledge` loads relevant AutoStream facts from the local JSON knowledge base. Second, `understand_user` classifies the user intent and extracts lead details such as name, email, and platform. Third, `generate_response` answers from retrieved knowledge, asks for missing lead fields, or triggers the mock lead capture tool once all required details exist. GPT-4o-mini is used through `langchain-openai` when `OPENAI_API_KEY` is available, while a deterministic fallback keeps the demo runnable locally. State is managed in a LangGraph `StateGraph` using a typed dictionary, so memory persists across 5-6 conversation turns and prevents premature tool calls.

## WhatsApp Deployment With Webhooks

To integrate this agent with WhatsApp, I would connect it to the WhatsApp Business Cloud API. A backend service such as FastAPI or Flask would expose a `/webhook/whatsapp` endpoint and verify Meta's webhook challenge during setup. When a WhatsApp user sends a message, Meta posts the event payload to this endpoint. The backend extracts the sender phone number and message text, loads or creates that user's LangGraph state from a database such as Redis or PostgreSQL, and passes the message into the AutoStream agent. The agent returns the next response and updates state with intent, retrieved context, and any collected lead fields. If the user becomes a high-intent lead and provides name, email, and platform, the backend runs `mock_lead_capture` or a real CRM API. Finally, the backend sends the agent's reply back to the user through the WhatsApp Cloud API `/messages` endpoint. Each phone number acts as a conversation/session key so memory persists across multiple turns.
