"""Command-line runner for the AutoStream social-to-lead agent."""

from __future__ import annotations

from dotenv import load_dotenv

from autostream_agent import AutoStreamAgent


def main() -> None:
    load_dotenv()

    agent = AutoStreamAgent()
    state = agent.initial_state()

    print("AutoStream Agent is ready. Type 'exit' to quit.")
    while True:
        user_message = input("You: ").strip()
        if user_message.lower() in {"exit", "quit"}:
            print("Agent: Thanks for chatting with AutoStream.")
            break
        if not user_message:
            continue

        state, response = agent.chat(state, user_message)
        print(f"Agent: {response}")


if __name__ == "__main__":
    main()
