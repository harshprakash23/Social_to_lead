import io
import unittest
from contextlib import redirect_stdout

from autostream_agent import AutoStreamAgent


class AutoStreamAgentWorkflowTest(unittest.TestCase):
    def test_expected_social_to_lead_flow(self) -> None:
        agent = AutoStreamAgent()
        state = agent.initial_state()

        state, pricing_response = agent.chat(state, "Hi, tell me about your pricing.")
        self.assertEqual(state["intent"], "product_or_pricing_inquiry")
        self.assertIn("$29/month", pricing_response)
        self.assertIn("$79/month", pricing_response)
        self.assertFalse(state["lead_capture_done"])

        state, lead_response = agent.chat(
            state,
            "That sounds good, I want to try the Pro plan for my YouTube channel.",
        )
        self.assertEqual(state["intent"], "high_intent_lead")
        self.assertEqual(state["lead"]["platform"], "YouTube")
        self.assertIn("name", state["missing_fields"])
        self.assertIn("email", state["missing_fields"])
        self.assertFalse(state["lead_capture_done"])
        self.assertIn("Please share", lead_response)

        state, name_response = agent.chat(state, "My name is Aisha Mehta.")
        self.assertEqual(state["lead"]["name"], "Aisha Mehta")
        self.assertEqual(state["missing_fields"], ["email"])
        self.assertFalse(state["lead_capture_done"])
        self.assertIn("email", name_response)

        output = io.StringIO()
        with redirect_stdout(output):
            state, final_response = agent.chat(state, "My email is aisha@example.com.")

        self.assertTrue(state["lead_capture_done"])
        self.assertEqual(state["missing_fields"], [])
        self.assertIn("Lead captured successfully: Aisha Mehta, aisha@example.com, YouTube", output.getvalue())
        self.assertIn("captured your interest", final_response)


if __name__ == "__main__":
    unittest.main()
