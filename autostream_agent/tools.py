"""Tool functions used by the AutoStream agent."""


def mock_lead_capture(name: str, email: str, platform: str) -> None:
    """Mock backend action for capturing a qualified creator lead."""
    print(f"Lead captured successfully: {name}, {email}, {platform}")
