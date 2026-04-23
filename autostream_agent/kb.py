"""Local knowledge-base retrieval for AutoStream."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Union


class AutoStreamKnowledgeBase:
    """Loads local AutoStream facts and returns relevant grounded snippets."""

    def __init__(self, path: Union[str, Path]) -> None:
        self.path = Path(path)
        self.data = self._load()
        self.documents = self._build_documents()

    def _load(self) -> dict[str, Any]:
        with self.path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def _build_documents(self) -> list[dict[str, str]]:
        docs: list[dict[str, str]] = []

        for plan in self.data["pricing_and_features"]:
            features = ", ".join(plan["features"])
            docs.append(
                {
                    "id": f"plan:{plan['plan'].lower()}",
                    "title": f"{plan['plan']} Plan",
                    "content": (
                        f"{plan['plan']} Plan costs {plan['price']}. "
                        f"It includes {plan['videos']}, {plan['resolution']} resolution, "
                        f"and these features: {features}."
                    ),
                }
            )

        for policy in self.data["company_policies"]:
            docs.append(
                {
                    "id": f"policy:{policy['policy'].lower()}",
                    "title": policy["policy"],
                    "content": policy["details"],
                }
            )

        docs.append(
            {
                "id": "company:summary",
                "title": "AutoStream Summary",
                "content": self.data["product_summary"],
            }
        )
        return docs

    def retrieve(self, query: str, limit: int = 4) -> list[dict[str, str]]:
        """Return relevant local documents using lightweight keyword scoring."""
        normalized_query = query.lower()
        query_terms = {
            term.strip(".,!?;:()[]{}\"'")
            for term in normalized_query.split()
            if len(term.strip(".,!?;:()[]{}\"'")) > 2
        }

        scored_docs: list[tuple[int, dict[str, str]]] = []
        for doc in self.documents:
            haystack = f"{doc['title']} {doc['content']}".lower()
            score = sum(1 for term in query_terms if term in haystack)

            if "pricing" in normalized_query or "price" in normalized_query or "cost" in normalized_query:
                if doc["id"].startswith("plan:"):
                    score += 4
            if "support" in normalized_query and "support" in haystack:
                score += 4
            if "refund" in normalized_query and "refund" in haystack:
                score += 4
            if "pro" in normalized_query and doc["id"] == "plan:pro":
                score += 4
            if "basic" in normalized_query and doc["id"] == "plan:basic":
                score += 4

            scored_docs.append((score, doc))

        relevant = [doc for score, doc in sorted(scored_docs, key=lambda item: item[0], reverse=True) if score > 0]
        return relevant[:limit] if relevant else self.documents[:limit]
