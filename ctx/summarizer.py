import json

import anthropic

from ctx.facts import Fact, FactStore
from ctx.session import Message

EXTRACT_PROMPT = """\
Extract structured facts from this conversation. Return JSON only, no other text.

Format:
{"facts": [{"claim": "...", "confidence": 0.0-1.0, "category": "...", "entities": ["..."]}]}

Rules:
- Extract durable facts that would be useful in future conversations
- confidence: how certain the fact is (1.0 = stated explicitly, 0.5 = implied)
- category: one of "preference", "technical", "biographical", "project", "opinion", "general"
- entities: key nouns or proper names mentioned in the claim
- Return at least 1 fact, aim for 5-10 if the conversation is substantive
"""


class Summarizer:
    def __init__(self, fact_store: FactStore, client: anthropic.Anthropic | None = None) -> None:
        self._fact_store = fact_store
        self._client = client or anthropic.Anthropic()

    def extract_facts(self, session_id: str, messages: list[Message]) -> list[Fact]:
        transcript = "\n".join(f"{m.role}: {m.content}" for m in messages)

        response = self._client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            system=EXTRACT_PROMPT,
            messages=[{"role": "user", "content": transcript}],
        )

        text = response.content[0].text
        data = json.loads(text)

        facts: list[Fact] = []
        for item in data["facts"]:
            fact = Fact(
                claim=item["claim"],
                confidence=item["confidence"],
                category=item["category"],
                entities=item.get("entities", []),
            )
            self._fact_store.insert(fact, session_id=session_id)
            facts.append(fact)

        return facts
