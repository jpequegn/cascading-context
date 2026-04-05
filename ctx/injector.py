import logging

from ctx.retriever import FactRetriever, ScoredFact

logger = logging.getLogger(__name__)

TOKEN_BUDGET = 2000
# Rough estimate: 1 token ≈ 4 chars
CHARS_PER_TOKEN = 4


class ContextInjector:
    def __init__(self, retriever: FactRetriever) -> None:
        self._retriever = retriever

    def build_system_prompt(
        self, base_prompt: str, query: str, top_k: int = 5
    ) -> str:
        results = self._retriever.retrieve(query, top_k=top_k)
        if not results:
            return base_prompt

        facts_section = self._format_facts(results)
        return f"{base_prompt}\n\n{facts_section}"

    def _format_facts(self, results: list[ScoredFact]) -> str:
        header = "## What I already know\n"
        lines: list[str] = []
        budget = TOKEN_BUDGET * CHARS_PER_TOKEN - len(header)

        for r in results:
            confidence_label = self._confidence_label(r.fact.confidence)
            line = f"- {r.fact.claim} [{confidence_label}]"
            if len("\n".join(lines + [line])) > budget:
                logger.info("Token budget reached, truncating facts")
                break
            lines.append(line)

        if not lines:
            return ""

        injected = [r.fact for r in results[: len(lines)]]
        logger.info("Injected %d facts into session", len(injected))

        return header + "\n".join(lines)

    @staticmethod
    def _confidence_label(confidence: float) -> str:
        if confidence >= 0.9:
            return "high confidence"
        if confidence >= 0.6:
            return "moderate confidence"
        return "low confidence"
