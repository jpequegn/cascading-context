import math
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from ctx.facts import Embedder, Fact, FactStore


@dataclass
class ScoredFact:
    fact: Fact
    score: float


class FactRetriever:
    def __init__(
        self,
        store: FactStore,
        embedder: Embedder,
        min_confidence: float = 0.0,
        decay_lambda: float = 0.0,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._min_confidence = min_confidence
        self._decay_lambda = decay_lambda

    def retrieve(self, query: str, top_k: int = 5) -> list[ScoredFact]:
        query_vec = np.array(self._embedder.embed(query))
        facts = self._store.get_all()
        now = datetime.now()

        scored: list[ScoredFact] = []
        for fact in facts:
            if fact.confidence < self._min_confidence:
                continue
            fact_vec = np.array(fact.embedding)
            similarity = self._cosine_similarity(query_vec, fact_vec)

            if self._decay_lambda > 0 and fact.created_at:
                days_since = (now - fact.created_at).total_seconds() / 86400
                decay_factor = math.exp(-self._decay_lambda * days_since)
                score = similarity * decay_factor
            else:
                score = similarity

            scored.append(ScoredFact(fact=fact, score=float(score)))

        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:top_k]

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
