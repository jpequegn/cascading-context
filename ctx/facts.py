from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol

import duckdb
import numpy as np


@dataclass
class Fact:
    claim: str
    category: str
    confidence: float
    entities: list[str] = field(default_factory=list)
    id: int | None = None
    session_id: str | None = None
    embedding: list[float] = field(default_factory=list)
    created_at: datetime | None = None


class Embedder(Protocol):
    def embed(self, text: str) -> list[float]: ...


class NumpyRandomEmbedder:
    """Deterministic pseudo-random embedder for testing. Replace with a real model."""

    def __init__(self, dim: int = 256) -> None:
        self._dim = dim

    def embed(self, text: str) -> list[float]:
        rng = np.random.default_rng(seed=hash(text) % 2**32)
        vec = rng.standard_normal(self._dim)
        vec = vec / np.linalg.norm(vec)
        return vec.tolist()


class FactStore:
    def __init__(self, conn: duckdb.DuckDBPyConnection, embedder: Embedder) -> None:
        self._conn = conn
        self._embedder = embedder

    def insert(self, fact: Fact, session_id: str | None = None) -> Fact:
        sid = session_id or fact.session_id
        embedding = fact.embedding or self._embedder.embed(fact.claim)
        fact_id = self._conn.execute("SELECT nextval('facts_id_seq')").fetchone()[0]
        now = datetime.now()
        self._conn.execute(
            "INSERT INTO facts (id, session_id, claim, category, confidence, entities, embedding, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [fact_id, sid, fact.claim, fact.category, fact.confidence, fact.entities, embedding, now],
        )
        fact.id = fact_id
        fact.session_id = sid
        fact.embedding = embedding
        fact.created_at = now
        return fact

    def get_all(self) -> list[Fact]:
        rows = self._conn.execute(
            "SELECT id, session_id, claim, category, confidence, entities, embedding, created_at "
            "FROM facts ORDER BY created_at"
        ).fetchall()
        return [
            Fact(
                id=r[0], session_id=r[1], claim=r[2], category=r[3],
                confidence=r[4], entities=r[5], embedding=r[6], created_at=r[7],
            )
            for r in rows
        ]

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
