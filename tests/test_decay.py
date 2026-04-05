from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np

from ctx.db import get_connection
from ctx.facts import Fact, FactStore
from ctx.retriever import FactRetriever


class IdenticalEmbedder:
    """Returns the same unit vector for all text — forces identical cosine similarity."""

    def __init__(self, dim: int = 16) -> None:
        self._vec = np.zeros(dim)
        self._vec[0] = 1.0

    def embed(self, text: str) -> list[float]:
        return self._vec.tolist()


def test_newer_fact_ranks_higher_with_decay(tmp_path: Path):
    conn = get_connection(tmp_path / "test.db")
    embedder = IdenticalEmbedder()
    store = FactStore(conn, embedder)

    now = datetime.now()
    old_time = now - timedelta(days=30)
    new_time = now - timedelta(days=1)

    # Insert old fact
    store.insert(Fact(claim="Old fact", category="test", confidence=0.9), session_id="s1")
    # Insert new fact
    store.insert(Fact(claim="New fact", category="test", confidence=0.9), session_id="s1")

    # Patch created_at timestamps directly in DB
    conn.execute("UPDATE facts SET created_at = ? WHERE claim = 'Old fact'", [old_time])
    conn.execute("UPDATE facts SET created_at = ? WHERE claim = 'New fact'", [new_time])

    retriever = FactRetriever(store, embedder, decay_lambda=0.05)
    results = retriever.retrieve("query", top_k=2)

    assert len(results) == 2
    assert results[0].fact.claim == "New fact"
    assert results[1].fact.claim == "Old fact"
    assert results[0].score > results[1].score


def test_no_decay_same_score(tmp_path: Path):
    conn = get_connection(tmp_path / "test.db")
    embedder = IdenticalEmbedder()
    store = FactStore(conn, embedder)

    now = datetime.now()
    conn.execute("BEGIN")
    store.insert(Fact(claim="Old fact", category="test", confidence=0.9), session_id="s1")
    store.insert(Fact(claim="New fact", category="test", confidence=0.9), session_id="s1")
    conn.execute("UPDATE facts SET created_at = ? WHERE claim = 'Old fact'", [now - timedelta(days=30)])
    conn.execute("UPDATE facts SET created_at = ? WHERE claim = 'New fact'", [now - timedelta(days=1)])

    # No decay — scores should be equal (identical embeddings)
    retriever = FactRetriever(store, embedder, decay_lambda=0.0)
    results = retriever.retrieve("query", top_k=2)

    assert abs(results[0].score - results[1].score) < 1e-9


def test_high_decay_strongly_penalizes_old(tmp_path: Path):
    conn = get_connection(tmp_path / "test.db")
    embedder = IdenticalEmbedder()
    store = FactStore(conn, embedder)

    now = datetime.now()
    store.insert(Fact(claim="Ancient fact", category="test", confidence=0.9), session_id="s1")
    store.insert(Fact(claim="Recent fact", category="test", confidence=0.9), session_id="s1")

    conn.execute("UPDATE facts SET created_at = ? WHERE claim = 'Ancient fact'", [now - timedelta(days=100)])
    conn.execute("UPDATE facts SET created_at = ? WHERE claim = 'Recent fact'", [now - timedelta(hours=1)])

    retriever = FactRetriever(store, embedder, decay_lambda=0.1)
    results = retriever.retrieve("query", top_k=2)

    # 100 days with lambda=0.1 -> decay = exp(-10) ≈ 0.0000454
    assert results[0].fact.claim == "Recent fact"
    assert results[0].score > results[1].score * 10  # Much higher
