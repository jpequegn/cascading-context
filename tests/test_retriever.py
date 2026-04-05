from pathlib import Path

import numpy as np

from ctx.db import get_connection
from ctx.facts import Embedder, Fact, FactStore
from ctx.retriever import FactRetriever


class ControlledEmbedder:
    """Embedder that returns preset vectors for testing similarity."""

    def __init__(self, mapping: dict[str, list[float]]) -> None:
        self._mapping = mapping
        self._dim = len(next(iter(mapping.values())))

    def embed(self, text: str) -> list[float]:
        if text in self._mapping:
            return self._mapping[text]
        # Return a random-ish vector for unknown text
        rng = np.random.default_rng(seed=hash(text) % 2**32)
        vec = rng.standard_normal(self._dim).tolist()
        return vec


def _make_unit(vec: list[float]) -> list[float]:
    arr = np.array(vec)
    return (arr / np.linalg.norm(arr)).tolist()


def test_retrieve_top_k(tmp_path: Path):
    # Set up embeddings: query is close to "python" and "coding", far from "cooking"
    python_vec = _make_unit([1.0, 0.0, 0.0])
    coding_vec = _make_unit([0.9, 0.1, 0.0])
    cooking_vec = _make_unit([0.0, 0.0, 1.0])
    query_vec = _make_unit([1.0, 0.05, 0.0])

    embedder = ControlledEmbedder({
        "Python is great": python_vec,
        "I love coding": coding_vec,
        "Cooking is fun": cooking_vec,
        "tell me about programming": query_vec,
    })

    conn = get_connection(tmp_path / "test.db")
    store = FactStore(conn, embedder)
    store.insert(Fact(claim="Python is great", category="tech", confidence=0.9), session_id="s1")
    store.insert(Fact(claim="I love coding", category="tech", confidence=0.8), session_id="s1")
    store.insert(Fact(claim="Cooking is fun", category="food", confidence=0.7), session_id="s1")

    retriever = FactRetriever(store, embedder)
    results = retriever.retrieve("tell me about programming", top_k=2)

    assert len(results) == 2
    assert results[0].fact.claim == "Python is great"
    assert results[1].fact.claim == "I love coding"
    assert results[0].score > results[1].score


def test_min_confidence_filter(tmp_path: Path):
    embedder = ControlledEmbedder({
        "high conf": _make_unit([1.0, 0.0]),
        "low conf": _make_unit([1.0, 0.0]),
        "query": _make_unit([1.0, 0.0]),
    })

    conn = get_connection(tmp_path / "test.db")
    store = FactStore(conn, embedder)
    store.insert(Fact(claim="high conf", category="a", confidence=0.9), session_id="s1")
    store.insert(Fact(claim="low conf", category="a", confidence=0.3), session_id="s1")

    retriever = FactRetriever(store, embedder, min_confidence=0.5)
    results = retriever.retrieve("query", top_k=5)

    assert len(results) == 1
    assert results[0].fact.claim == "high conf"


def test_retrieve_from_50_facts(tmp_path: Path):
    dim = 32
    rng = np.random.default_rng(42)

    # Create a target direction
    target = rng.standard_normal(dim)
    target = target / np.linalg.norm(target)

    mapping = {}
    claims = []
    for i in range(50):
        vec = rng.standard_normal(dim)
        vec = vec / np.linalg.norm(vec)
        claim = f"Fact about topic {i}"
        mapping[claim] = vec.tolist()
        claims.append(claim)

    # Make fact 7 very similar to query
    mapping[claims[7]] = (target + rng.standard_normal(dim) * 0.01).tolist()
    mapping["my query"] = target.tolist()

    embedder = ControlledEmbedder(mapping)
    conn = get_connection(tmp_path / "test.db")
    store = FactStore(conn, embedder)

    for claim in claims:
        store.insert(Fact(claim=claim, category="test", confidence=0.8), session_id="s1")

    retriever = FactRetriever(store, embedder)
    results = retriever.retrieve("my query", top_k=5)

    assert len(results) == 5
    # Fact 7 should be the top result (closest to query)
    assert results[0].fact.claim == claims[7]
    # Scores should be descending
    for i in range(len(results) - 1):
        assert results[i].score >= results[i + 1].score
