from pathlib import Path

import numpy as np

from ctx.db import get_connection
from ctx.facts import Fact, FactStore
from ctx.injector import ContextInjector
from ctx.retriever import FactRetriever


class FixedEmbedder:
    def __init__(self, dim: int = 16) -> None:
        self._dim = dim

    def embed(self, text: str) -> list[float]:
        rng = np.random.default_rng(seed=hash(text) % 2**32)
        vec = rng.standard_normal(self._dim)
        return (vec / np.linalg.norm(vec)).tolist()


def _setup(tmp_path: Path, facts_data: list[tuple[str, str, float]]):
    conn = get_connection(tmp_path / "test.db")
    embedder = FixedEmbedder()
    store = FactStore(conn, embedder)
    for claim, category, confidence in facts_data:
        store.insert(Fact(claim=claim, category=category, confidence=confidence), session_id="s1")
    retriever = FactRetriever(store, embedder)
    return ContextInjector(retriever)


def test_build_system_prompt_with_facts(tmp_path: Path):
    injector = _setup(tmp_path, [
        ("Python is great", "tech", 0.95),
        ("User likes TDD", "preference", 0.8),
    ])
    prompt = injector.build_system_prompt("You are a helpful assistant.", "tell me about Python")
    assert "You are a helpful assistant." in prompt
    assert "## What I already know" in prompt
    assert "Python is great" in prompt


def test_confidence_labels(tmp_path: Path):
    injector = _setup(tmp_path, [
        ("High fact", "a", 0.95),
        ("Mid fact", "a", 0.7),
        ("Low fact", "a", 0.4),
    ])
    prompt = injector.build_system_prompt("Base.", "query", top_k=10)
    assert "high confidence" in prompt
    assert "moderate confidence" in prompt
    assert "low confidence" in prompt


def test_no_facts_returns_base_prompt(tmp_path: Path):
    conn = get_connection(tmp_path / "test.db")
    embedder = FixedEmbedder()
    store = FactStore(conn, embedder)
    retriever = FactRetriever(store, embedder)
    injector = ContextInjector(retriever)

    prompt = injector.build_system_prompt("You are helpful.", "anything")
    assert prompt == "You are helpful."


def test_token_budget_truncates(tmp_path: Path):
    # Create many facts with long claims to exceed budget
    facts_data = [
        (f"This is a very long fact number {i} with lots of extra words to fill up the token budget quickly " * 5,
         "general", 0.8)
        for i in range(50)
    ]
    injector = _setup(tmp_path, facts_data)
    prompt = injector.build_system_prompt("Base.", "query", top_k=50)

    # Should have truncated before including all 50 facts
    fact_lines = [l for l in prompt.split("\n") if l.startswith("- ")]
    assert len(fact_lines) < 50
    assert len(prompt) < 10000  # Well within reasonable bounds
