from pathlib import Path

from ctx.db import get_connection
from ctx.facts import Fact, FactStore, NumpyRandomEmbedder


def test_insert_and_count(tmp_path: Path):
    conn = get_connection(tmp_path / "test.db")
    store = FactStore(conn, NumpyRandomEmbedder())

    assert store.count() == 0

    store.insert(Fact(claim="Python is a language", category="tech", confidence=0.95, entities=["Python"]),
                 session_id="s1")
    store.insert(Fact(claim="DuckDB is fast", category="tech", confidence=0.9, entities=["DuckDB"]),
                 session_id="s1")

    assert store.count() == 2


def test_get_all_returns_facts_with_embeddings(tmp_path: Path):
    conn = get_connection(tmp_path / "test.db")
    store = FactStore(conn, NumpyRandomEmbedder(dim=128))

    store.insert(Fact(claim="Fact one", category="general", confidence=0.8), session_id="s1")
    store.insert(Fact(claim="Fact two", category="general", confidence=0.7), session_id="s1")

    facts = store.get_all()
    assert len(facts) == 2
    assert facts[0].claim == "Fact one"
    assert len(facts[0].embedding) == 128
    assert facts[1].id == 2


def test_insert_20_facts_with_embeddings(tmp_path: Path):
    conn = get_connection(tmp_path / "test.db")
    store = FactStore(conn, NumpyRandomEmbedder(dim=256))

    for i in range(20):
        store.insert(
            Fact(claim=f"Fact number {i}", category="test", confidence=0.5 + i * 0.025,
                 entities=[f"entity_{i}"]),
            session_id="sess1",
        )

    assert store.count() == 20
    facts = store.get_all()
    for f in facts:
        assert len(f.embedding) == 256
        assert f.embedding != [0.0] * 256


def test_embedder_deterministic():
    embedder = NumpyRandomEmbedder(dim=64)
    v1 = embedder.embed("hello world")
    v2 = embedder.embed("hello world")
    assert v1 == v2

    v3 = embedder.embed("different text")
    assert v3 != v1
