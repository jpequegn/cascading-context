import json
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

from ctx.db import get_connection
from ctx.facts import FactStore, NumpyRandomEmbedder
from ctx.session import Message
from ctx.summarizer import Summarizer


def _make_mock_client(facts_json: list[dict]) -> MagicMock:
    client = MagicMock()
    response = MagicMock()
    response.content = [MagicMock(text=json.dumps({"facts": facts_json}))]
    client.messages.create.return_value = response
    return client


def _make_messages(session_id: str, pairs: list[tuple[str, str]]) -> list[Message]:
    messages = []
    for i, (role, content) in enumerate(pairs):
        messages.append(Message(id=i + 1, session_id=session_id, role=role,
                                content=content, created_at=datetime.now()))
    return messages


def test_extract_facts_from_session(tmp_path: Path):
    conn = get_connection(tmp_path / "test.db")
    store = FactStore(conn, NumpyRandomEmbedder())

    mock_facts = [
        {"claim": "User is building a CLI tool", "confidence": 0.95,
         "category": "project", "entities": ["CLI"]},
        {"claim": "User prefers Python", "confidence": 0.9,
         "category": "preference", "entities": ["Python"]},
        {"claim": "Project uses DuckDB for storage", "confidence": 1.0,
         "category": "technical", "entities": ["DuckDB"]},
    ]
    client = _make_mock_client(mock_facts)
    summarizer = Summarizer(store, client=client)

    messages = _make_messages("s1", [
        ("user", "I'm building a CLI tool in Python"),
        ("assistant", "Great! What storage backend?"),
        ("user", "DuckDB for the local store"),
    ])

    facts = summarizer.extract_facts("s1", messages)

    assert len(facts) == 3
    assert facts[0].claim == "User is building a CLI tool"
    assert facts[0].confidence == 0.95
    assert facts[2].category == "technical"
    assert store.count() == 3


def test_extract_facts_stores_with_embeddings(tmp_path: Path):
    conn = get_connection(tmp_path / "test.db")
    store = FactStore(conn, NumpyRandomEmbedder(dim=64))

    mock_facts = [
        {"claim": "Fact one", "confidence": 0.8, "category": "general", "entities": []},
        {"claim": "Fact two", "confidence": 0.7, "category": "general", "entities": []},
    ]
    client = _make_mock_client(mock_facts)
    summarizer = Summarizer(store, client=client)

    messages = _make_messages("s1", [("user", "Hello"), ("assistant", "Hi")])
    summarizer.extract_facts("s1", messages)

    all_facts = store.get_all()
    assert len(all_facts) == 2
    assert len(all_facts[0].embedding) == 64


def test_extract_sends_transcript_to_claude(tmp_path: Path):
    conn = get_connection(tmp_path / "test.db")
    store = FactStore(conn, NumpyRandomEmbedder())

    client = _make_mock_client([
        {"claim": "test", "confidence": 0.5, "category": "general", "entities": []},
    ])
    summarizer = Summarizer(store, client=client)

    messages = _make_messages("s1", [
        ("user", "First message"),
        ("assistant", "Second message"),
    ])
    summarizer.extract_facts("s1", messages)

    call_args = client.messages.create.call_args
    user_msg = call_args.kwargs["messages"][0]["content"]
    assert "user: First message" in user_msg
    assert "assistant: Second message" in user_msg


def test_extract_five_plus_facts(tmp_path: Path):
    conn = get_connection(tmp_path / "test.db")
    store = FactStore(conn, NumpyRandomEmbedder())

    mock_facts = [
        {"claim": f"Fact {i}", "confidence": 0.5 + i * 0.05,
         "category": "general", "entities": [f"e{i}"]}
        for i in range(7)
    ]
    client = _make_mock_client(mock_facts)
    summarizer = Summarizer(store, client=client)

    messages = _make_messages("s1", [
        ("user", f"Message {i}") for i in range(10)
    ])
    facts = summarizer.extract_facts("s1", messages)

    assert len(facts) >= 5
    assert store.count() == 7
