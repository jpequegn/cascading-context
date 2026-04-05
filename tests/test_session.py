from pathlib import Path

from ctx.db import get_connection
from ctx.session import SessionManager


def test_create_and_list_sessions(tmp_path: Path):
    conn = get_connection(tmp_path / "test.db")
    mgr = SessionManager(conn)

    s1 = mgr.create_session("coding", title="Build CLI")
    s2 = mgr.create_session("research", title="Read papers")
    s3 = mgr.create_session("coding", title="Fix bug")

    sessions = mgr.list_sessions()
    assert len(sessions) == 3
    ids = {s.id for s in sessions}
    assert s1.id in ids
    assert s2.id in ids
    assert s3.id in ids


def test_add_and_get_messages(tmp_path: Path):
    conn = get_connection(tmp_path / "test.db")
    mgr = SessionManager(conn)

    session = mgr.create_session("coding")
    mgr.add_message(session.id, "user", "Hello")
    mgr.add_message(session.id, "assistant", "Hi there!")
    mgr.add_message(session.id, "user", "How are you?")

    messages = mgr.get_messages(session.id)
    assert len(messages) == 3
    assert messages[0].role == "user"
    assert messages[0].content == "Hello"
    assert messages[1].role == "assistant"
    assert messages[2].id == 3


def test_messages_isolated_between_sessions(tmp_path: Path):
    conn = get_connection(tmp_path / "test.db")
    mgr = SessionManager(conn)

    s1 = mgr.create_session("a")
    s2 = mgr.create_session("b")
    mgr.add_message(s1.id, "user", "msg for s1")
    mgr.add_message(s2.id, "user", "msg for s2")

    assert len(mgr.get_messages(s1.id)) == 1
    assert len(mgr.get_messages(s2.id)) == 1
    assert mgr.get_messages(s1.id)[0].content == "msg for s1"
