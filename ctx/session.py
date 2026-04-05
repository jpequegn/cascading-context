import uuid
from dataclasses import dataclass
from datetime import datetime

import duckdb


@dataclass
class Message:
    id: int
    session_id: str
    role: str
    content: str
    created_at: datetime


@dataclass
class Session:
    id: str
    title: str
    created_at: datetime
    domain: str


class SessionManager:
    def __init__(self, conn: duckdb.DuckDBPyConnection) -> None:
        self._conn = conn

    def create_session(self, domain: str, title: str | None = None) -> Session:
        session_id = uuid.uuid4().hex[:12]
        now = datetime.now()
        t = title or f"Session {session_id[:6]}"
        self._conn.execute(
            "INSERT INTO sessions (id, title, created_at, domain) VALUES (?, ?, ?, ?)",
            [session_id, t, now, domain],
        )
        return Session(id=session_id, title=t, created_at=now, domain=domain)

    def add_message(self, session_id: str, role: str, content: str) -> Message:
        row = self._conn.execute(
            "SELECT COALESCE(MAX(id), 0) + 1 FROM messages WHERE session_id = ?",
            [session_id],
        ).fetchone()
        msg_id = row[0]
        now = datetime.now()
        self._conn.execute(
            "INSERT INTO messages (id, session_id, role, content, created_at) VALUES (?, ?, ?, ?, ?)",
            [msg_id, session_id, role, content, now],
        )
        return Message(id=msg_id, session_id=session_id, role=role, content=content, created_at=now)

    def get_messages(self, session_id: str) -> list[Message]:
        rows = self._conn.execute(
            "SELECT id, session_id, role, content, created_at FROM messages WHERE session_id = ? ORDER BY id",
            [session_id],
        ).fetchall()
        return [Message(*row) for row in rows]

    def list_sessions(self) -> list[Session]:
        rows = self._conn.execute(
            "SELECT id, title, created_at, domain FROM sessions ORDER BY created_at DESC"
        ).fetchall()
        return [Session(*row) for row in rows]
