import os
from pathlib import Path

import duckdb

DEFAULT_DB_PATH = Path.home() / ".ctx" / "store.db"


def get_connection(db_path: Path | None = None) -> duckdb.DuckDBPyConnection:
    path = db_path or DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(path))
    _init_schema(conn)
    return conn


def _init_schema(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id VARCHAR PRIMARY KEY,
            title VARCHAR,
            created_at TIMESTAMP DEFAULT current_timestamp,
            domain VARCHAR
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER,
            session_id VARCHAR,
            role VARCHAR,
            content TEXT,
            created_at TIMESTAMP DEFAULT current_timestamp
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS facts (
            id INTEGER PRIMARY KEY,
            session_id VARCHAR,
            claim TEXT,
            category VARCHAR,
            confidence FLOAT,
            entities VARCHAR[],
            embedding FLOAT[],
            created_at TIMESTAMP DEFAULT current_timestamp
        )
    """)
    conn.execute("""
        CREATE SEQUENCE IF NOT EXISTS facts_id_seq START 1
    """)
