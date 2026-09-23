"""
Local transcript + summary storage, with full-text search.

Everything lives in a single SQLite file on disk — no cloud sync, no
external service. FTS5 gives free-text search ("what did we say about
pricing last Tuesday") without needing an embedding model, keeping the NPU
budget free for ASR + summarization.
"""

import sqlite3
import time
from pathlib import Path


SCHEMA = """
CREATE TABLE IF NOT EXISTS segments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    speaker TEXT NOT NULL,
    text TEXT NOT NULL,
    timestamp REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS summaries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    summary_text TEXT NOT NULL,
    timestamp REAL NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS segments_fts USING fts5(
    text, speaker, session_id, content='segments', content_rowid='id'
);

CREATE TRIGGER IF NOT EXISTS segments_ai AFTER INSERT ON segments BEGIN
  INSERT INTO segments_fts(rowid, text, speaker, session_id)
  VALUES (new.id, new.text, new.speaker, new.session_id);
END;
"""


class TranscriptStore:
    def __init__(self, db_path: str):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def add_segment(self, session_id: str, speaker: str, text: str):
        self._conn.execute(
            "INSERT INTO segments (session_id, speaker, text, timestamp) "
            "VALUES (?, ?, ?, ?)",
            (session_id, speaker, text, time.time()),
        )
        self._conn.commit()

    def add_summary(self, session_id: str, summary_text: str):
        self._conn.execute(
            "INSERT INTO summaries (session_id, summary_text, timestamp) "
            "VALUES (?, ?, ?)",
            (session_id, summary_text, time.time()),
        )
        self._conn.commit()

    def get_transcript(self, session_id: str) -> str:
        rows = self._conn.execute(
            "SELECT speaker, text FROM segments WHERE session_id = ? "
            "ORDER BY timestamp ASC",
            (session_id,),
        ).fetchall()
        return "\n".join(f"{speaker}: {text}" for speaker, text in rows)

    def search(self, query: str, limit: int = 20):
        return self._conn.execute(
            "SELECT session_id, speaker, text FROM segments_fts "
            "WHERE segments_fts MATCH ? LIMIT ?",
            (query, limit),
        ).fetchall()

    def close(self):
        self._conn.close()
