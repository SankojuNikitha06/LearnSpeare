"""
User-specific history for Text, Code, Audio, and Visual explanations.
Stored in SQLite (storage/history.db).
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# DB path set by app
_history_db: Optional[str] = None


def init_history(db_path: str) -> None:
    """Set database path and create table if needed."""
    global _history_db
    _history_db = db_path
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                type TEXT NOT NULL,
                topic TEXT NOT NULL,
                created_at TEXT NOT NULL,
                payload TEXT NOT NULL
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_history_user_type ON history(user_id, type)"
        )
        conn.commit()
    finally:
        conn.close()


def add_record(
    user_id: str,
    type_: str,
    topic: str,
    payload: Dict[str, Any],
) -> None:
    """Append one history record. user_id must be non-empty."""
    if not (user_id or "").strip():
        return
    if _history_db is None:
        return
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    conn = sqlite3.connect(_history_db)
    try:
        conn.execute(
            "INSERT INTO history (user_id, type, topic, created_at, payload) VALUES (?, ?, ?, ?, ?)",
            (user_id.strip(), type_.strip(), (topic or "").strip(), now, json.dumps(payload)),
        )
        conn.commit()
    finally:
        conn.close()


def get_history(
    user_id: str,
    type_filter: Optional[str] = None,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    """Return history rows for user, newest first. type_filter: 'text'|'code'|'audio'|'images' or None for all."""
    if not (user_id or "").strip():
        return []
    if _history_db is None:
        return []
    conn = sqlite3.connect(_history_db)
    try:
        if type_filter and type_filter.strip():
            rows = conn.execute(
                "SELECT id, type, topic, created_at, payload FROM history WHERE user_id = ? AND type = ? ORDER BY id DESC LIMIT ?",
                (user_id.strip(), type_filter.strip(), limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, type, topic, created_at, payload FROM history WHERE user_id = ? ORDER BY id DESC LIMIT ?",
                (user_id.strip(), limit),
            ).fetchall()
    except Exception:
        rows = []
    finally:
        conn.close()

    out = []
    for r in rows:
        id_, type_, topic, created_at, payload_str = r
        try:
            payload = json.loads(payload_str) if payload_str else {}
        except Exception:
            payload = {}
        out.append({
            "id": id_,
            "type": type_,
            "topic": topic,
            "created_at": created_at,
            "payload": payload,
        })
    return out


def delete_record(user_id: str, record_id: int) -> bool:
    """Delete one history record if it belongs to user. Returns True if a row was deleted."""
    if not (user_id or "").strip() or record_id is None:
        return False
    if _history_db is None:
        return False
    conn = sqlite3.connect(_history_db)
    try:
        cur = conn.execute(
            "DELETE FROM history WHERE user_id = ? AND id = ?",
            (user_id.strip(), record_id),
        )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def clear_all(user_id: str) -> int:
    """Delete all history records for user. Returns number of rows deleted."""
    if not (user_id or "").strip():
        return 0
    if _history_db is None:
        return 0
    conn = sqlite3.connect(_history_db)
    try:
        cur = conn.execute("DELETE FROM history WHERE user_id = ?", (user_id.strip(),))
        conn.commit()
        return cur.rowcount
    finally:
        conn.close()
