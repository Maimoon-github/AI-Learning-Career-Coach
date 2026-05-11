"""SQLite user profile store."""

# src/memory/long_term.py

import sqlite3
import json
from datetime import datetime
from src.state.schema import UserProfile


class LongTermMemory:
    def __init__(self, db_path: str = "./data/coach.db"):
        self.db_path = db_path
        self._init_schema()

    def _init_schema(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id TEXT PRIMARY KEY,
                    profile_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS session_summaries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    session_date TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_notes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    note TEXT NOT NULL,
                    topic TEXT,
                    created_at TEXT NOT NULL
                )
            """)
            conn.commit()

    def save_profile(self, profile: UserProfile) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO user_profiles (user_id, profile_json, updated_at)
                VALUES (?, ?, ?)
            """, (profile.user_id, profile.model_dump_json(), datetime.utcnow().isoformat()))
            conn.commit()

    def load_profile(self, user_id: str) -> UserProfile | None:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT profile_json FROM user_profiles WHERE user_id = ?", (user_id,)
            ).fetchone()
        if row:
            return UserProfile.model_validate_json(row[0])
        return None

    def save_session_summary(self, user_id: str, summary: str) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO session_summaries (user_id, summary, session_date)
                VALUES (?, ?, ?)
            """, (user_id, summary, datetime.utcnow().isoformat()))
            conn.commit()

    def save_note(self, user_id: str, note: str, topic: str = "") -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO user_notes (user_id, note, topic, created_at)
                VALUES (?, ?, ?, ?)
            """, (user_id, note, topic, datetime.utcnow().isoformat()))
            conn.commit()

    def get_notes(self, user_id: str, limit: int = 100) -> list[dict]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("""
                SELECT note, topic, created_at FROM user_notes
                WHERE user_id = ? ORDER BY created_at DESC LIMIT ?
            """, (user_id, limit)).fetchall()
        return [{"note": r[0], "topic": r[1], "created_at": r[2]} for r in rows]