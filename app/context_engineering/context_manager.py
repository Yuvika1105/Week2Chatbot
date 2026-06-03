import sqlite3
import os
from pathlib import Path
from config import Config

class ConversationContextManager:
    def __init__(self, user_id: str = "default", session_id: str = "default_session"):
        self.user_id = user_id
        self.session_id = session_id
        self._init_db()

    def _init_db(self):
        db_path = Config.MEMORY_DB_PATH
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        conn = sqlite3.connect(db_path)
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chat_turns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    user_msg TEXT NOT NULL,
                    ai_res TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Dynamically migrate table if session_id is missing
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(chat_turns)")
            columns = [info[1] for info in cursor.fetchall()]
            if "session_id" not in columns:
                conn.execute("ALTER TABLE chat_turns ADD COLUMN session_id TEXT NOT NULL DEFAULT 'default_session'")
            
            # Create chat_sessions table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
        finally:
            conn.close()

    def add_turn(self, user_msg: str, ai_res: str, session_id: str = None) -> None:
        self._init_db()
        db_path = Config.MEMORY_DB_PATH
        conn = sqlite3.connect(db_path)
        sid = session_id or self.session_id
        try:
            conn.execute(
                "INSERT INTO chat_turns (user_id, session_id, user_msg, ai_res) VALUES (?, ?, ?, ?)",
                (self.user_id, sid, user_msg, ai_res)
            )
            # Create session if it does not exist
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM chat_sessions WHERE session_id = ?", (sid,))
            if not cursor.fetchone():
                title = user_msg[:30] + "..." if len(user_msg) > 30 else user_msg
                conn.execute(
                    "INSERT INTO chat_sessions (session_id, user_id, title) VALUES (?, ?, ?)",
                    (sid, self.user_id, title)
                )
            else:
                # Update title if it's default or placeholder
                cursor.execute("SELECT title FROM chat_sessions WHERE session_id = ?", (sid,))
                row = cursor.fetchone()
                if row and (row[0].startswith("New Chat") or row[0] == "default_session"):
                    title = user_msg[:30] + "..." if len(user_msg) > 30 else user_msg
                    conn.execute(
                        "UPDATE chat_sessions SET title = ? WHERE session_id = ?",
                        (title, sid)
                    )
            conn.commit()
        finally:
            conn.close()

    def get_turns(self, session_id: str = None) -> list:
        self._init_db()
        db_path = Config.MEMORY_DB_PATH
        conn = sqlite3.connect(db_path)
        sid = session_id or self.session_id
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT user_msg, ai_res FROM chat_turns WHERE user_id = ? AND session_id = ? ORDER BY id ASC",
                (self.user_id, sid)
            )
            rows = cursor.fetchall()
            return [{"user": row[0], "ai": row[1]} for row in rows]
        finally:
            conn.close()

    def get_window(self, max_tokens: int = 800, session_id: str = None) -> list:
        sid = session_id or self.session_id
        turns = self.get_turns(session_id=sid)
        max_chars = max_tokens * 4
        current_chars = 0
        selected = []
        for turn in reversed(turns):
            turn_str = f"User: {turn['user']}\nAI: {turn['ai']}\n"
            if current_chars + len(turn_str) > max_chars:
                break
            selected.insert(0, turn)
            current_chars += len(turn_str)
        return selected

    def clear(self, session_id: str = None) -> None:
        self._init_db()
        db_path = Config.MEMORY_DB_PATH
        conn = sqlite3.connect(db_path)
        sid = session_id or self.session_id
        try:
            conn.execute("DELETE FROM chat_turns WHERE user_id = ? AND session_id = ?", (self.user_id, sid))
            conn.execute("DELETE FROM chat_sessions WHERE session_id = ? AND user_id = ?", (sid, self.user_id))
            conn.commit()
        finally:
            conn.close()

    def get_sessions(self) -> list:
        self._init_db()
        db_path = Config.MEMORY_DB_PATH
        conn = sqlite3.connect(db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT s.session_id, s.title, s.created_at,
                       COALESCE(MAX(t.timestamp), s.created_at) as last_activity
                FROM chat_sessions s
                LEFT JOIN chat_turns t ON s.session_id = t.session_id
                WHERE s.user_id = ?
                GROUP BY s.session_id
                ORDER BY last_activity DESC
            """, (self.user_id,))
            rows = cursor.fetchall()
            return [{"session_id": row[0], "title": row[1], "created_at": row[2]} for row in rows]
        finally:
            conn.close()

    def create_session(self, session_id: str, title: str) -> None:
        self._init_db()
        db_path = Config.MEMORY_DB_PATH
        conn = sqlite3.connect(db_path)
        try:
            conn.execute(
                "INSERT OR REPLACE INTO chat_sessions (session_id, user_id, title) VALUES (?, ?, ?)",
                (session_id, self.user_id, title)
            )
            conn.commit()
        finally:
            conn.close()

    def delete_session(self, session_id: str) -> None:
        self._init_db()
        db_path = Config.MEMORY_DB_PATH
        conn = sqlite3.connect(db_path)
        try:
            conn.execute("DELETE FROM chat_turns WHERE user_id = ? AND session_id = ?", (self.user_id, session_id))
            conn.execute("DELETE FROM chat_sessions WHERE session_id = ? AND user_id = ?", (session_id, self.user_id))
            conn.commit()
        finally:
            conn.close()

    def get_last_active_session(self) -> str | None:
        sessions = self.get_sessions()
        if sessions:
            return sessions[0]["session_id"]
        return None