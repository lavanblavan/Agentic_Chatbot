import sqlite3
import time

class ChatMemory:
    def __init__(self, db_path=":memory:"):
        """
        db_path=":memory:" -> temporary DB in RAM
        db_path="chat_memory.db" -> persistent file-based DB
        """
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.create_table()

    def create_table(self):
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT,
                content TEXT,
                timestamp REAL
            )
        """)
        self.conn.commit()

    def add_message(self, role, content):
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO memory (role, content, timestamp) VALUES (?, ?, ?)",
            (role, content, time.time())
        )
        self.conn.commit()

    def get_history(self, limit=10):
        """
        Retrieve last N messages (default 10).
        Returns list of dicts [{role, content}, ...]
        """
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT role, content FROM memory ORDER BY id DESC LIMIT ?",
            (limit,)
        )
        rows = cursor.fetchall()
        return [{"role": r, "content": c} for r, c in reversed(rows)]

    def clear(self):
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM memory")
        self.conn.commit()
