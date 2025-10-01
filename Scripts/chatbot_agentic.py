import os
import time
import sqlite3
import groq
import os
import sys
import nest_asyncio
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
nest_asyncio.apply()

import logging
logging.basicConfig(level=logging.INFO)
from Settings.Settings import Config 

logger = logging.getLogger(__name__)
# -----------------------------
# Memory DB
# -----------------------------
class ChatMemory:
    def __init__(self, db_path=":memory:"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.create_tables()

    def create_tables(self):
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT,
                content TEXT,
                timestamp REAL
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                summary TEXT,
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

    def get_history(self, limit=20):
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT role, content FROM memory ORDER BY id DESC LIMIT ?",
            (limit,)
        )
        rows = cursor.fetchall()
        return [{"role": r, "content": c} for r, c in reversed(rows)]

    def add_summary(self, summary):
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO summaries (summary, timestamp) VALUES (?, ?)",
            (summary, time.time())
        )
        self.conn.commit()

    def get_summaries(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT summary FROM summaries ORDER BY id ASC")
        rows = cursor.fetchall()
        return [r[0] for r in rows]

# -----------------------------
# Summarizer
# -----------------------------
class ChatSummarizer:
    def __init__(self, memory, api_key=None, model="llama-3.3-70b-versatile"):
        if api_key is None:
            api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("Groq API key is required (set GROQ_API_KEY)")
        self.client = groq.Client(api_key=api_key)
        self.model = model
        self.memory = memory

    def summarize(self, limit=20):
        history = self.memory.get_history(limit=limit)
        if not history:
            return None

        conversation = ""
        for h in history:
            conversation += f"{h['role'].capitalize()}: {h['content']}\n"

        prompt = (
            "Summarize the following conversation in the shortest possible way. "
            "Use 1-3 bullet points or one concise sentence. "
            "Keep only the key facts and decisions.\n\n"
            f"{conversation}"
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that makes very short summaries."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=80
        )
        summary = response.choices[0].message.content.strip()
        self.memory.add_summary(summary)
        return summary

# -----------------------------
# Chatbot
# -----------------------------
class Chatbot:
    def __init__(self, memory, api_key=None, model="llama-3.3-70b-versatile"):
        if api_key is None:
            api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("Groq API key is required (set GROQ_API_KEY)")
        self.client = groq.Client(api_key=api_key)
        self.model = model
        self.memory = memory

    def ask(self, message: str, use_summary=False):
        # Save user message
        self.memory.add_message("user", message)

        # Build context: last few messages
        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        if use_summary:
            summaries = self.memory.get_summaries()
            if summaries:
                messages.append({"role": "system", "content": f"Conversation summary so far: {summaries[-1]}"})
        messages += self.memory.get_history(limit=6)

        # Get Groq response
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7,
            max_tokens=200
        )
        reply = response.choices[0].message.content.strip()

        # Save assistant reply
        self.memory.add_message("assistant", reply)
        return reply

# -----------------------------
# Agent orchestrator
# -----------------------------
class AgenticAI:
    def __init__(self, use_summary=True, db_path=":memory:"):
        self.memory = ChatMemory(db_path=db_path)
        self.chatbot = Chatbot(self.memory)
        self.summarizer = ChatSummarizer(self.memory)
        self.use_summary = use_summary

    def run(self):
        print("Agentic AI Chat (type 'exit' to finish)\n")
        while True:
            user_input = input("You: ")
            if user_input.lower() in ("exit", "quit"):
                print("\n--- Chat Ended ---")
                break
            response = self.chatbot.ask(user_input, use_summary=self.use_summary)
            print("Bot:", response)

        # At end → summarize conversation
        print("\n--- Final Summary ---")
        summary = self.summarizer.summarize(limit=20)
        print(summary if summary else "No summary generated.")

        # Show stored memory
        print("\n--- Memory Stored ---")
        for h in self.memory.get_history(limit=50):
            print(f"{h['role'].capitalize()}: {h['content']}")

        # Show all summaries
        print("\n--- Stored Summaries ---")
        for s in self.memory.get_summaries():
            print("-", s)

# -----------------------------
# Run agent
# -----------------------------
if __name__ == "__main__":
    agent = AgenticAI(use_summary=True, db_path="chat_memory.db")
    agent.run()
