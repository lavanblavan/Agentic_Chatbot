import sqlite3
import time
import os
import groq

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
        # Get the last N messages
        history = self.memory.get_history(limit=limit)
        if not history:
            return None

        # Build compact summarization request
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
            max_tokens=80   # smaller output size
        )
        summary = response.choices[0].message.content.strip()

        # Save summary to DB
        self.memory.add_summary(summary)
        return summary
