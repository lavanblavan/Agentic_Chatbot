import groq
import os
import sys
import time
import json
import re
import threading
import asyncio
import nest_asyncio
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
nest_asyncio.apply()

import logging
logging.basicConfig(level=logging.INFO)
from Settings.Settings import Config 

logger = logging.getLogger(__name__)

class Chatbot:
    def __init__(self):
        self.api_key = Config.Groq_API_KEY
        self.client = groq.Client(api_key=self.api_key)
        # You can change model to another available one
        self.model ="llama-3.3-70b-versatile"

    def ask(self, message: str) -> str:
        """
        Send a message to Groq API and return response
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": message}],
                temperature=0.7,
                max_tokens=200
            )
            # Extract text
            reply = response.choices[0].message.content
            return reply
        except Exception as e:
            logger.error(f"Error in chatbot: {e}")
            return "Sorry, I couldn't process that."

if __name__ == "__main__":
    bot = Chatbot()
    print("Simple Chatbot (type 'quit' to exit)")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit"]:
            print("Exiting...")
            break
        reply = bot.ask(user_input)
        print("Bot:", reply)
