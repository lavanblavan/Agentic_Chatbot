from fileinput import filename
import groq
import os
import sys
import time
import json
import re
import threading
import asyncio
import nest_asyncio
import numpy as np
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
nest_asyncio.apply()

import logging
logging.basicConfig(level=logging.INFO)
from Settings.Settings import Config 

logger = logging.getLogger(__name__)
class summary_create:
    def __init__(self):
        self.api_key = Config.Groq_API_KEY
        self.client = groq.Client(api_key=self.api_key)
        self.model ="llama-3.3-70b-versatile"
        self.array_of_summaries = []
        

    
    def find_minititles(self,text):
        """"
        Find the sub titles which can be used for summarization in future"""
        try:
            prompt = (
                "You are a summary maker. Read the text below and break it into "
                "small, meaningful sections of summary for a page. try to get all the context into summary "
                "Each summary should be short, clear, and descriptive. "
                "Return only the numbered list of summary.\n\n"
                f"Text:\n{text}"
            )

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=300
            )

            reply = response.choices[0].message.content
            
            print("Subtitles found successfully.",reply)
            return reply

        except Exception as e:
            logger.error(f"Error in structured subtitle maker: {e}")
            return "Sorry, I couldn't generate subtitles."
    def put_summary(self,document_name,summary):
        """""
        put the summary of the document in json file
        
        """
        try:
            with open(document_name, 'w') as f:
                f.write(summary)
            print(f"Summary saved to {document_name}")
        except Exception as e:
            logger.error(f"Error saving array to file: {e}")
            print("Failed to save array data.")

        