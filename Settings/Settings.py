from dotenv import load_dotenv
import os
load_dotenv()
class Config:
    Groq_API_KEY = os.getenv("Groq_API_KEY")
    if Groq_API_KEY is None:
        raise ValueError("Groq_API_KEY not found in environment variables.")
    # print(Groq_API_KEY)
