from dotenv import load_dotenv
import os
load_dotenv()
class Config:
    Groq_API_KEY = os.getenv("Groq_API_KEY")
    if Groq_API_KEY is None:
        raise ValueError("Groq_API_KEY not found in environment variables.")
    documents_folder = r"C:\Users\Lavan\Desktop\Chatbot\Document_Summarizer\Documents"  # Folder containing document subfolders
    
    # print(Groq_API_KEY)
