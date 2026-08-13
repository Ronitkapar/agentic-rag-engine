# Get the env related api keys and other settings from the .env file

import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_FALLBACK_API_KEY = os.getenv("GROQ_FALLBACK_API_KEY")
    GROQ_MODEL = "llama-3.3-70b-versatile"
    GROQ_SLUG = "rag-app"
    GROQ_SLUG_2 = "rag-app1"
    PORTKEY_API_KEY = os.getenv("PORTKEY_API_KEY")
    PORTKEY_CONFIG_SLUG = os.getenv("PORTKEY_CONFIG_SLUG")

    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    QDRANT_URL = os.getenv("QDRANT_CLUSTER_ENDPOINT")
    QDRANT_COLLECTION = "rag-app"

    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

settings = Settings()
