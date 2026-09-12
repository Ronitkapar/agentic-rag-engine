# Get the env related api keys and other settings from the .env file

import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_FALLBACK_API_KEY = os.getenv("GROQ_FALLBACK_API_KEY")
    # Groq decommissioned the llama-3.* chat models; current OpenAI-source models:
    GROQ_MODEL = "openai/gpt-oss-120b"        # main RAG synthesis / planner
    GROQ_GUARD_MODEL = "openai/gpt-oss-20b"   # fast guardrail intent gate
    GROQ_SLUG = "rag-app"
    GROQ_SLUG_2 = "rag-app1"
    PORTKEY_API_KEY = os.getenv("PORTKEY_API_KEY")
    PORTKEY_CONFIG_SLUG = os.getenv("PORTKEY_CONFIG_SLUG")
    # Set USE_PORTKEY=true only when the saved Portkey config targets live models.
    USE_PORTKEY = os.getenv("USE_PORTKEY", "false").strip().lower() in ("1", "true", "yes")

    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    QDRANT_URL = os.getenv("QDRANT_CLUSTER_ENDPOINT")
    QDRANT_COLLECTION = "rag-app"

    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

settings = Settings()
