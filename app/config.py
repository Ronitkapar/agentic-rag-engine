# Get the env related api keys and other settings from the .env file

import os
from dotenv import load_dotenv

load_dotenv()


def _bool_env(name: str, default: str = "false") -> bool:
    """Read a boolean-ish env var ('1', 'true', 'yes' are all truthy)."""
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes")


def _int_env(name: str, default: str) -> int:
    """Read an int env var, falling back to the default if unset or malformed."""
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return int(default)


def _optional_int_env(name: str):
    """Read an optional int env var. Returns None when unset or blank."""
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


class Settings:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    # Groq decommissioned the llama-3.* chat models; current OpenAI-source models:
    GROQ_MODEL = "openai/gpt-oss-120b"        # main RAG synthesis / planner
    GROQ_GUARD_MODEL = "openai/gpt-oss-20b"   # fast guardrail intent gate
    # Kill switch for the two extra detection layers in app/guardrails/rails.py
    # — the jailbreak regex pre-filter and the refusal classifier. Turn false to
    # fall back to plain NeMo substring matching if either misbehaves in
    # production, without waiting for a redeploy. NeMo itself always runs.
    GUARD_FALLBACK_ENABLED = _bool_env("GUARD_FALLBACK_ENABLED", "true")
    GROQ_SLUG = "rag-app"
    GROQ_SLUG_2 = "rag-app1"
    PORTKEY_API_KEY = os.getenv("PORTKEY_API_KEY")
    PORTKEY_CONFIG_SLUG = os.getenv("PORTKEY_CONFIG_SLUG")
    # Set USE_PORTKEY=true only when the saved Portkey config targets live models.
    USE_PORTKEY = os.getenv("USE_PORTKEY", "false").strip().lower() in ("1", "true", "yes")

    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    QDRANT_URL = os.getenv("QDRANT_CLUSTER_ENDPOINT")
    QDRANT_COLLECTION = "rag-app"

    # --- Embedding provider selection ---
    # "aicredits" → OpenAI-compatible gateway (api.aicredits.in)
    # "local"     → offline sentence-transformers (no API key required)
    EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "aicredits").strip().lower()

    # AICredits gateway — OpenAI-compatible. Its /v1/embeddings route serves ONLY
    # OpenAI embedding models (verified: text-embedding-3-large/small/ada-002).
    # Google/Gemini ids return 400 "invalid model ID" there despite the catalog.
    AICREDITS_API_KEY = os.getenv("AICREDITS_API_KEY")
    AICREDITS_BASE_URL = os.getenv("AICREDITS_BASE_URL", "https://api.aicredits.in/v1")
    AICREDITS_EMBEDDING_MODEL = os.getenv(
        "AICREDITS_EMBEDDING_MODEL", "text-embedding-3-large"
    )

    # Offline fallback — used when EMBEDDING_PROVIDER=local (or as last resort)
    EMBEDDING_LOCAL_MODEL = os.getenv("EMBEDDING_LOCAL_MODEL", "all-mpnet-base-v2")

    EMBEDDING_BATCH_SIZE = _int_env("EMBEDDING_BATCH_SIZE", "50")
    # When the selected cloud provider fails its probe, fall back to the local
    # model instead of hard-failing. The ingestion dim-guard still refuses to
    # write mismatched vectors into an existing collection.
    EMBEDDING_ALLOW_LOCAL_FALLBACK = _bool_env("EMBEDDING_ALLOW_LOCAL_FALLBACK", "true")
    # Optional MRL truncation — only sent when the provider honours `dimensions`.
    EMBEDDING_DIMENSIONS = _optional_int_env("EMBEDDING_DIMENSIONS")

    # --- Retrieval & reranking (Render-friendly defaults, all env-tunable) ---
    # Qdrant candidates fetched before reranking. Each candidate costs one
    # cross-encoder pass: measured ~1.7 core-seconds for 15 docs vs ~0.8 for 8 —
    # matters on the free tier's 0.1 vCPU.
    RETRIEVER_CANDIDATES = _int_env("RETRIEVER_CANDIDATES", "8")
    # Chunks kept after reranking and passed to the responder.
    RERANK_TOP_N = _int_env("RERANK_TOP_N", "5")
    # FlashRank tokenises every passage up to max_length. Measured peak RSS
    # deltas for a 15-doc batch: 512 -> +182 MB, 256 -> +44 MB, 128 -> +18 MB.
    # 256 keeps a full query well inside a 512 MB instance.
    RERANK_MAX_LENGTH = _int_env("RERANK_MAX_LENGTH", "256")
    # Set false to skip the cross-encoder entirely (saves its RAM + CPU).
    RERANK_ENABLED = _bool_env("RERANK_ENABLED", "true")
    # Model cache dir. The deploy image pre-bakes the model to /opt/flashrank;
    # /tmp/flashrank is the local-dev default (re-downloads ~3.3 MB if missing).
    RERANK_CACHE_DIR = os.getenv("RERANK_CACHE_DIR", "/tmp/flashrank")

settings = Settings()
