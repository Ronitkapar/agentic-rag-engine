# Embedding backend with runtime provider selection.
#
# EMBEDDING_PROVIDER chooses the backend:
#   "aicredits" -> OpenAI-compatible gateway (api.aicredits.in)  [default]
#   "local"     -> offline sentence-transformers (no API key needed)
#
# The vector dimension is *probed* from the live provider on first use instead
# of being hard-coded, so changing embedding models is an env-var change only.
# Consumers keep calling the same three helpers:
#   embed_query(), embed_texts(), get_embedding_dim()

import time

import logfire

from app.config import settings

_VALID_PROVIDERS = ("aicredits", "local")
_CLOUD_PROVIDERS = ("aicredits",)

_RETRYABLE_STATUS = (429, 500, 502, 503, 504)
_BATCH_ERROR_HINTS = (
    "batch",
    "too many",
    "maximum",
    "max ",
    "array",
    "payload",
    "input",
    "length",
    "413",
)

_active_model = None
_model_type: str | None = None  # "aicredits" | "local"
_active_model_id: str | None = None
_active_dim: int | None = None
_last_probe_error: str | None = None


#  Provider builders (imported lazily so an unused SDK is never required)


def _build_aicredits():
    """OpenAI-compatible embeddings against the AICredits gateway."""
    if not settings.AICREDITS_API_KEY:
        raise RuntimeError(
            "AICREDITS_API_KEY is not set. Add it to .env, or set "
            "EMBEDDING_PROVIDER=local to use the offline model."
        )
    from langchain_openai import OpenAIEmbeddings

    model = OpenAIEmbeddings(
        model=settings.AICREDITS_EMBEDDING_MODEL,
        base_url=settings.AICREDITS_BASE_URL,
        api_key=settings.AICREDITS_API_KEY,
        # Send raw strings instead of tiktoken token ids: AICredits is not an
        # OpenAI backend, and langchain's default tokenization silently falls
        # back to cl100k_base for model names it doesn't recognise.
        check_embedding_ctx_length=False,
        # Only applied when EMBEDDING_DIMENSIONS is set (MRL truncation).
        dimensions=settings.EMBEDDING_DIMENSIONS,
    )
    return model, settings.AICREDITS_EMBEDDING_MODEL


def _build_local():
    """Offline sentence-transformers fallback — no network, no API key."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        # Slim deploy builds (Render image) intentionally omit the ~3.8 GB of
        # torch/CUDA this would pull in. Fail with an actionable message instead
        # of a bare ImportError inside the fallback path.
        raise RuntimeError(
            "EMBEDDING_PROVIDER=local requires sentence-transformers, which is not "
            "installed in this build. Use EMBEDDING_PROVIDER=aicredits, or install "
            "the full requirements.txt (not requirements-deploy.txt)."
        ) from e

    logfire.info(
        f"Loading sentence-transformers ({settings.EMBEDDING_LOCAL_MODEL}) — offline fallback."
    )
    model = SentenceTransformer(settings.EMBEDDING_LOCAL_MODEL)
    return model, settings.EMBEDDING_LOCAL_MODEL


def _probe_cloud():
    """Build the AICredits client and run one embed call. Returns (model, model_id, dim)."""
    model, model_id = _build_aicredits()
    dim = len(model.embed_query("probe"))
    return model, model_id, dim


def _probe_local(model) -> int:
    """Resolve the local model's vector dimension."""
    try:
        return int(model.get_sentence_embedding_dimension())
    except Exception:  # defensive — older sentence-transformers builds
        return len(model.encode(["probe"])[0])


#  Initialisation (once per process, lazy)


def _init():
    """Initialise the embedding model once per process. Called lazily on first use."""
    global _active_model, _model_type, _active_model_id, _active_dim, _last_probe_error
    if _active_model is not None:
        return

    provider = settings.EMBEDDING_PROVIDER
    if provider not in _VALID_PROVIDERS:
        raise ValueError(
            f"EMBEDDING_PROVIDER='{provider}' is invalid. "
            f"Use one of: {', '.join(_VALID_PROVIDERS)}."
        )

    if provider in _CLOUD_PROVIDERS:
        try:
            model, model_id, dim = _probe_cloud()
            _active_model = model
            _model_type = provider
            _active_model_id = model_id
            _active_dim = dim
            logfire.info(f"✅ {provider} embeddings ready ({model_id}, {dim}-dim).")
            return
        except Exception as e:
            _last_probe_error = str(e)
            if not settings.EMBEDDING_ALLOW_LOCAL_FALLBACK:
                raise RuntimeError(
                    f"Embedding provider '{provider}' failed its probe: {e}"
                ) from e
            logfire.warning(
                f"⚠️ {provider} embedding probe failed: {e} — falling back to "
                f"local '{settings.EMBEDDING_LOCAL_MODEL}'."
            )

    model, model_id = _build_local()
    _active_model = model
    _model_type = "local"
    _active_model_id = model_id
    _active_dim = _probe_local(model)
    logfire.info(f"✅ local embeddings ready ({model_id}, {_active_dim}-dim).")


def active_embedding_info() -> dict:
    """Describe the active backend — used for logging and index fingerprints."""
    _init()
    return {
        "provider": _model_type,
        "model": _active_model_id,
        "dim": _active_dim,
        "fallback": bool(_last_probe_error) and _model_type == "local",
        "requested_provider": settings.EMBEDDING_PROVIDER,
        "probe_error": _last_probe_error,
    }


def get_embedding_dim() -> int:
    """Vector dimension of the active model (probed from the provider, never hard-coded)."""
    _init()
    return _active_dim


#  Error classification for retries


def _is_retryable(error: Exception) -> bool:
    """True for rate limits and transient transport / server errors."""
    try:
        import openai
    except ImportError:  # pragma: no cover - openai ships with langchain-openai
        openai = None

    if openai is not None:
        if isinstance(
            error,
            (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError),
        ):
            return True
        if isinstance(error, openai.APIStatusError):
            return error.status_code in _RETRYABLE_STATUS

    # Google SDK and generic gateways: match on the message instead.
    text = str(error).lower()
    return any(
        hint in text
        for hint in (
            "429",
            "rate limit",
            "quota",
            "resource_exhausted",
            "timeout",
            "unavailable",
            "502",
            "503",
            "504",
        )
    )


def _is_batch_error(error: Exception) -> bool:
    """True when the request was rejected because the input array was too large."""
    status = getattr(error, "status_code", None)
    if status == 413:
        return True
    if status != 400:
        return False
    text = str(error).lower()
    return any(hint in text for hint in _BATCH_ERROR_HINTS)


#  Batch embedding with retry / auto-split


def _embed_batch(batch: list[str]) -> list[list[float]]:
    if _model_type == "local":
        return _active_model.encode(batch, show_progress_bar=False).tolist()

    # Exponential backoff: 1s → 2s → 4s (4 attempts total)
    for attempt in range(4):
        try:
            return _active_model.embed_documents(batch)
        except Exception as e:
            if _is_retryable(e) and attempt < 3:
                wait = 2 ** attempt
                logfire.warning(
                    f"{_model_type} rate limit / transient error — retrying in {wait}s "
                    f"(attempt {attempt + 1}/4)."
                )
                time.sleep(wait)
                continue
            if _is_batch_error(e) and len(batch) > 1:
                # Gateway caps the input array — halve it and retry recursively.
                mid = len(batch) // 2
                logfire.warning(
                    f"{_model_type} rejected a batch of {len(batch)} — "
                    f"splitting into 2 × {mid}."
                )
                return _embed_batch(batch[:mid]) + _embed_batch(batch[mid:])
            logfire.error(f"{_model_type} embedding failed: {e}")
            raise
    raise RuntimeError(f"{_model_type} embedding failed after 4 attempts.")


#  Public API (stable signatures — consumers depend on these)


def embed_query(query: str) -> list[float]:
    _init()
    if _model_type == "local":
        return _active_model.encode([query])[0].tolist()
    return _active_model.embed_query(query)


def embed_texts(texts: list[str]) -> list[list[float]]:
    _init()
    if not texts:
        return []

    all_embeddings: list[list[float]] = []
    batch_size = max(1, settings.EMBEDDING_BATCH_SIZE)
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        with logfire.span(
            "Embed batch",
            model=_active_model_id,
            provider=_model_type,
            start=i,
            size=len(batch),
        ):
            all_embeddings.extend(_embed_batch(batch))
    return all_embeddings
