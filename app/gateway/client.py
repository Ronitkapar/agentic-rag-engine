import logfire
from langchain_groq import ChatGroq
from app.config import settings

# Portkey is OPTIONAL. It is only used when USE_PORTKEY=true AND both
# PORTKEY_API_KEY and PORTKEY_CONFIG_SLUG are set. Otherwise we call Groq
# directly so the app keeps working without the gateway.
_portkey_client = None
_portkey_enabled = False

if settings.USE_PORTKEY and settings.PORTKEY_API_KEY and settings.PORTKEY_CONFIG_SLUG:
    try:
        from portkey_ai import Portkey, createHeaders, PORTKEY_GATEWAY_URL
        from langchain_openai import ChatOpenAI

        _portkey_client = Portkey(
            api_key=settings.PORTKEY_API_KEY,
            config=settings.PORTKEY_CONFIG_SLUG,
        )
        _portkey_enabled = True
        logfire.info("🌐 Portkey gateway enabled.")
    except Exception as e:  # pragma: no cover - defensive
        logfire.warning(f"⚠️ Portkey unavailable ({e}) — using direct Groq instead.")
else:
    logfire.info("🔌 Portkey disabled — using direct Groq.")

portkey_client = _portkey_client  # kept for backwards compatibility


def _get_groq_llm(feature: str) -> ChatGroq:
    """Direct Groq chat model using the current (non-decommissioned) model."""
    return ChatGroq(
        api_key=settings.GROQ_API_KEY,
        model=settings.GROQ_MODEL,
        temperature=0,
    )


def get_langchain_llm(feature: str = "rag-app"):
    """
    Returns a LangChain chat model.

    Preferred path: Portkey-backed ChatOpenAI (routing, fallbacks, caching).
    Fallback path:   direct ChatGroq when Portkey is not enabled.
    Both expose the same .invoke() interface, so nodes don't care which one.
    """
    if _portkey_enabled:
        portkey_llm = ChatOpenAI(
            api_key=settings.PORTKEY_API_KEY,
            base_url=PORTKEY_GATEWAY_URL,
            model=f"@{settings.GROQ_SLUG}/{settings.GROQ_MODEL}",
            temperature=0,
            default_headers=createHeaders(
                api_key=settings.PORTKEY_API_KEY,
                config=settings.PORTKEY_CONFIG_SLUG,
                metadata={
                    "feature": feature,
                    "_user": "rag-system",
                    "environment": "production",
                },
            ),
        )
        # Auto-fallback to direct Groq if the gateway errors (e.g. stale config)
        return portkey_llm.with_fallbacks([_get_groq_llm(feature)])
    return _get_groq_llm(feature)


def generate_completion(messages: list[dict], temperature: float = 0.1) -> tuple[str, str]:
    """
    Unified one-shot chat completion used by the responder node.
    Returns (content, cache_status). cache_status is 'MISS' off Portkey.
    """
    if _portkey_enabled:
        try:
            response = _portkey_client.chat.completions.create(
                messages=messages,
                temperature=temperature,
            )
            content = response.choices[0].message.content
            return content, extract_cache_status(response)
        except Exception as e:
            logfire.warning(f"⚠️ Portkey call failed ({e}) — falling back to direct Groq.")

    llm = ChatGroq(
        api_key=settings.GROQ_API_KEY,
        model=settings.GROQ_MODEL,
        temperature=temperature,
    )
    content = llm.invoke(messages).content
    return content, "MISS"


def extract_cache_status(response) -> str:
    """
    Pull x-portkey-cache-status from the Portkey native client response headers.
    Tries multiple attribute paths defensively — returns 'MISS' if not found.
    Always 'MISS' when Portkey is disabled.
    """
    if not _portkey_enabled:
        return "MISS"
    for attr in ("_raw_response", "_response", "_http_response"):
        raw = getattr(response, attr, None)
        if raw is not None:
            status = getattr(raw, "headers", {}).get("x-portkey-cache-status", "")
            if status:
                return status.upper()
    return "MISS"