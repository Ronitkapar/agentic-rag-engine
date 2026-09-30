import logfire
from langchain_groq import ChatGroq
from nemoguardrails import RailsConfig, LLMRails

from app.config import settings
from app.guardrails.colang_rules import COLANG_CONTENT, YAML_CONTENT, RAIL_INDICATORS


_rails: LLMRails | None = None


def _render_yaml_content() -> str:
    """Substitute runtime secrets into the YAML template.

    The `type: embeddings` block routes NeMo's user-message index (intent
    classification over the `define user` examples) through the AICredits
    gateway. Without it NeMo loads a local ~305 MB fastembed ONNX model on the
    first request — fatal on Render's 512 MB free tier.
    """
    return (
        YAML_CONTENT
        .replace("{{AICREDITS_BASE_URL}}", settings.AICREDITS_BASE_URL or "")
        .replace("{{AICREDITS_API_KEY}}", settings.AICREDITS_API_KEY or "")
        .replace(
            "{{AICREDITS_EMBEDDING_MODEL}}",
            settings.AICREDITS_EMBEDDING_MODEL or "",
        )
    )


def initialize_rails() -> None:
    """
    Build the NeMo LLMRails singleton at app startup.
    Uses the fast guard model (openai/gpt-oss-20b) for intent classification
    at the gate — the heavier openai/gpt-oss-120b is reserved for the RAG pipeline.
    """
    global _rails

    guard_llm = ChatGroq(
        api_key=settings.GROQ_API_KEY,
        model=settings.GROQ_GUARD_MODEL,
        temperature=0
    )

    config = RailsConfig.from_content(
        colang_content=COLANG_CONTENT,
        yaml_content=_render_yaml_content()
    )

    _rails = LLMRails(config, llm=guard_llm)
    logfire.info(f"🛡️ NeMo Guardrails initialised ({settings.GROQ_GUARD_MODEL}).")
    
    


def guard(message: str) -> tuple[bool, str | None]:
    """
    Run a user message through the NeMo rails gate.

    Returns:
        (True,  rail_response) — a rail fired; return this response immediately,
                                skip the RAG pipeline entirely.
        (False, None)          — message is clean; proceed to LangGraph.
    """
    if _rails is None:
        logfire.warning("⚠️ Guardrails not initialised — skipping gate.")
        return False, None

    with logfire.span("🛡️ Guardrails Check"):
        try:
            result = _rails.generate(messages=[{"role": "user", "content": message}])
        except Exception as e:
            # Availability first: if the gate itself errors (e.g. the AICredits
            # embedding index provider is down), let the request continue with a
            # loud warning instead of failing the whole query. Retrieval and
            # synthesis are unaffected by the gate.
            logfire.warning(
                f"⚠️ Guardrail gate failed ({type(e).__name__}: {e}) — allowing request unguarded."
            )
            return False, None

        # NeMo returns {'role': 'assistant', 'content': '...'} — extract text
        content = result.get("content", "") if isinstance(result, dict) else str(result)

        fired = any(indicator in content for indicator in RAIL_INDICATORS)

        if fired:
            logfire.info(f"🛡️ Guardrails fired | query='{message[:80]}'")
            return True, content

        logfire.info("✅ Guardrails passed.")
        return False, None
