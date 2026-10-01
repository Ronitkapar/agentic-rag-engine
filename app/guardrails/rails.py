import re

import logfire
from langchain_groq import ChatGroq
from nemoguardrails import RailsConfig, LLMRails

from app.config import settings
from app.guardrails.colang_rules import (
    COLANG_CONTENT,
    JAILBREAK_PATTERNS,
    JAILBREAK_REFUSAL,
    RAIL_INDICATORS,
    REFUSAL_MARKERS,
    YAML_CONTENT,
)


_rails: LLMRails | None = None

# Held so the refusal-classification fallback reuses the guard client instead of
# building a second one. The deploy image is budgeted to 512 MB; ChatGroq is
# cheap but there is no reason to pay for two.
_guard_llm: ChatGroq | None = None

_JAILBREAK_RE = tuple(re.compile(p, re.IGNORECASE) for p in JAILBREAK_PATTERNS)

# The classifier is asked a yes/no question and must answer with one word. Kept
# blunt on purpose: a hedged or chatty answer is treated as NO, so the fallback
# can only ever *propose* a block. A refusal is confirmed only on an explicit YES.
_CLASSIFIER_PROMPT = """You are a strict content classifier for a security gate.

An assistant was asked a question and produced this reply:

REPLY:
{content}

The assistant's remit is Kubernetes, Intel hardware, and enterprise networking.
Answer YES if the REPLY declines to answer — because the question is outside
that scope, or because complying would break the assistant's guidelines.
Answer NO if the REPLY actually attempts to answer a question about those
topics, even briefly or with caveats.

A greeting, an offer to help, or a substantive technical answer is NO.

Reply with exactly one word: YES or NO."""


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
    global _rails, _guard_llm

    guard_llm = ChatGroq(
        api_key=settings.GROQ_API_KEY,
        model=settings.GROQ_GUARD_MODEL,
        temperature=0
    )
    _guard_llm = guard_llm

    config = RailsConfig.from_content(
        colang_content=COLANG_CONTENT,
        yaml_content=_render_yaml_content()
    )

    _rails = LLMRails(config, llm=guard_llm)
    logfire.info(f"🛡️ NeMo Guardrails initialised ({settings.GROQ_GUARD_MODEL}).")


def looks_like_jailbreak(message: str) -> bool:
    """True if the message is an unambiguous instruction-override attempt.

    Deliberately narrow: these are phrasings where there is no reasonable
    reading that isn't an attack. A jailbreak that needs judgement to spot
    ("roleplay as a chef who explains syscalls") is a normal query here and
    goes to NeMo as usual.
    """
    return any(pattern.search(message) for pattern in _JAILBREAK_RE)


def _looks_like_refusal(content: str) -> bool:
    """Cheap, wording-agnostic pre-check for 'this reply declines to answer'.

    Never decisive on its own — see REFUSAL_MARKERS. A match only means the
    question is worth one classifier call.
    """
    lowered = content.lower()
    return any(marker in lowered for marker in REFUSAL_MARKERS)


def _llm_says_it_refused(content: str) -> bool:
    """Ask the fast guard model whether `content` is a refusal.

    Only reached when the configured bot-message substring check missed but the
    reply still reads like a refusal. Fails OPEN on error, matching the
    existing gate behaviour: a classifier outage must not block every query.
    """
    if _guard_llm is None:
        return False
    try:
        reply = _guard_llm.invoke(
            _CLASSIFIER_PROMPT.format(content=content[:2000])
        ).content.strip().upper()
    except Exception as e:
        logfire.warning(
            f"⚠️ Refusal classifier failed ({type(e).__name__}: {e}) — "
            "treating reply as a normal answer."
        )
        return False

    fired = reply.startswith("YES")
    logfire.info(
        f"🔎 Refusal classifier: {reply[:8]} (rejected as refusal: {not fired})"
    )
    return fired


def guard(message: str) -> tuple[bool, str | None]:
    """
    Run a user message through the NeMo rails gate.

    Three layers, cheapest first:

    1. A regex pre-filter for unambiguous instruction-override attempts. No LLM
       call, no NeMo round trip.
    2. NeMo's rails. "Fired" is decided by substring match against the
       configured `define bot` messages, which is exact and free.
    3. If no indicator matched but the reply still reads like a refusal, ask the
       fast guard model to confirm. This is the layer that catches a refusal
       NeMo wrote in its own words because the query never matched a
       `define user ...` flow.

    Returns:
        (True,  rail_response) — a rail fired; return this response immediately,
                                skip the RAG pipeline entirely.
        (False, None)          — message is clean; proceed to LangGraph.
    """
    if settings.GUARD_FALLBACK_ENABLED and looks_like_jailbreak(message):
        logfire.info(f"🚫 Jailbreak pattern matched | query='{message[:80]}'")
        return True, JAILBREAK_REFUSAL

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

        if any(indicator in content for indicator in RAIL_INDICATORS):
            logfire.info(f"🛡️ Guardrails fired | query='{message[:80]}'")
            return True, content

        # No configured bot message, but NeMo may have written its own refusal
        # after falling through to free generation. Cheap phrase check, then one
        # confirmation call. Without this, a reworded refusal reads as "passed"
        # and the query goes on to search Qdrant.
        if settings.GUARD_FALLBACK_ENABLED and _looks_like_refusal(content):
            if _llm_says_it_refused(content):
                logfire.info(
                    f"🛡️ Refusal confirmed by classifier | query='{message[:80]}'"
                )
                return True, content

        logfire.info("✅ Guardrails passed.")
        return False, None
