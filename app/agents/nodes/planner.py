from app.agents.state import AgentState
from app.gateway import get_langchain_llm
from app.guardrails.colang_rules import REFUSAL_MARKERS
import logfire

# Lazy initialization - resolved on first call so that importing this module
# (and therefore app.main) never binds a Groq/Portkey key at import time.
# Mirrors _get_ranker() in ranking_service.py and _get_client() in
# qdrant_service.py.
_llm = None

# Shown when the client says it has earlier turns in this thread but the
# server has none. On Render's free tier the container sleeps after ~15 min
# idle, which kills the in-process MemorySaver — the loss is real, so we say
# so rather than answering as if the thread were new.
_MEMORY_LOST = (
    "⚠️ Earlier conversation memory is unavailable — the server restarted "
    "(free-tier sleep), so this answer has no prior context."
)

# Route target for "this is not a question worth searching". Distinct from
# CONVERSATIONAL, which means "answer from memory" — both skip the retriever,
# but this one carries the refusal as the final answer.
REFUSED = "REFUSED"

# A refined search query is a phrase. Anything longer is the model having
# answered the question inside the planner instead of handing back a query,
# which is how a refusal used to reach Qdrant as a search term.
_MAX_QUERY_CHARS = 300

_REFUSAL_FALLBACK = (
    "I'm an Enterprise IT Assistant focused on Kubernetes, Intel hardware, and "
    "networking — I can't help with that. But ask me anything technical!"
)


def _looks_like_refusal(text: str) -> bool:
    """True if free-form model output is a refusal rather than a search query.

    The planner is asked for a refined search query, but a model handed an
    off-topic question sometimes answers it instead — returning a refusal where
    a query was expected. That string then became the Qdrant search term, and
    the user got "I can't help" *after* a full retrieval and synthesis pass.
    """
    lowered = text.lower()
    return any(marker in lowered for marker in REFUSAL_MARKERS)


def _get_llm():
    """Resolve the planner LLM lazily, once per process."""
    global _llm
    if _llm is None:
        # Portkey-backed LLM: fallback + cache + retry — same .invoke()
        # interface as ChatGroq.
        _llm = get_langchain_llm(feature="planner")
    return _llm


def _client_has_prior_turns(config) -> bool:
    """
    True when the client reports earlier turns for this thread.

    The checkpointer lives in-process, so a restarted container cannot tell
    "never seen this thread" from "saw it, then lost it". The client does
    know — it renders the transcript — so it passes its own turn count through
    the runnable config. This is runtime config, not AgentState, so the state
    schema is unchanged.
    """
    if not config:
        return False
    configurable = config.get("configurable") or {}
    try:
        return int(configurable.get("client_turns") or 0) > 0
    except (TypeError, ValueError):
        return False


def planner_node(state: AgentState, config=None):
    """
    The Planner determines if a search is needed based on the ENTIRE conversation.
    """
    # Get the conversation history (excluding the latest message)
    history = ""
    for msg in state["messages"][:-1]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history += f"{role}: {msg['content']}\n"

    # No history in-state, but the client says this thread has earlier turns:
    # the server lost them (spin-down / restart). Surface it, don't hide it.
    memory_lost = not history and _client_has_prior_turns(config)

    user_message = state["messages"][-1]["content"] if state["messages"] else ""

    prompt = f"""
    You are an intelligent Assistant Planner.
    Analyze the conversation history and the latest user message.

    CONVERSATION HISTORY:
    {history}

    LATEST MESSAGE:
    "{user_message}"

    Task:
    1. If the latest message is a greeting (hi, hello) or a question that can be answered using ONLY the conversation history above (e.g., "what is my name"), respond with 'CONVERSATIONAL'.
    2. If it is a technical question about Kubernetes, Intel, or Networking that requires fresh documentation, output a refined search query.

    Output ONLY 'CONVERSATIONAL' or the search query.
    """

    with logfire.span("🧠 Planner Decision"):
        decision = _get_llm().invoke(prompt).content.strip()
        logfire.info(f"Intent identified: {decision}")

    # The planner was asked for a query, not an answer. When the gate lets an
    # off-topic question through, the model sometimes supplies the refusal
    # itself — and that string used to be handed straight to Qdrant as a
    # search term. Anything that isn't a usable query short-circuits instead.
    if decision != "CONVERSATIONAL" and (
        not decision or _looks_like_refusal(decision) or len(decision) > _MAX_QUERY_CHARS
    ):
        logfire.info("🛡️ Planner returned no usable query — skipping retrieval.")
        refusal = decision if _looks_like_refusal(decision) else _REFUSAL_FALLBACK
        return {
            "current_query": REFUSED,
            "status": "Blocked by guardrails.",
            "plan": ["Intent: Guardrails Fired", "Retrieval: Skipped"],
            "final_answer": refusal,
        }

    if decision == "CONVERSATIONAL":
        plan = ["Intent: Conversational/Memory", "Retrieval: Skipped"]
        status = "Handling conversationally (using memory)..."
        if memory_lost:
            plan.append(_MEMORY_LOST)
            status = _MEMORY_LOST
        return {
            "current_query": "CONVERSATIONAL",
            "status": status,
            "plan": plan,
        }

    plan = ["Intent: Technical", f"Search Term: {decision}"]
    status = f"Technical research needed. Searching for: {decision}"
    if memory_lost:
        plan.append(_MEMORY_LOST)
        status = f"{status} {_MEMORY_LOST}"
    return {
        "current_query": decision,
        "status": status,
        "plan": plan,
    }
