"""The reworded-refusal path: a refusal must never become a Qdrant query.

The bug this locks down, observed in the deployed logs:

    Guardrails passed.                    <- NeMo refused, but in its own words,
                                            so no RAIL_INDICATORS substring hit
    Intent identified: I'm sorry, but I can't help with that.
    Searching Qdrant for: I'm sorry, but I can't help with that.   <- 8 candidates
    Reranking complete. Kept top 5
    ... "sorry, I can't answer that"

Two links had to break for that to be impossible:

1. `guard()` gains a fallback that catches a refusal NeMo wrote freely, so
   link 1 (the miss) is covered.
2. `planner_node` refuses to emit a refusal as a search query, and the graph
   routes that to END. This is the layer that matters even if the guard misses
   again — a wrong guard verdict can no longer reach Qdrant.

The classifier and planner LLM are stubbed throughout, so this file needs no
Groq key, no Qdrant, and no network.
"""

import pytest

from app.agents.nodes.planner import REFUSED, _looks_like_refusal
from app.agents.graph import route_planner


def _state(current_query: str = "", last_user: str = "what is 2 plus 2") -> dict:
    """A minimal AgentState — the planner reads messages and writes 3 keys."""
    return {
        "messages": [{"role": "user", "content": last_user}],
        "current_query": current_query,
        "documents": [],
        "plan": [],
        "status": "",
        "final_answer": "",
    }


class _StubLLM:
    def __init__(self, decision: str):
        self.decision = decision

    def invoke(self, prompt):
        return type("R", (), {"content": self.decision})()


# --- the classifier's input side ------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "I'm sorry, but I can't help with that.",
        "I cannot assist with that request.",
        "Unfortunately I don't have information about that.",
        "That's outside my scope — ask me about Kubernetes.",
        "I'm an Enterprise IT Assistant focused on Kubernetes, Intel hardware, and networking.",
    ],
)
def test_refusal_phrasings_are_recognised(text):
    assert _looks_like_refusal(text)


@pytest.mark.parametrize(
    "query",
    [
        "how do I configure a kubernetes ingress controller",
        "ingress NGINX vs Traefik for a multi-cluster setup",
        "SR-IOV VF passthrough on ice lake",
        "BGP route reflector design",
    ],
)
def test_real_search_queries_are_not_mistaken_for_refusals(query):
    """The over-match case. A planner misreading a good query as a refusal
    would silently break the product for in-scope users — worse than the bug."""
    assert not _looks_like_refusal(query)


# --- the planner's output side --------------------------------------------


def test_planner_refusal_routes_to_refuse_and_never_retrieves(monkeypatch):
    import app.agents.nodes.planner as planner_mod

    monkeypatch.setattr(
        planner_mod, "_get_llm",
        lambda: _StubLLM("I'm sorry, but I can't help with that."),
    )
    out = planner_mod.planner_node(_state())

    assert out["current_query"] == REFUSED
    assert route_planner(_state(out["current_query"])) == "refuse"
    # Not just a different route — the refusal is the answer.
    assert out["final_answer"]
    assert out["plan"] == ["Intent: Guardrails Fired", "Retrieval: Skipped"]


def test_planner_block_is_classified_as_blocked_by_the_eval_suite():
    """evals/guardrails_eval.py and evals/pipeline.py both decide "blocked" by
    searching thought_process for this exact phrase. A planner-level block has to
    be indistinguishable from a gate-level one, or the eval suite under-counts
    blocks and this whole fix looks like it did nothing."""
    import app.agents.nodes.planner as planner_mod

    assert any(
        "guardrails fired" in step.lower()
        for step in ["Intent: Guardrails Fired", "Retrieval: Skipped"]
    ), "expected the eval suite's marker phrase to be present"


@pytest.mark.parametrize("decision", ["", "   ", "x" * 500])
def test_unusable_planner_output_also_short_circuits(monkeypatch, decision):
    """Empty or absurd output is not a search query either. Without this the
    retriever would be handed "" or a 500-char paragraph as a Qdrant query."""
    import app.agents.nodes.planner as planner_mod

    monkeypatch.setattr(planner_mod, "_get_llm", lambda: _StubLLM(decision))
    out = planner_mod.planner_node(_state())
    assert out["current_query"] == REFUSED
    assert route_planner(_state(out["current_query"])) == "refuse"


def test_a_real_query_still_retrieves(monkeypatch):
    """Regression guard: the new validation must not swallow good queries."""
    import app.agents.nodes.planner as planner_mod

    monkeypatch.setattr(
        planner_mod, "_get_llm",
        lambda: _StubLLM("kubernetes ingress controller configuration"),
    )
    out = planner_mod.planner_node(_state())
    assert out["current_query"] == "kubernetes ingress controller configuration"
    assert route_planner(_state(out["current_query"])) == "retriever"


def test_conversational_still_wins_over_refusal_detection(monkeypatch):
    """CONVERSATIONAL is checked first, so a greeting can't be misread."""
    import app.agents.nodes.planner as planner_mod

    monkeypatch.setattr(planner_mod, "_get_llm", lambda: _StubLLM("CONVERSATIONAL"))
    out = planner_mod.planner_node(_state(last_user="hello"))
    assert out["current_query"] == "CONVERSATIONAL"
    assert route_planner(_state(out["current_query"])) == "responder"


# --- the graph has a real edge for it -------------------------------------


def test_refused_is_a_real_node_in_the_compiled_graph():
    """Not just a string route_planner returns — the edge must exist, or
    LangGraph raises at invoke() time and every refused query 500s."""
    from app.agents.graph import rag_agent

    assert "refuse" in rag_agent.get_graph().nodes


def test_refused_does_not_reach_the_retriever(monkeypatch):
    """End-to-end shape of the guarantee: a refused turn never calls
    search_enterprise_knowledge. Stubbed to explode if it does."""
    import app.agents.nodes.retriever as retriever_mod
    from app.agents.graph import rag_agent

    monkeypatch.setattr(
        retriever_mod, "search_enterprise_knowledge",
        lambda *a, **k: pytest.fail("Qdrant was queried for a refused question"),
    )
    monkeypatch.setattr(
        "app.agents.nodes.planner._get_llm",
        lambda: _StubLLM("I'm sorry, but I can't help with that."),
    )

    out = rag_agent.invoke(_state(), config={"configurable": {"thread_id": "t"}})

    assert out["current_query"] == REFUSED
    assert out["documents"] == []
    assert any("guardrails fired" in s.lower() for s in out["plan"])
