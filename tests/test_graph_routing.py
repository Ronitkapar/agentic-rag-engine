"""Agent graph routing invariants.

route_planner is the only place the CONVERSATIONAL sentinel is interpreted. The
planner node returns that literal string to mean "don't search, just answer from
memory"; if the sentinel in planner.py and the comparison in graph.py ever drift
apart, a purely conversational turn falls through to the retriever and the
assistant starts hallucinating answers to "hi" instead of greeting.

Importing this module compiles the graph (a MemorySaver and a node registry),
which is in-process and cheap — no Qdrant, no LLM client, no network.
"""

import pytest

from app.agents.graph import route_planner


def _state(current_query: str) -> dict:
    """A minimal AgentState — route_planner only reads current_query."""
    return {
        "messages": [{"role": "user", "content": "hello"}],
        "current_query": current_query,
        "documents": [],
        "plan": [],
        "status": "",
        "final_answer": "",
    }


def test_conversational_routes_to_responder():
    """The sentinel must skip retrieval entirely."""
    assert route_planner(_state("CONVERSATIONAL")) == "responder"


def test_technical_query_routes_to_retriever():
    assert route_planner(_state("kubernetes pod scheduling")) == "retriever"


def test_any_non_sentinel_value_routes_to_retriever():
    """Only the exact sentinel short-circuits.

    Planner output is free-form LLM text, so a near-miss like a trailing
    newline must still retrieve rather than be answered from thin air.
    """
    for value in ("conversational", "CONVERSATIONAL ", "\nCONVERSATIONAL", "", "CONVERSE"):
        assert route_planner(_state(value)) == "retriever", value


def test_planner_emits_a_routable_value(monkeypatch):
    """A stubbed planner must produce something route_planner can route.

    Stubs the LLM so no Groq/Portkey key is needed, then asserts the planner's
    output lands on the right edge — for the conversational branch and the
    technical branch.
    """

    class _StubLLM:
        def __init__(self, decision: str):
            self.decision = decision

        def invoke(self, prompt):
            return type("R", (), {"content": self.decision})()

    import app.agents.nodes.planner as planner_mod

    for decision, expected_edge in (
        ("CONVERSATIONAL", "responder"),
        ("kubernetes ingress controllers", "retriever"),
    ):
        monkeypatch.setattr(planner_mod, "_get_llm", lambda d=decision: _StubLLM(d))
        out = planner_mod.planner_node(_state(""))
        assert out["current_query"] == decision
        assert route_planner(_state(out["current_query"])) == expected_edge


def test_planner_reports_lost_memory_on_a_restarted_thread(monkeypatch):
    """Empty history + client-said-it-has-history => say so, don't fake it.

    On Render's free tier the container sleeps after ~15 min idle, killing the
    in-process MemorySaver. The server cannot tell "never seen this thread"
    from "saw it, then lost it", so the client reports its own turn count and
    the planner surfaces the gap instead of answering as if the thread were new.
    """

    class _StubLLM:
        def invoke(self, prompt):
            return type("R", (), {"content": "CONVERSATIONAL"})()

    import app.agents.nodes.planner as planner_mod

    monkeypatch.setattr(planner_mod, "_get_llm", lambda: _StubLLM())
    monkeypatch.setitem(planner_mod.__dict__, "_MEMORY_LOST", "⚠️ MEMORY LOST")

    # Client holds 3 earlier turns, server state has none.
    out = planner_mod.planner_node(
        _state(""),
        config={"configurable": {"thread_id": "abc", "client_turns": 3}},
    )
    assert "MEMORY LOST" in out["status"]

    # A genuinely new thread (client_turns absent / zero) must NOT claim loss.
    fresh = planner_mod.planner_node(_state(""), config={"configurable": {"thread_id": "abc"}})
    assert "MEMORY LOST" not in fresh["status"]

    # And a thread that still has server-side history is definitely not lost.
    with_history = planner_mod.planner_node(
        {
            "messages": [
                {"role": "user", "content": "my name is Ada"},
                {"role": "assistant", "content": "Hi Ada"},
                {"role": "user", "content": "what is my name"},
            ],
            "current_query": "",
            "documents": [],
            "plan": [],
            "status": "",
            "final_answer": "",
        },
        config={"configurable": {"thread_id": "abc", "client_turns": 2}},
    )
    assert "MEMORY LOST" not in with_history["status"]


def test_route_planner_ignores_a_missing_current_query():
    """Defensive: a state that never reached the planner must not KeyError here."""
    with pytest.raises(KeyError):
        route_planner({"messages": []})
