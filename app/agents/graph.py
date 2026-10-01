from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from app.agents.state import AgentState
from app.agents.nodes.planner import planner_node, REFUSED
from app.agents.nodes.retriever import retrieve_node
from app.agents.nodes.responder import generate_node


# 1. Initialize the State Graph
workflow = StateGraph(AgentState)


# 2. Define the Nodes
workflow.add_node("planner", planner_node)
workflow.add_node("retriever", retrieve_node)
workflow.add_node("responder", generate_node)
# A refusal is already the final answer — the planner set final_answer on the
# way past. This node exists so the "refuse" edge resolves to a real node; it
# deliberately does no retrieval and no LLM synthesis.
workflow.add_node("refuse", lambda state: {})

# 3. Define the Edges & Routing Logic
def route_planner(state: AgentState):
    """
    Routes the workflow based on the planner's decision.

    Three outcomes, and only one of them searches:
      CONVERSATIONAL -> answer from memory, no retrieval
      REFUSED       -> the planner already produced the refusal as the answer,
                       so go straight to END; no retrieval, no synthesis
      anything else -> a real search query, retrieve and answer
    """
    if state["current_query"] == "CONVERSATIONAL":
        return "responder"
    if state["current_query"] == REFUSED:
        return "refuse"
    return "retriever"

workflow.set_entry_point("planner")


workflow.add_conditional_edges(
    "planner",
    route_planner,
    {
        "retriever": "retriever",
        "responder": "responder",
        "refuse": "refuse",
    }
)

workflow.add_edge("retriever", "responder")
workflow.add_edge("responder", END)
# A refusal is already the final answer — nothing left to compute.
workflow.add_edge("refuse", END)

checkpointer = MemorySaver()


# 4. Compile the Graph with Memory
rag_agent = workflow.compile(checkpointer=checkpointer)
