"""LangGraph StateGraph for the scholar-rag multi-agent pipeline."""

from typing import TypedDict

from langgraph.graph import END, StateGraph


class ResearchState(TypedDict):
    query: str
    intent: str                  # "retrieval" | "direct" | "out_of_scope"
    retrieved_texts: list[str]   # raw tool-output strings from the ReAct agent
    agent_scratchpad: list       # full message history from the retriever agent
    answer: str
    sources: list[str]           # paper IDs extracted from the final answer


def _route_after_router(state: ResearchState) -> str:
    intent = state.get("intent", "retrieval")
    if intent == "out_of_scope":
        return "refuser"
    if intent == "direct":
        return "responder"
    return "retriever"


def build_graph() -> StateGraph:
    """Construct and compile the research StateGraph."""
    from scholar_rag.nodes.refuser import refuser_node
    from scholar_rag.nodes.responder import responder_node
    from scholar_rag.nodes.retriever import retriever_node
    from scholar_rag.nodes.router import router_node

    graph = StateGraph(ResearchState)

    graph.add_node("router", router_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("responder", responder_node)
    graph.add_node("refuser", refuser_node)

    graph.set_entry_point("router")
    graph.add_conditional_edges("router", _route_after_router)
    graph.add_edge("retriever", "responder")
    graph.add_edge("responder", END)
    graph.add_edge("refuser", END)

    return graph.compile()
