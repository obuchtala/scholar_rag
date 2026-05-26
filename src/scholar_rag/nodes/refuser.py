"""Refuser node: returns a polite out-of-scope message without any LLM call."""

from scholar_rag.graph import ResearchState

_MESSAGE = (
    "I'm focused on searching and synthesising academic literature. "
    "I can help with questions about scientific papers, citations, authors, and research topics — "
    "but that question is outside my scope."
)


def refuser_node(state: ResearchState) -> dict:
    """Return a fixed refusal message. No LLM call — fast and cheap."""
    return {"answer": _MESSAGE, "sources": []}
