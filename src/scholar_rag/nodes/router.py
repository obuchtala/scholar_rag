"""Router node: classifies query intent as 'retrieval', 'direct', or 'out_of_scope'."""

import os

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

from scholar_rag.graph import ResearchState
from scholar_rag.constants import DEFAULT_LLM_MODEL_ID

_SYSTEM = """\
You are a routing assistant for an academic literature RAG system focused on scientific papers.
Classify the user query into exactly one of:

  retrieval    — needs searching the paper corpus, traversing citations, or comparing publications
  direct       — a domain-related question answerable from general knowledge (no corpus search needed)
  out_of_scope — completely unrelated to academic research or scientific literature
                 (e.g. coding tasks, cooking, weather, general chat)

Reply with only one word: retrieval  OR  direct  OR  out_of_scope
"""


def router_node(state: ResearchState) -> dict:
    """Classify query intent and route accordingly."""
    llm = init_chat_model(
        model=os.getenv("LLM_MODEL", DEFAULT_LLM_MODEL_ID),
        max_tokens=10,
    )
    response = llm.invoke([
        SystemMessage(content=_SYSTEM),
        HumanMessage(content=state["query"]),
    ])
    intent = response.content.strip().lower()
    if intent not in ("retrieval", "direct", "out_of_scope"):
        intent = "retrieval"  # default to retrieval when uncertain

    return {"intent": intent}
