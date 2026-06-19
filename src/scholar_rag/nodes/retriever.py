"""Retriever node: ReAct agent that calls tools to gather context."""

import os

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage
from langchain.agents import create_agent

from scholar_rag.graph import ResearchState
from scholar_rag.tools.qdrant_search import search_corpus
from scholar_rag.tools.semantic_scholar import get_citations, get_paper_details, get_references
from scholar_rag.constants import DEFAULT_LLM_MODEL_ID

_TOOLS = [search_corpus, get_paper_details, get_citations, get_references]

_SYSTEM = """\
You are a research assistant with access to an indexed corpus of academic papers and the live \
Semantic Scholar API.

Your job is to gather all context needed to answer the user's research question. Use tools \
iteratively:
1. Start with search_corpus to find directly relevant papers in the indexed collection.
2. Use get_paper_details to fetch full metadata for promising papers.
3. Use get_citations to discover papers that built on a specific paper (forward traversal).
4. Use get_references to find foundational work a paper cites (backward traversal).

Gather enough context for a thorough, cited answer. When you have sufficient information, \
stop calling tools and summarise what you found in a structured way.
"""

def retriever_node(state: ResearchState) -> dict:
    """Run a ReAct agent to gather context via tool calls."""
    llm = init_chat_model(
        model=os.getenv("LLM_MODEL", DEFAULT_LLM_MODEL_ID),
        max_tokens=4096,
    )
    agent = create_agent(llm, _TOOLS, system_prompt=_SYSTEM)

    result = agent.invoke({"messages": [
        HumanMessage(content=state["query"])
    ]})

    messages = result.get("messages", [])

    # Collect all tool output messages as retrieved docs (store as plain strings for state)
    retrieved_texts = []
    for msg in messages:
        if hasattr(msg, "type") and msg.type == "tool":
            retrieved_texts.append(msg.content)

    # The final AI message is the one with no tool_calls — the agent's synthesis
    final_answer = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and not msg.tool_calls:
            final_answer = msg.content if isinstance(msg.content, str) else str(msg.content)
            break

    return {
        "agent_scratchpad": messages,
        "retrieved_texts": retrieved_texts,
        "answer": final_answer,
    }
