"""Responder node: synthesises retrieved context into a cited answer."""

import os
import re

from string import Template
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

from scholar_rag.graph import ResearchState
from scholar_rag.constants import DEFAULT_LLM_MODEL_ID

_SYSTEM = """\
You are a research synthesiser. Given a user question and retrieved context from academic papers, \
write a clear, thorough answer.

Rules:
- Cite every factual claim using [Author et al., Year] or the paper title in brackets.
- Include a "Sources" section at the end listing paper IDs and titles.
- Be concise but complete — aim for 3-6 paragraphs.
- If context is insufficient, say so explicitly rather than speculating.
"""

_PROMPT_TEMPLATE=Template("""\
Question: $query

Retrieved context:
$context
""")

_BLOCK_SEPARATOR = "\n\n---\n\n"

def responder_node(state: ResearchState) -> dict:
    """Synthesise the gathered context into a final cited answer."""
    llm = init_chat_model(
        model=os.getenv("LLM_MODEL", DEFAULT_LLM_MODEL_ID),
        max_tokens=2048,
    )

    context_block = _BLOCK_SEPARATOR.join(state.get("retrieved_texts", []))
    if not context_block:
        context_block = state.get("answer", "No context retrieved.")

    prompt = _PROMPT_TEMPLATE.substitute(query=state['query'], context=context_block)

    response = llm.invoke([
        SystemMessage(content=_SYSTEM),
        HumanMessage(content=prompt),
    ])

    answer = response.content if isinstance(response.content, str) else str(response.content)

    # Extract cited paper IDs from the answer.
    #
    # This is not general LLM-output parsing: Semantic Scholar paper IDs are SHA-1
    # hashes (40 lowercase hex chars), a format that never appears naturally in
    # academic prose. The search tool embeds them verbatim in the retrieved context
    # (e.g. "[649def34...] Attention Is All You Need"), and the system prompt
    # instructs the model to reproduce them in a Sources section — so the regex is
    # scanning for a known, machine-generated token, not interpreting free text.
    #
    # False positives: practically impossible (random 40-char hex in prose ≈ 0).
    # Failure mode: if the model omits the Sources section, sources is silently [],
    # which is preferable to garbage. The structured-output alternative
    # (with_structured_output) would give a stronger guarantee here and is the
    # right call if this node is ever extended.
    sources = list(dict.fromkeys(re.findall(r"\b[0-9a-f]{40}\b", answer)))

    return {"answer": answer, "sources": sources}
