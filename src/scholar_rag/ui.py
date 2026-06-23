"""Gradio interface for the scholar-rag ask pipeline."""

import gradio as gr

EXAMPLES = [
    "What are the main contributions of the evolutionary RBF classifier?",
    "What is the main contribution of the 2005 IEEE SMC paper?",
    "Which later works built on the evolutionary RBF approach?",
    "Compare the fuzzy classifier and RBF approaches across the publications",
]

_graph = None


def _get_graph():
    global _graph
    if _graph is None:
        from scholar_rag.graph import build_graph
        _graph = build_graph()
    return _graph


def _ask(query: str) -> str:
    result = _get_graph().invoke({
        "query": query,
        "intent": "",
        "retrieved_texts": [],
        "agent_scratchpad": [],
        "answer": "",
        "sources": [],
    })
    return result.get("answer", "No answer generated.")


demo = gr.Interface(
    fn=_ask,
    inputs=gr.Textbox(label="Research question", lines=2, placeholder="Ask about the publications…"),
    outputs=gr.Markdown(label="Answer"),
    examples=EXAMPLES,
    title="Scholar RAG",
    description="Ask questions about Oliver Buchtala's research publications.",
)
