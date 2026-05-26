"""Integration tests — require live services (Qdrant + Mistral).

Run with:
    uv run pytest -m integration

Skipped by default in the normal test suite.
"""

import os

import pytest


# ---------------------------------------------------------------------------
# Fixtures / skip guards
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def qdrant_client():
    """Return a live QdrantClient, skip if Qdrant is unreachable."""
    from qdrant_client import QdrantClient
    url = os.getenv("QDRANT_URL", "http://localhost:6333")
    client = QdrantClient(url=url)
    try:
        client.get_collections()
    except Exception:
        pytest.skip("Qdrant not reachable — start docker compose first")
    collection = os.getenv("QDRANT_COLLECTION", "papers")
    if not client.collection_exists(collection):
        pytest.skip(f"Collection '{collection}' not found — run 'scholar-rag ingest' first")
    return client


@pytest.fixture(scope="module")
def mistral_key():
    """Skip if MISTRAL_API_KEY is not set."""
    key = os.getenv("MISTRAL_API_KEY", "")
    if not key:
        pytest.skip("MISTRAL_API_KEY not set")
    return key


# ---------------------------------------------------------------------------
# 1. Hybrid search — real Qdrant, no LLM
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_hybrid_search(qdrant_client):
    from scholar_rag.tools.qdrant_search import hybrid_search

    results = hybrid_search("evolutionary RBF classifier", top_k=3)

    assert len(results) > 0, "Expected at least one result from the corpus"
    for doc in results:
        assert doc.page_content, "Document should have text content"
        assert doc.metadata.get("paper_id"), "Document should have a paper_id"
        assert doc.metadata.get("title"), "Document should have a title"


@pytest.mark.integration
def test_search_corpus_tool(qdrant_client):
    from scholar_rag.tools.qdrant_search import search_corpus

    result = search_corpus.invoke({"query": "fuzzy classifier", "top_k": 3})

    assert isinstance(result, str)
    assert "Found" in result
    assert "No relevant papers" not in result


# ---------------------------------------------------------------------------
# 2. Full retrieval path — real Qdrant + real Mistral
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_graph_retrieval_path(qdrant_client, mistral_key):
    from scholar_rag.graph import build_graph

    graph = build_graph()
    result = graph.invoke({
        "query": "What are the main contributions of the evolutionary RBF classifier approach?",
        "intent": "",
        "retrieved_texts": [],
        "agent_scratchpad": [],
        "answer": "",
        "sources": [],
    })

    assert result["intent"] == "retrieval"
    assert isinstance(result["answer"], str)
    assert len(result["answer"]) > 100, "Answer should be a substantive response"
    assert len(result["retrieved_texts"]) > 0, "Retriever should have gathered context"


# ---------------------------------------------------------------------------
# 3. Out-of-scope path — real Mistral, no Qdrant needed
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_graph_out_of_scope(mistral_key):
    from scholar_rag.graph import build_graph

    graph = build_graph()
    result = graph.invoke({
        "query": "Write me a Python function to sort a list.",
        "intent": "",
        "retrieved_texts": [],
        "agent_scratchpad": [],
        "answer": "",
        "sources": [],
    })

    assert result["intent"] == "out_of_scope"
    assert "scope" in result["answer"].lower()
    assert result["sources"] == []
