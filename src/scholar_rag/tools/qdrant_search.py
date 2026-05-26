"""Hybrid search tool: dense + BM42 sparse over the Qdrant papers collection."""

import os
from typing import Optional

from fastembed import SparseTextEmbedding
from langchain_core.documents import Document
from langchain_core.tools import tool
from qdrant_client import QdrantClient
from qdrant_client.models import Fusion, FusionQuery, NamedSparseVector, NamedVector, SparseVector
from sentence_transformers import SentenceTransformer

from scholar_rag.ingest import DENSE_MODEL, SPARSE_MODEL
from scholar_rag.constants import DEFAULT_QDRANT_URL

_dense_model: Optional[SentenceTransformer] = None
_sparse_model: Optional[SparseTextEmbedding] = None
_qdrant_client: Optional[QdrantClient] = None


def _get_models() -> tuple[SentenceTransformer, SparseTextEmbedding]:
    global _dense_model, _sparse_model
    if _dense_model is None:
        _dense_model = SentenceTransformer(DENSE_MODEL)
    if _sparse_model is None:
        _sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL)
    return _dense_model, _sparse_model


def _get_client() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        url = os.getenv("QDRANT_URL", DEFAULT_QDRANT_URL)
        api_key = (os.getenv("QDRANT_API_KEY") or "").strip() or None
        _qdrant_client = QdrantClient(url=url, api_key=api_key)
    return _qdrant_client


def hybrid_search(
        query: str,
        top_k: int = 5,
        collection: Optional[str] = None) -> list[Document]:
    """Run hybrid (dense + sparse) search and return LangChain Documents."""

    collection = collection or os.getenv("QDRANT_COLLECTION", "papers")
    dense_model, sparse_model = _get_models()
    client = _get_client()

    dense_vec = dense_model.encode(query).tolist()
    sparse_emb = next(sparse_model.embed([query]))

    results = client.query_points(
        collection_name=collection,
        prefetch=[
            {"query": dense_vec, "using": "dense", "limit": top_k * 2},
            {
                "query": SparseVector(
                    indices=sparse_emb.indices.tolist(),
                    values=sparse_emb.values.tolist(),
                ),
                "using": "sparse",
                "limit": top_k * 2,
            },
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=top_k,
        with_payload=True,
    )

    docs = []
    for point in results.points:
        payload = point.payload or {}
        docs.append(
            Document(
                page_content=payload.get("text", ""),
                metadata={
                    "paper_id": payload.get("paper_id", ""),
                    "title": payload.get("title", ""),
                    "authors": payload.get("authors", []),
                    "year": payload.get("year"),
                    "venue": payload.get("venue", ""),
                    "citation_count": payload.get("citation_count", 0),
                    "source": payload.get("source", ""),
                    "score": point.score,
                },
            )
        )
    return docs


@tool
def search_corpus(query: str, top_k: int = 5) -> str:
    """Search the indexed paper corpus using hybrid semantic search (dense + BM42 sparse).

    Use this to find papers in the local corpus that are relevant to a research question.

    Args:
        query: Natural language search query
        top_k: Number of results to return (default 5)

    Returns:
        Formatted list of matching papers with titles, years, and relevant excerpts.
    """
    docs = hybrid_search(query, top_k=top_k)

    if not docs:
        return "No relevant papers found in the corpus for this query."

    lines = [f"Found {len(docs)} relevant papers:\n"]
    for i, doc in enumerate(docs, 1):
        m = doc.metadata
        authors = ", ".join(m.get("authors", [])[:3])
        if len(m.get("authors", [])) > 3:
            authors += " et al."
        lines.append(
            f"{i}. [{m['paper_id']}] {m['title']} ({m.get('year', 'N/A')})\n"
            f"   Authors: {authors}\n"
            f"   Venue: {m.get('venue', 'N/A')} | Citations: {m.get('citation_count', 0)}\n"
            f"   Excerpt: {doc.page_content[:300]}…\n"
        )
    return "\n".join(lines)
