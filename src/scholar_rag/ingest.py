"""Ingestion pipeline: Semantic Scholar → embed → Qdrant."""

import os
import time
import uuid
from typing import Optional

from dotenv import load_dotenv
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)
from rich.console import Console
from rich.progress import track
from semanticscholar import SemanticScholar
from sentence_transformers import SentenceTransformer

load_dotenv()

console = Console()

DENSE_MODEL = "all-MiniLM-L6-v2"
SPARSE_MODEL = "Qdrant/bm42-all-minilm-l6-v2-attentions"
DENSE_DIM = 384

# Conservative sleep between S2 API calls when no API key is set.
# With a key: 1 req/s is the limit; without: ~100 req/5min → 1 req/3s to be safe.
_SLEEP_WITH_KEY = 1.0
_SLEEP_NO_KEY = 3.0

S2_FIELDS = [
    "paperId", "title", "abstract", "year", "venue",
    "authors", "citationCount", "tldr",
]


def build_document_text(paper: dict) -> str:
    """Build the text to embed: title + abstract + TLDR."""
    parts = [paper.get("title", "")]
    if paper.get("abstract"):
        parts.append(paper["abstract"])
    tldr = paper.get("tldr")
    if tldr:
        parts.append(f"TLDR: {tldr}")
    return "\n\n".join(p for p in parts if p)


def _paper_to_dict(paper, source: str) -> dict:
    """Normalise a Semantic Scholar paper object into a plain dict."""
    authors = []
    if paper.authors:
        authors = [a.name for a in paper.authors if hasattr(a, "name") and a.name]

    tldr_text = None
    if paper.tldr:
        if isinstance(paper.tldr, dict):
            tldr_text = paper.tldr.get("text")
        elif hasattr(paper.tldr, "text"):
            tldr_text = paper.tldr.text

    return {
        "paper_id": paper.paperId,
        "title": paper.title or "",
        "abstract": paper.abstract or "",
        "year": paper.year,
        "venue": paper.venue or "",
        "authors": authors,
        "citation_count": paper.citationCount or 0,
        "tldr": tldr_text,
        "source": source,
    }


def fetch_seed_papers(author_name: str, sch: SemanticScholar, sleep_interval: float = 1.0) -> list[dict]:
    """Look up author on Semantic Scholar and return their papers with full details."""
    console.print(f"[cyan]Searching for author:[/cyan] {author_name}")

    results = sch.search_author(author_name, fields=["authorId", "name", "papers"])
    if not results:
        raise ValueError(f"No author found for: {author_name!r}")

    author = results[0]
    console.print(f"[green]Found:[/green] {author.name} (ID: {author.authorId})")

    author_detail = sch.get_author(
        author.authorId,
        fields=["papers.paperId", "papers.title", "papers.year"],
    )
    paper_ids = [p.paperId for p in author_detail.papers if p.paperId]
    console.print(f"[cyan]Fetching details for[/cyan] {len(paper_ids)} seed papers…")

    papers: list[dict] = []
    for paper_id in track(paper_ids, description="Fetching seed papers…"):
        try:
            p = sch.get_paper(paper_id, fields=S2_FIELDS)
            if p and p.title:
                papers.append(_paper_to_dict(p, source="seed"))
            time.sleep(sleep_interval)
        except Exception as exc:
            console.print(f"[yellow]Skipping {paper_id}: {exc}[/yellow]")

    return papers


def _get_citing_paper_ids(paper_id: str, limit: int = 100, sleep_interval: float = 1.0) -> list[str]:
    """Fetch IDs of papers that cite the given paper via the REST API."""
    import requests
    ids: list[str] = []
    offset = 0
    while len(ids) < limit:
        r = requests.get(
            f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}/citations",
            params={"fields": "paperId", "limit": 100, "offset": offset},
        )
        if r.status_code != 200:
            break
        data = r.json().get("data") or []
        for item in data:
            pid = (item.get("citingPaper") or {}).get("paperId")
            if pid:
                ids.append(pid)
        time.sleep(sleep_interval)
        if len(data) < 100:
            break
        offset += 100
    return ids[:limit]


def expand_corpus(
    seed_papers: list[dict],
    sch: SemanticScholar,
    max_papers: int = 50,
    sleep_interval: float = 1.0,
) -> list[dict]:
    """1-hop expansion: fetch papers that cite the seed set.

    Uses forward citations rather than references because the publisher has
    elided the references field from the API ("references" returns null).
    """
    seed_ids = {p["paper_id"] for p in seed_papers}

    candidate_ids: set[str] = set()
    for paper in seed_papers:
        candidate_ids.update(_get_citing_paper_ids(paper["paper_id"], sleep_interval=sleep_interval))
    candidate_ids -= seed_ids
    candidate_ids = set(list(candidate_ids)[:max_papers])

    console.print(
        f"[cyan]Expanding corpus:[/cyan] fetching up to {len(candidate_ids)} citing papers…"
    )

    expanded: list[dict] = []
    for paper_id in track(candidate_ids, description="Expanding corpus…  "):
        try:
            p = sch.get_paper(
                paper_id,
                fields=["paperId", "title", "abstract", "year", "venue",
                        "authors", "citationCount", "tldr"],
            )
            if p and p.title and p.abstract:
                expanded.append(_paper_to_dict(p, source="expanded"))
            time.sleep(sleep_interval)
        except Exception as exc:
            console.print(f"[yellow]Skipping {paper_id}: {exc}[/yellow]")

    return expanded


def ensure_collection(client: QdrantClient, collection: str) -> None:
    """Create the Qdrant collection (dense + BM42 sparse) if it doesn't already exist."""
    if client.collection_exists(collection):
        console.print(f"[cyan]Collection[/cyan] '{collection}' already exists — skipping creation.")
        return

    client.create_collection(
        collection_name=collection,
        vectors_config={
            "dense": VectorParams(size=DENSE_DIM, distance=Distance.COSINE),
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams(
                index=SparseIndexParams(on_disk=False),
            ),
        },
    )
    console.print(f"[green]Created collection[/green] '{collection}' (dense + sparse BM42).")


def ingest(
    author: str = "Oliver Buchtala",
    expand_hops: int = 1,
    qdrant_url: Optional[str] = None,
    collection: Optional[str] = None,
    s2_api_key: Optional[str] = None,
) -> int:
    """Full ingest pipeline. Returns the number of documents upserted."""
    qdrant_url = qdrant_url or os.getenv("QDRANT_URL", "http://localhost:6333")
    collection = collection or os.getenv("QDRANT_COLLECTION", "papers")
    s2_api_key = s2_api_key or os.getenv("S2_API_KEY") or None

    qdrant_api_key = (os.getenv("QDRANT_API_KEY") or "").strip() or None

    sch = SemanticScholar(api_key=s2_api_key)
    sleep_interval = _SLEEP_WITH_KEY if s2_api_key else _SLEEP_NO_KEY
    qdrant = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    dense_model = SentenceTransformer(DENSE_MODEL)
    sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL)

    # --- fetch ---
    seed_papers = fetch_seed_papers(author, sch, sleep_interval)
    console.print(f"[green]Fetched[/green] {len(seed_papers)} seed papers.")

    all_papers = list(seed_papers)
    if expand_hops > 0:
        expanded = expand_corpus(seed_papers, sch, sleep_interval=sleep_interval)
        all_papers.extend(expanded)
        console.print(
            f"[green]Total corpus:[/green] {len(all_papers)} papers "
            f"({len(expanded)} expanded)."
        )

    # --- deduplicate ---
    seen: set[str] = set()
    unique: list[dict] = []
    for p in all_papers:
        if p["paper_id"] not in seen:
            seen.add(p["paper_id"])
            unique.append(p)
    all_papers = [p for p in unique if p["title"] or p["abstract"]]

    # --- embed ---
    texts = [build_document_text(p) for p in all_papers]
    console.print(f"[cyan]Embedding {len(texts)} documents…[/cyan]")
    dense_vecs = dense_model.encode(texts, show_progress_bar=True, batch_size=32)
    sparse_vecs = list(sparse_model.embed(texts))

    # --- upsert ---
    ensure_collection(qdrant, collection)
    console.print("[cyan]Upserting to Qdrant…[/cyan]")

    points: list[PointStruct] = []
    for paper, text, dense, sparse in zip(all_papers, texts, dense_vecs, sparse_vecs):
        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, paper["paper_id"]))
        points.append(
            PointStruct(
                id=point_id,
                vector={
                    "dense": dense.tolist(),
                    "sparse": SparseVector(
                        indices=sparse.indices.tolist(),
                        values=sparse.values.tolist(),
                    ),
                },
                payload={
                    "paper_id": paper["paper_id"],
                    "title": paper["title"],
                    "authors": paper["authors"],
                    "year": paper["year"],
                    "venue": paper["venue"],
                    "citation_count": paper["citation_count"],
                    "source": paper["source"],
                    "text": text,
                },
            )
        )

    batch_size = 50
    for i in range(0, len(points), batch_size):
        qdrant.upsert(collection_name=collection, points=points[i : i + batch_size])

    console.print(
        f"[bold green]✓ Ingested {len(points)} papers into '{collection}'.[/bold green]"
    )
    return len(points)
