"""LangChain tools for live Semantic Scholar citation graph traversal."""

import os

from langchain_core.tools import tool
from semanticscholar import SemanticScholar

_S2_FIELDS_DETAIL = [
    "paperId", "title", "abstract", "year", "venue",
    "authors", "citationCount", "tldr",
]


def _get_client() -> SemanticScholar:
    api_key = os.getenv("S2_API_KEY") or None
    return SemanticScholar(api_key=api_key, timeout=10, retry=False)


def _format_paper(paper) -> str:
    """Format a Semantic Scholar paper object as a readable string."""
    if paper is None:
        return "Paper not found."

    authors = ""
    if paper.authors:
        names = [a.name for a in paper.authors if hasattr(a, "name")]
        authors = ", ".join(names[:5])
        if len(paper.authors) > 5:
            authors += " et al."

    tldr = ""
    if paper.tldr:
        text = paper.tldr.get("text") if isinstance(paper.tldr, dict) else getattr(paper.tldr, "text", None)
        if text:
            tldr = f"\nTLDR: {text}"

    abstract = (paper.abstract or "")[:500]
    if len(paper.abstract or "") > 500:
        abstract += "…"

    return (
        f"[{paper.paperId}] {paper.title} ({paper.year})\n"
        f"Authors: {authors}\n"
        f"Venue: {paper.venue or 'N/A'} | Citations: {paper.citationCount or 0}\n"
        f"Abstract: {abstract}"
        f"{tldr}"
    )


@tool
def get_paper_details(paper_id: str) -> str:
    """Fetch full metadata for a paper by its Semantic Scholar paper ID.

    Args:
        paper_id: Semantic Scholar paper ID (e.g. "649def34f8be52c8b66281af98ae884c09aef38b")

    Returns:
        Formatted string with title, authors, venue, year, citation count, abstract, and TLDR.
    """
    sch = _get_client()
    try:
        paper = sch.get_paper(paper_id, fields=_S2_FIELDS_DETAIL)
        return _format_paper(paper)
    except Exception as exc:
        return f"Error fetching paper {paper_id}: {exc}"


@tool
def get_citations(paper_id: str, limit: int = 10) -> str:
    """Get papers that CITE the given paper (forward citations).

    Use this to find work that built upon a specific paper.

    Args:
        paper_id: Semantic Scholar paper ID
        limit: Maximum number of citations to return (default 10, max 50)

    Returns:
        Formatted list of citing papers with IDs, titles, years, and citation counts.
    """
    sch = _get_client()
    limit = min(limit, 50)
    try:
        paper = sch.get_paper(
            paper_id,
            fields=["title", "citations.paperId", "citations.title",
                    "citations.year", "citations.authors", "citations.citationCount",
                    "citations.venue"],
        )
        if not paper:
            return f"Paper {paper_id} not found."

        citations = paper.citations or []
        if not citations:
            return f"No citations found for '{paper.title}'."

        citations_sorted = sorted(
            citations,
            key=lambda c: c.citationCount or 0,
            reverse=True,
        )[:limit]

        lines = [f"Papers citing '{paper.title}' (top {len(citations_sorted)}):"]
        for c in citations_sorted:
            authors = ""
            if c.authors:
                names = [a.name for a in c.authors[:2] if hasattr(a, "name")]
                authors = ", ".join(names)
            lines.append(
                f"  [{c.paperId}] {c.title} ({c.year}) — {authors} — "
                f"{c.citationCount or 0} citations"
            )
        return "\n".join(lines)
    except Exception as exc:
        return f"Error fetching citations for {paper_id}: {exc}"


@tool
def get_references(paper_id: str, limit: int = 10) -> str:
    """Get papers REFERENCED BY the given paper (backward references).

    Use this to find the foundational work that a paper builds on.

    Args:
        paper_id: Semantic Scholar paper ID
        limit: Maximum number of references to return (default 10, max 50)

    Returns:
        Formatted list of referenced papers with IDs, titles, and years.
    """
    sch = _get_client()
    limit = min(limit, 50)
    try:
        paper = sch.get_paper(
            paper_id,
            fields=["title", "references.paperId", "references.title",
                    "references.year", "references.authors", "references.citationCount"],
        )
        if not paper:
            return f"Paper {paper_id} not found."

        references = paper.references or []
        if not references:
            return f"No references found for '{paper.title}'."

        refs_sorted = sorted(
            references,
            key=lambda r: r.citationCount or 0,
            reverse=True,
        )[:limit]

        lines = [f"References in '{paper.title}' (top {len(refs_sorted)} by citation count):"]
        for r in refs_sorted:
            authors = ""
            if r.authors:
                names = [a.name for a in r.authors[:2] if hasattr(a, "name")]
                authors = ", ".join(names)
            lines.append(
                f"  [{r.paperId}] {r.title} ({r.year}) — {authors} — "
                f"{r.citationCount or 0} citations"
            )
        return "\n".join(lines)
    except Exception as exc:
        return f"Error fetching references for {paper_id}: {exc}"
