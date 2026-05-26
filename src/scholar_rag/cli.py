"""Typer CLI entrypoint for scholar-rag."""

import os
from typing import Optional

import typer
from dotenv import load_dotenv

# using rich for nicer console output (tables, markdown, colored ouptut etc.)
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

load_dotenv()

app = typer.Typer(
    name="scholar-rag",
    help="Multi-agent RAG for exploring academic literature via citation graph traversal.",
    add_completion=False,
)
console = Console()


@app.command()
def ingest(
    author: str = typer.Option(
        "Oliver Buchtala",
        "--author",
        help="Author name to look up on Semantic Scholar.",
    ),
    expand_hops: int = typer.Option(
        1,
        "--expand-hops",
        help="Number of citation hops to expand beyond seed papers (0 = seed only).",
    ),
) -> None:
    """Ingest an author's papers from Semantic Scholar into the Qdrant vector store."""
    from scholar_rag.ingest import ingest as run_ingest

    count = run_ingest(author=author, expand_hops=expand_hops)
    console.print(f"[bold green]Done![/bold green] Indexed {count} papers.")


@app.command()
def ask(
    query: str = typer.Argument(..., help="Research question to answer."),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Print retrieved documents and Langfuse trace URL.",
    ),
) -> None:
    """Ask a research question. The agent searches the corpus and traverses citation graphs."""
    from scholar_rag.graph import build_graph

    # Langfuse tracing (optional — skipped if keys not set; use Langfuse Cloud)
    callbacks = []
    langfuse_url: Optional[str] = None
    lf_public = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    lf_secret = os.getenv("LANGFUSE_SECRET_KEY", "")
    lf_host = os.getenv("LANGFUSE_HOST", "")
    if lf_public and lf_secret:
        try:
            from langfuse.langchain import CallbackHandler
            handler = CallbackHandler()
            callbacks.append(handler)
            langfuse_url = lf_host or "https://cloud.langfuse.com"
        except Exception as exc:
            console.print(f"[yellow]Langfuse tracing unavailable: {exc}[/yellow]")

    console.print(f"[cyan]Query:[/cyan] {query}\n")

    graph = build_graph()

    result = graph.invoke(
        {
            "query": query,
            "intent": "",
            "retrieved_texts": [],
            "agent_scratchpad": [],
            "answer": "",
            "sources": [],
        },
        config={"callbacks": callbacks} if callbacks else {},
    )

    if verbose and result.get("retrieved_texts"):
        console.rule("[dim]Retrieved context[/dim]")
        for i, text in enumerate(result["retrieved_texts"], 1):
            console.print(f"[dim]── Source {i} ──[/dim]")
            console.print(text[:600] + ("…" if len(text) > 600 else ""))
        console.rule()

    console.print(Markdown(result.get("answer", "No answer generated.")))

    if verbose and langfuse_url:
        console.print(f"\n[dim]Langfuse trace: {langfuse_url}[/dim]")


@app.command(name="list-papers")
def list_papers() -> None:
    """List all papers indexed in the Qdrant vector store."""
    from qdrant_client import QdrantClient

    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    collection = os.getenv("QDRANT_COLLECTION", "papers")

    client = QdrantClient(url=qdrant_url)

    if not client.collection_exists(collection):
        console.print(f"[yellow]Collection '{collection}' does not exist. Run 'ingest' first.[/yellow]")
        raise typer.Exit(1)

    # Scroll through all points
    all_points = []
    offset = None
    while True:
        result, offset = client.scroll(
            collection_name=collection,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        all_points.extend(result)
        if offset is None:
            break

    if not all_points:
        console.print("[yellow]No papers indexed yet.[/yellow]")
        return

    # Split by source
    seed = [p for p in all_points if p.payload.get("source") == "seed"]
    expanded = [p for p in all_points if p.payload.get("source") == "expanded"]

    table = Table(title=f"Indexed papers ({len(all_points)} total)", show_lines=False)
    table.add_column("Source", style="cyan", width=8)
    table.add_column("Year", width=6)
    table.add_column("Citations", width=9, justify="right")
    table.add_column("Title")
    table.add_column("Venue")

    for point in sorted(all_points, key=lambda p: (
        p.payload.get("source", ""),
        -(p.payload.get("year") or 0),
    )):
        p = point.payload
        table.add_row(
            p.get("source", ""),
            str(p.get("year") or ""),
            str(p.get("citation_count") or 0),
            p.get("title", "")[:60],
            (p.get("venue") or "")[:30],
        )

    console.print(table)
    console.print(
        f"\n[green]{len(seed)} seed papers[/green] + "
        f"[blue]{len(expanded)} expanded[/blue] = {len(all_points)} total"
    )
