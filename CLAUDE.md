# scholar-rag

Multi-agent RAG system for academic literature exploration via citation graph traversal.
See README.md for architecture, setup, and example queries.

---

## Development

### Running tests

```bash
uv run pytest                   # unit tests (default — no live connections needed)
uv run pytest -m integration    # integration tests (requires live Qdrant + Mistral key)
```

### Environment

```bash
cp .env.example .env
# Required: MISTRAL_API_KEY
# Optional: S2_API_KEY, QDRANT_API_KEY, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY
```

### Docker

```bash
docker compose up -d            # start Qdrant
docker compose run app ingest   # ingest via compose network
docker compose run app ask "..."
```

---

## Key technical decisions

- **LLM is provider-agnostic** — `init_chat_model()` from LangChain reads `LLM_MODEL` env var; swap model without code changes (default: `mistralai:mistral-medium-3-5`)
- **Hybrid search** — dense (`all-MiniLM-L6-v2`, 384-dim) + sparse (BM42 via `fastembed`), merged with Reciprocal Rank Fusion in Qdrant
- **Langfuse tracing is opt-in** — absent keys → runs without tracing, no errors raised
- **Corpus expansion uses forward citations** — papers that cite the seed author's work, not backward references (rationale: captures influence, not just prior art)
- **Refuser node** — `out_of_scope` intent routes to a fixed reply with no further LLM calls, avoiding wasted inference on unrelated queries
