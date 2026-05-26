# scholar-rag

A multi-agent RAG system that explores and synthesises academic literature through live citation graph traversal, built with LangGraph, Qdrant, and Mistral.

The indexed corpus is the peer-reviewed publications of **Oliver Buchtala** (fuzzy systems, ML, pattern recognition — JKU Linz, 2001–2013), plus their first-hop references — all fetched at ingest time from the Semantic Scholar API. No PDFs required.

---

## Architecture

```mermaid
flowchart TD
    CLI["CLI  (Typer)"]

    subgraph Ingest
        S2A["Semantic Scholar API\nauthor lookup + 1-hop expansion"]
        EMB["Embed\nall-MiniLM-L6-v2 dense\nBM42 sparse"]
        QDB[("Qdrant\ncollection: papers")]
        S2A --> EMB --> QDB
    end

    subgraph Ask ["ask  (LangGraph StateGraph)"]
        Router["router\nclassifies intent"]
        Retriever["retriever\nReAct agent"]
        Responder["responder\nsynthesises answer"]
        Refuser["refuser\nout-of-scope reply"]

        Router -->|retrieval| Retriever
        Router -->|direct| Responder
        Router -->|out_of_scope| Refuser
        Retriever --> Responder
    end

    subgraph Tools
        T1["search_corpus\nhybrid search · RRF"]
        T2["get_paper_details"]
        T3["get_citations"]
        T4["get_references"]
    end

    CLI --> Ingest
    CLI --> Ask
    Retriever <-->|tool calls| Tools
    T1 <--> QDB
    T2 & T3 & T4 <--> S2B["Semantic Scholar API\nlive"]
```

### How a query flows

1. **`router`** classifies intent with a single LLM call (`max_tokens=10`): `retrieval`, `direct`, or `out_of_scope`.
2. **`retriever`** runs a ReAct loop — calls tools iteratively until it has enough context.
3. **`responder`** synthesises the gathered context into a cited answer.
4. **`refuser`** short-circuits unrelated queries with a fixed message — no further LLM calls.

---

## Tech stack

| Layer | Choice | Notes |
|---|---|---|
| Agent orchestration | **LangGraph** | StateGraph with conditional routing |
| LLM | **Mistral Medium 3.5** | Via `langchain-mistralai`; swap via `LLM_MODEL` env var |
| Vector store | **Qdrant** | Hybrid search: dense + BM42 sparse |
| Dense embeddings | `all-MiniLM-L6-v2` | Local, 384-dim, no API cost |
| Sparse embeddings | BM42 via `fastembed` | Qdrant's attention-weighted keyword model |
| Data source | Semantic Scholar API | Abstracts, TLDRs, citation graph |
| Observability | **Langfuse Cloud** | Full trace per `ask` invocation (optional) |
| CLI | Typer | `ingest` / `ask` / `list-papers` |
| Package manager | uv | Fast, lockfile-based |
| Runtime | Python 3.12, Docker Compose | |

---

## Prerequisites

- **Docker Desktop** (runs Qdrant locally)
- **Mistral API key** — [console.mistral.ai](https://console.mistral.ai)
- **uv** — `pip install uv` (or see [astral.sh/uv](https://astral.sh/uv))
- Semantic Scholar API key — recommended for reliable corpus expansion; institutions and companies can request one at [semanticscholar.org/product/api](https://www.semanticscholar.org/product/api)
- Langfuse Cloud account — optional, free at [cloud.langfuse.com](https://cloud.langfuse.com) (for observability)

---

## Quick start

```bash
# 1. Clone and configure
git clone https://github.com/obuchtala/scholar-rag
cd scholar-rag
cp .env.example .env
# edit .env — set MISTRAL_API_KEY (required)
# optionally set LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY for tracing

# 2. Start Qdrant
docker compose up -d

# 3. Set up Python environment
uv venv .venv
uv pip install -e . --python .venv/bin/python

# 4. Ingest the corpus (~60 papers, takes 2–3 min)
uv run scholar-rag ingest

# 5. Ask questions
uv run scholar-rag ask "What are the main contributions of the evolutionary RBF classifier?"
```

### Via Docker Compose (production-style)

```bash
docker compose run app ingest
docker compose run app ask "Compare the fuzzy and RBF classification approaches"
```

The `app` service uses `QDRANT_URL=http://qdrant:6333` (compose network) automatically.

---

## Example queries

### 1 — Targeted paper lookup

```
$ uv run scholar-rag ask "What is the main contribution of the 2005 IEEE SMC paper?"
```

```
The 2005 IEEE Transactions on Systems, Man, and Cybernetics paper by Buchtala et al.
introduces an **evolutionary optimisation approach for RBF classifier design**. The key
contribution is a genetic algorithm that simultaneously evolves the structure (number and
placement of radial basis functions) and parameters of the classifier, avoiding the need
for manual architecture selection.

The method was benchmarked on standard UCI datasets and outperformed fixed-architecture
RBF networks trained with gradient descent, particularly on small-sample problems where
overfitting is a concern [Buchtala et al., 2005].

**Sources**
- [<paper_id>] An Evolutionary Algorithm for RBF Classifier Design — IEEE Trans. SMC, 2005
```

---

### 2 — Multi-hop citation traversal  *(demonstrates agentic tool use)*

```
$ uv run scholar-rag ask "Which later works built on the evolutionary RBF approach?" --verbose
```

```
── Retrieved context ──
[search_corpus] Found 5 relevant papers: ...
[get_citations] Papers citing 'An Evolutionary Algorithm for RBF Classifier Design' (top 10): ...
[get_paper_details] [<paper_id>] Hybrid Evolutionary-Gradient RBF Training (2008) ...
──────────────────────

Several lines of work extended the evolutionary RBF approach:

**Direct extensions (citing the 2005 paper):**
- [Author et al., 2008] combined the evolutionary structure search with local gradient
  refinement, reducing convergence time while retaining the architecture flexibility.
- [Author et al., 2011] applied the same evolutionary framework to deep RBF networks,
  scaling to higher-dimensional feature spaces.

**Parallel development:**
The fuzzy-RBF hybrid approach explored in [Buchtala et al., 2007] shares the motivation
of avoiding manual architecture decisions, using fuzzy partitioning to initialise the
RBF centres before evolutionary refinement.

**Sources**
- [<paper_id>] An Evolutionary Algorithm for RBF Classifier Design, 2005
- [<paper_id>] Hybrid Evolutionary-Gradient RBF Training, 2008
- ...

Langfuse trace: https://cloud.langfuse.com/traces/<trace-id>
```

The `--verbose` flag shows each tool call the agent made. This query triggered `search_corpus` → `get_citations` → `get_paper_details` — three tools, two API hops.

---

### 3 — Cross-corpus synthesis

```
$ uv run scholar-rag ask "Compare the fuzzy classifier and RBF approaches across the publications"
```

```
Across Oliver Buchtala's publications, two complementary approaches to pattern
classification emerge:

**Fuzzy classifier line** [papers from 2001–2006]
Focuses on interpretability: classification rules are expressed as fuzzy IF-THEN
statements, making decisions human-readable. The key challenge addressed is learning
compact, accurate rule bases from data. These papers prioritise explainability over
raw accuracy.

**RBF classifier line** [papers from 2003–2008]
Focuses on generalisation: radial basis function networks provide smooth decision
boundaries and are well-suited to small-sample regimes. The evolutionary approach
automates architecture selection — the main engineering bottleneck.

**Convergence point**
The 2007 work on fuzzy-RBF hybrids synthesises both lines: RBF centres are initialised
from fuzzy partitions, and the final model retains interpretable structure while achieving
RBF-level accuracy. This paper has the highest citation count in the corpus.

**Sources**
- [<paper_id>] Evolutionary Fuzzy Classifier, 2003
- [<paper_id>] An Evolutionary Algorithm for RBF Classifier Design, 2005
- [<paper_id>] Fuzzy-RBF Hybrid Classifier, 2007
```

---

## Project structure

```
scholar-rag/
├── src/scholar_rag/
│   ├── cli.py                  # ingest / ask / list-papers
│   ├── ingest.py               # Semantic Scholar → embed → Qdrant
│   ├── graph.py                # LangGraph StateGraph
│   ├── nodes/
│   │   ├── router.py           # intent classification (retrieval/direct/out_of_scope)
│   │   ├── retriever.py        # ReAct agent
│   │   ├── responder.py        # synthesis + citation formatting
│   │   └── refuser.py          # out-of-scope short-circuit
│   └── tools/
│       ├── qdrant_search.py    # hybrid search (dense + BM42 sparse, RRF)
│       └── semantic_scholar.py # get_paper_details / get_citations / get_references
├── tests/                      # 37 unit + 4 integration tests
├── doc/                        # deployment guide
├── docker-compose.yml          # Qdrant + app
├── Dockerfile                  # python:3.12-slim + uv
└── pyproject.toml              # uv-managed dependencies
```

---

## Running tests

```bash
uv pip install pytest pytest-asyncio pytest-mock --python .venv/bin/python
uv run pytest                   # 37 unit tests (no live connections needed)
uv run pytest -m integration    # 4 integration tests (requires live Qdrant + Mistral key)
```

All unit tests mock external services (Qdrant, Semantic Scholar API, Mistral) — no API keys or running containers needed.

---

## Observability

Every `ask` invocation is traced to [Langfuse Cloud](https://cloud.langfuse.com):

- One root trace per query
- Child spans per LangGraph node (`router`, `retriever`, `responder`)
- All tool calls captured (name, input, output, latency)
- Token counts and model logged per LLM call

Tracing is opt-in — set `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` in `.env` to enable.
If those keys are absent the graph runs normally with no tracing and no errors.

---

## Development environment

Developed in a VS Code devcontainer on a DigitalOcean droplet with persistent storage —
providing a consistent, isolated workspace across sessions.

Development was assisted by [Claude Code](https://claude.ai/code) throughout.

---

## About

Built by **Oliver Buchtala**.
The corpus intentionally uses Oliver's own publications, making it a live demonstration of
RAG over a domain the author knows deeply — useful for spotting hallucinations and evaluating answer quality.

- GitHub: [github.com/obuchtala](https://github.com/obuchtala)
- IBM Agentic AI & RAG Certificate, May 2026 (LangGraph, RAG pipelines, vector databases)
- Former Research Associate, JKU Linz (2009–2013)
