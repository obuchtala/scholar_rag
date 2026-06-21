# scholar-rag — Solution Proposal

*Working title: scholar-rag. An agentic RAG system over a personal academic paper corpus.*

---

## Policy / Values

- **Production-fidelity [PRD]** — every component choice is defensible in a real deployment; no demo shortcuts
- **Debuggability [DBG]** — execution is traceable, control flow is explicit in code, failures are diagnosable
- **Open-stack / Data sovereignty [OSS]** — European provider preference, self-hostable components, no proprietary lock-in
- **Portfolio breadth [PBR]** — the project covers the full relevant stack: hybrid retrieval, agentic orchestration, observability, deployment
- **Domain relevance [DRL]** — corpus and retrieval strategy reflect the author's research identity, not a generic test dataset

---

## Problem Definition (compressed)

### Motivation

Academic paper discovery is fragmented. A researcher wants to explore the literature around their own work — papers that cite their publications, foundational references, adjacent contributions — through natural-language questions rather than keyword search. Standard keyword search fails on semantic queries; a generic LLM fails on specificity and currency. A RAG system over a curated personal corpus bridges this gap: it grounds the model in actual documents while preserving semantic query flexibility.

The project serves a dual purpose: a working research tool and a portfolio demonstration of the full AI engineering stack — retrieval architecture, agentic orchestration, observability, and deployment.

### Starting Point

- ~12 seed papers from the author's academic publication record
- Semantic Scholar API available for corpus expansion and live metadata lookup
- No existing retrieval or QA system
- Solo developer; implementation complexity must remain manageable

### Expected Outcome

- Natural-language interface over a personal academic corpus, returning cited answers
- Agentic retrieval that can traverse the citation graph iteratively, not just one-shot top-k
- All major AI engineering layers present: retrieval, orchestration, observability, containerized deployment
- Every design decision defensible as a production-pattern choice

### Constraints

- **Solo developer**: component count and integration complexity must be controlled
- **European data sovereignty**: LLM and cloud services must be operated by European providers where possible
- **Semantic Scholar rate limit**: ~100 requests / 5 min unauthenticated; ingest must be cached
- **No production budget**: free-tier infrastructure where feasible

### Glossary

> **[TODO — full proposal]** Define key terms used throughout this document:
>
> - RAG taxonomy: Naive RAG / Advanced RAG / Agentic RAG — definitions and distinctions between tiers
> - Hybrid search: dense vectors, sparse vectors, and what fusion achieves
> - Reciprocal Rank Fusion (RRF): the parameter-free rank combination mechanism
> - BM42: attention-weighted sparse embeddings and how they differ from BM25
> - ReAct pattern: the reason → act → observe loop and how it differs from a fixed pipeline
> - LangGraph StateGraph: nodes, conditional edges, shared typed state
> - Citation graph: forward citations (papers citing this work) vs. backward references (papers cited by this work)
> - Semantic Scholar API: the live data source for corpus expansion and metadata lookup

### Related Problems

> **[TODO — full proposal]** Document adjacent problems that are acknowledged but not addressed by this project:
>
> - Multi-user access: each user having their own corpus, session isolation, per-user rate limiting
> - Retrieval evaluation: automated quality measurement against a labeled query set (MRR, NDCG, LLM-as-judge)
> - Corpus freshness: detecting new citations and triggering incremental ingestion without full re-ingest
> - Multi-author corpus: expanding beyond a single author's publication record
> - Full-text ingestion: papers are indexed as title + abstract + TLDR only; chunking full PDFs is a separate problem

### Out of Scope

> **[TODO — full proposal]** Explicit exclusions — what this system does not address and why:
>
> - Real-time or streaming ingestion
> - Fine-tuning on the corpus
> - Multi-user access and authentication
> - Non-academic corpora (general document QA is a different use case)
> - Full-text paper ingestion (title + abstract + TLDR is the current unit; full PDF chunking deferred)
> - Voice interface
> - Automated evaluation harness (present as a Future Idea; not in scope for v1)

---

## Problem Frame

### State of the Art

> **[TODO — full proposal]** Survey the landscape this system sits within:
>
> - RAG taxonomy: where scholar-rag sits — Agentic RAG (retrieval as a tool in an agent loop) with Advanced RAG elements (hybrid search, RRF). Contrast with Naive RAG (fixed embed → retrieve → generate pipeline).
> - Competing frameworks: LlamaIndex (retrieval-first, rich query engine abstractions), Haystack (pipeline-based), raw LangChain chains — contrast with LangGraph's explicit graph approach and what each trades off.
> - Academic search products as prior art: Elicit, Consensus, Semantic Scholar's own search interface — these are end-user products; scholar-rag is infrastructure. The distinction matters for framing.
> - What is novel: the combination of a local hybrid corpus with live citation graph traversal via the Semantic Scholar API. Standard RAG systems operate over a static corpus; the agentic tools here allow the retriever to discover papers not in the local index at query time.

### Alternatives and Options

#### Q1: Agent orchestration framework

- **Option A: LangGraph (StateGraph)**
  - Explicit directed graph: nodes are steps or agent calls, edges are transitions (conditional or unconditional). A shared typed state object flows through the graph. Cycles are first-class — ReAct loops are native.
  - Pros: full control over execution flow; conditional routing (router → branch) is directly expressible in code; traces map one-to-one to the graph; LangChain callback integration for observability.
  - Cons: higher setup cost than declarative frameworks; more boilerplate for simple linear flows.

- **Option B: CrewAI**
  - Persona-based: each agent has a role, a backstory, and a task. Process is declared (sequential or hierarchical); the LLM resolves how each agent executes its piece.
  - Pros: very fast to stand up; natural fit for role-based multi-agent workflows; accessible to non-engineers.
  - Cons: control flow is emergent from role dynamics, not code — hard to inspect and debug; traces do not map cleanly to a graph; not suited for stateful retrieval loops with conditional retry.

- **Option C: LangChain Expression Language (LCEL)**
  - Composable linear pipelines: `chain = prompt | llm | parser`. Parallelism and fan-out supported.
  - Pros: minimal boilerplate for simple pipelines; tight LangChain integration.
  - Cons: linear by design — conditional routing and cycles are not expressible; cannot model the router → branch → ReAct loop pattern natively.

---

#### Q2: Vector store

- **Option A: Qdrant**
  - Open-source, self-hostable, native hybrid search (dense + sparse in a single query), Docker-friendly, production-grade ANN index. Free Cloud tier available.
  - Pros: self-hostable (data sovereignty); native hybrid search without external fusion layer; clean Python client; Qdrant Cloud free tier for solo projects.
  - Cons: requires running a separate service locally; fewer managed integrations than Pinecone.

- **Option B: Pinecone**
  - Fully managed, proprietary vector database.
  - Pros: zero infrastructure overhead; mature ecosystem.
  - Cons: US-domiciled managed service (data sovereignty concern); no self-hosting option; hybrid search requires additional configuration.

- **Option C: Weaviate**
  - Open-source, self-hostable, hybrid search supported.
  - Pros: mature, feature-rich, self-hostable.
  - Cons: more complex setup and schema definition than Qdrant; heavier resource footprint for a solo project.

- **Option D: ChromaDB**
  - Lightweight, embeddable, in-memory or local-file persistence.
  - Pros: minimal setup; good for prototyping.
  - Cons: not production-grade ANN index; no native hybrid search; unsuitable beyond prototyping.

---

#### Q3: Retrieval strategy

- **Option A: Pure dense retrieval**
  - Single embedding model, cosine similarity over dense vectors.
  - Pros: simple; single model to manage; widely understood.
  - Cons: loses precision on exact terminology — specific paper IDs, author names, technical abbreviations ("QLoRA", "NF4") may not retrieve correctly because rare terms embed similarly to semantically related concepts but without the specificity.

- **Option B: Hybrid retrieval — dense + sparse + RRF**
  - Dense vectors for semantic similarity; sparse vectors for term-level precision; Reciprocal Rank Fusion to combine ranked lists without a learned reranker.
  - Pros: catches both semantic paraphrase and exact terminology; RRF is parameter-free; implemented natively in Qdrant's `prefetch` + `FusionQuery` API.
  - Cons: two models to manage (dense + sparse); slightly more complex query construction.

- **Option C: Dense + cross-encoder reranker**
  - Dense retrieval for recall; cross-encoder model scores candidate pairs for precision.
  - Pros: highest precision for the final ranked list.
  - Cons: O(n) cross-encoder inference calls per query — slow; additional model to host; diminishing returns at small top-k.

---

#### Q4: Sparse embedding model

- **Option A: BM25**
  - Classical bag-of-words: term frequency × inverse document frequency. No neural component.
  - Pros: no model required; fully interpretable; universally available.
  - Cons: treats all occurrences of a term equally regardless of context — "attention mechanism" and "attention span" are indistinguishable; purely statistical, no semantic signal.

- **Option B: BM42**
  - BM25-compatible sparse format with term weights derived from transformer attention scores. Available in Qdrant's `fastembed` library.
  - Pros: same sparse index format as BM25; attention-based weighting gives modest semantic awareness within the sparse channel; integrates natively with Qdrant's hybrid search API; no separate inference server required.
  - Cons: less widely understood than BM25; dependent on Qdrant's fastembed implementation.

- **Option C: SPLADE**
  - Learned sparse representations: a trained model produces sparse vectors with learned vocabulary expansion.
  - Pros: highest accuracy among sparse models; handles synonyms and query expansion natively.
  - Cons: requires a dedicated inference server; significantly heavier than BM42; adds operational complexity disproportionate to this corpus scale.

---

#### Q5: Corpus construction strategy

- **Option A: Seed papers only**
  - Ingest only the author's own publications (~12 papers).
  - Pros: minimal API calls; fast ingest; controlled corpus.
  - Cons: very small retrieval surface; cannot answer questions about literature that builds on the author's work.

- **Option B: Forward citations — 1-hop**
  - Expand the seed set by fetching papers that *cite* the seed papers via Semantic Scholar API (~60 papers from 12 seeds).
  - Pros: captures research that directly engages with the author's contributions — the most relevant literature to the author's research identity; manageable API call volume within unauthenticated rate limits.
  - Cons: excludes foundational work that predates the seed papers; 1-hop may miss later-generation citations.

- **Option C: Backward references — 1-hop**
  - Expand by fetching what the seed papers cite.
  - Pros: captures foundational literature the author built on.
  - Cons: less directly relevant to the author's specific contributions; foundational papers tend to be widely known and offer less personal research assistance value.

- **Option D: Both directions, 2-hop**
  - Full citation graph expansion in both directions.
  - Pros: most complete literature coverage.
  - Cons: corpus size grows exponentially; unauthenticated S2 API rate limits make this impractical without significant ingest infrastructure.

---

#### Q6: LLM choice

- **Option A: Mistral Medium 3.5**
  - European provider (Mistral AI, Paris); strong instruction-following and structured output; configurable via LangChain's `init_chat_model`; managed API.
  - Pros: European data sovereignty; competitive performance for RAG synthesis tasks; `init_chat_model` makes the model swappable via environment variable without code changes; managed API reduces infrastructure burden for a portfolio project.
  - Cons: managed API (data leaves local environment); not fully open-weights at inference time; lower capability ceiling than frontier models.

- **Option B: Llama 3.x via Ollama / vLLM**
  - Fully open-weights, runs locally.
  - Pros: complete data locality; no API cost; full control over model version.
  - Cons: requires local GPU or accepts slow CPU inference; operational overhead of running a local inference server.

- **Option C: Proprietary frontier model (e.g. GPT-4o, Claude Sonnet)**
  - Highest capability ceiling.
  - Pros: best reasoning quality; mature, stable APIs.
  - Cons: US-domiciled providers (data sovereignty concern); proprietary weights — no local deployment path; vendor lock-in on model roadmap.

---

#### Q7: Observability platform

- **Option A: Langfuse**
  - LLM-native tracing: captures prompts, completions, tool calls, latency, token counts per trace. LangChain `CallbackHandler` integration. European project, self-hostable. Free Cloud tier.
  - Pros: zero instrumentation overhead via LangChain callback; trace model is LLM-native, not generic APM; self-hostable; designed for production RAG debugging (per-trace drill-down, retrieved chunk inspection).
  - Cons: self-hosted v2 incompatible with LangChain v0.3+ callbacks — Cloud version required; less mature experiment comparison features than W&B (relevant for fine-tuning, not for production tracing).

- **Option B: LangSmith**
  - LangChain's own tracing and evaluation platform.
  - Pros: deepest LangChain integration; built-in evaluation tooling.
  - Cons: US-domiciled, no self-hosting option; ties observability stack to a single vendor.

- **Option C: W&B Weave**
  - W&B's dedicated LLM observability product (distinct from core W&B experiment tracking). LLM-native tracing: prompt/completion capture, tool call chains, evaluation. LangChain integration via `WeaveTracer`.
  - Pros: LLM-native trace model; LangChain integration; backed by W&B's mature platform; free tier.
  - Cons: US-domiciled (San Francisco); no self-hosting option — managed only; data sovereignty concern for regulated deployments.

- **Option D: Custom logging**
  - Structured logs to stdout or a log aggregator.
  - Pros: no external dependency; full control.
  - Cons: no UI; no per-trace drill-down; significant implementation cost to replicate what Langfuse provides out of the box.

---

#### Q8: Retrieval agent pattern

- **Option A: ReAct agent**
  - Iterative: reason → select tool → observe result → reason again. Implemented as a LangGraph loop in the retriever node.
  - Pros: handles variable query complexity naturally — a simple query may need one `search_corpus` call; a complex query can chain `search_corpus → get_citations → get_paper_details → search_corpus`; tool selection driven by intermediate observations, not a fixed plan.
  - Cons: non-deterministic number of LLM calls (latency and cost vary per query); requires explicit stopping criteria and tool selection rules to prevent runaway loops.

- **Option B: Plan-and-execute**
  - Agent plans all retrieval steps upfront, then executes the plan.
  - Pros: execution is predictable; plan is inspectable before it runs.
  - Cons: a wrong plan has no mid-execution recovery path; less suited for discovery tasks where intermediate results should change the retrieval direction.

- **Option C: Fixed retrieval pipeline**
  - Always: `search_corpus(query) → top-k → pass to responder`.
  - Pros: fully deterministic; single LLM call; lowest latency.
  - Cons: cannot traverse the citation graph; eliminates the agentic retrieval pattern; reduces to Naive RAG.

---

#### Q9: Out-of-scope handling

- **Option A: Dedicated refuser node**
  - The router classifies queries as `out_of_scope` and routes directly to a refuser node that returns a fixed reply — no LLM call.
  - Pros: zero LLM cost for known out-of-scope queries; fast response; refusal logic is explicit in the graph, not buried in a prompt instruction.
  - Cons: refusal message is static; slightly more complex graph structure (additional node).

- **Option B: Route to responder with refusal instruction**
  - The router sends `out_of_scope` queries to the responder with an instruction to decline.
  - Pros: simpler graph (fewer nodes); refusal can be phrased dynamically per query.
  - Cons: full LLM call and token cost for every out-of-scope query; risk of the model failing to follow the refusal instruction.

---

## Proposal

### Decisions and Alignment

**D1: LangGraph as orchestration framework** `[agreed]`

LangGraph is chosen because the system's control flow cannot be expressed in a linear chain: the router must branch to three different paths (retrieval / direct / refusal), and the retriever must loop (ReAct). LangGraph's StateGraph models both natively. A shared `ResearchState` makes data flow explicit and inspectable at every step.

*What was given up*: faster initial prototyping speed (CrewAI would have been quicker to stand up) and a higher-level declarative abstraction easier for non-engineers to modify. This trade-off is intentional — for a system where execution traces need to be readable and failures diagnosable, implicit control flow is a liability, not a convenience.

*Aligned with: [DBG], [PRD]*

---

**D2: Qdrant as vector store** `[agreed]`

Qdrant is chosen for three compounding reasons: it is self-hostable (satisfying data sovereignty requirements), it supports hybrid search natively in a single query (no external fusion layer needed), and it provides a production-grade ANN index that holds up at scale. The Docker Compose integration is clean for a solo developer and maps directly to a production deployment pattern.

*What was given up*: managed hosting convenience (Pinecone). ChromaDB was considered and rejected — its in-memory model and absence of native hybrid search make it unsuitable for anything beyond prototyping.

*Aligned with: [OSS], [PRD]*

---

**D3: Hybrid retrieval — dense + sparse + RRF** `[agreed]`

Pure dense retrieval loses precision on exact terminology: specific paper IDs, author names, and technical abbreviations may embed similarly to semantically related terms without the specificity needed for recall. Adding a sparse channel restores keyword precision. RRF fuses the two ranked lists without trainable parameters — it is robust, interpretable, and adds no inference overhead beyond the two retrieval passes already running.

A cross-encoder reranker would yield higher precision but at O(n) inference cost per query. For this corpus size, RRF captures most of the precision benefit at a fraction of the latency.

*Aligned with: [PRD], [PBR]*

---

**D4: BM42 over BM25 for sparse embeddings** `[agreed]`

BM25's term weighting is purely statistical — frequency-based, no semantic signal. BM42 uses the same sparse index format but derives term importance from transformer attention weights, giving a modest semantic signal within the sparse channel. Domain-specific terms with variable occurrence patterns get more appropriate weights. BM42 is available in Qdrant's `fastembed` library with no additional inference infrastructure.

*What was given up*: BM25's full interpretability and universal availability. SPLADE was considered and rejected — a dedicated inference server adds operational burden disproportionate to the precision gain at this corpus scale.

*Aligned with: [PRD]*

---

**D5: Forward citation expansion for corpus construction** `[agreed]`

Papers that *cite* the author's work engage directly with the author's contributions — they build on, extend, critique, or apply them. This is the literature most relevant to a researcher's identity and most likely to support useful answers to questions about the author's research area. A 1-hop expansion from ~12 seeds produces a corpus of ~60 papers — large enough for meaningful retrieval, manageable within unauthenticated S2 API rate limits.

*What was given up*: backward references (foundational literature the author built on) and 2-hop coverage. Backward references are valuable but less personal; 2-hop expansion is impractical without an authenticated API key at unauthenticated rate limits.

*Aligned with: [DRL]*

---

**D6: Mistral Medium 3.5 as LLM** `[agreed]`

Mistral AI is a European provider — data processed via their API stays within European jurisdiction. The model performs well on instruction-following and structured output. The implementation uses LangChain's `init_chat_model`, making the model identifier configurable via `LLM_MODEL` environment variable — switching to a different model requires no code changes.

*What was given up*: the capability ceiling of frontier models and full local control (Llama via Ollama). The sovereign cloud deployment pattern this choice demonstrates is directly relevant to regulated-industry deployments where data cannot leave a defined jurisdiction.

*Aligned with: [OSS], [PRD]*

---

**D7: Langfuse for observability** `[agreed]`

Langfuse is purpose-built for LLM tracing: every LLM call, tool call, and token count is captured automatically via LangChain's `CallbackHandler` — no manual instrumentation per node required. The trace model is LLM-native (prompt → completion → tool result chains), not generic APM. It is self-hostable and European in origin.

The closest alternative is W&B Weave — W&B's dedicated LLM observability product, which is LLM-native and has LangChain integration. The deciding factors against it: Weave has no self-hosting option (managed-only) and is US-domiciled, both of which conflict with [OSS].

The data sovereignty concern is more acute for an LLM observability platform than for an experiment tracking platform. W&B core (training metrics) sends numbers — loss curves, validation scores, hyperparameters. An LLM observability platform sends the full content of every trace: user queries, retrieved document chunks, and model completions. In a RAG system, this means the entire corpus passes through the platform at runtime. For a deployment with confidential documents in the corpus — clinical notes, internal policies, financial data — a managed-only observability platform is a hard disqualifier, not a preference. Langfuse's self-hosting option is what makes it viable in those contexts.

*Note*: Langfuse self-hosted v2 is incompatible with LangChain v0.3+ callbacks. The Cloud-hosted version is used for this project — acceptable for a portfolio corpus; self-hosted would be the production choice for sensitive data.

*Aligned with: [DBG], [PRD], [OSS]*

---

**D8: ReAct agent pattern for the retriever** `[agreed]`

A fixed retrieval pipeline is single-shot: retrieve once, pass to responder. The value of the live citation graph tools (`get_citations`, `get_references`, `get_paper_details`) lies precisely in *not* being fixed: find a paper, check who cited it, retrieve a foundational work, search again. ReAct enables this — each tool call is driven by the agent's observation of the previous result.

*What was given up*: fully deterministic query execution and predictable latency. Explicit stopping criteria and tool selection rules were added to the agent prompt to prevent runaway loops and redundant calls — a failure mode encountered during development and corrected through prompt optimization.

*Aligned with: [PRD], [PBR]*

---

**D9: Dedicated refuser node** `[agreed]`

Out-of-scope queries incur no LLM cost. The router classifies intent at minimal token cost (`max_tokens=10`); an `out_of_scope` result routes directly to a static refuser node. This is both cheaper and more reliable than instructing a responder to decline — there is no risk of the model ignoring the refusal instruction and answering anyway.

The underlying principle: expensive operations (LLM calls) should be gated behind cheap classification. This is a production cost-control pattern, not just a portfolio optimisation.

*Aligned with: [PRD], [DBG]*

---

## Future Ideas — Production Readiness

The following are not missing features — they are the delta between a portfolio-ready system and a production-ready one. The answer to "what would you add before putting this in front of real users?" is this list.

- **Evaluation harness**: a labeled query set with expected outputs and automated scoring (RAGAS or LLM-as-judge). Without this, there is no way to detect quality regressions. This is the first production requirement — it separates a system you can monitor from one you have to trust blindly.
- **Async ingestion pipeline**: current ingest is synchronous and single-threaded. Production needs background workers, retry logic on S2 API failures, and corpus freshness monitoring.
- **Graceful degradation**: defined fallback behavior when Qdrant or the Semantic Scholar API is unavailable. Currently: unhandled exception. Production: circuit breaker, fallback to cached results, or a structured error response to the caller.
- **Authentication and rate limiting**: no access control exists. A real deployment needs both at the API layer before any external exposure.
- **Corpus update strategy**: the current corpus is a static snapshot. A live system needs a mechanism to detect new citations and trigger incremental ingestion without a full re-ingest.

## Future Ideas — Research and Product Extensions

> **[TODO — full proposal]** Extend beyond production readiness to research and product directions:
> 
> - 2-hop citation expansion with an authenticated Semantic Scholar API key — significantly larger corpus, manageable with higher rate limits
> - Cross-encoder reranker as the next retrieval upgrade once RRF precision is the bottleneck
> - Multi-author corpus: extend ingest to a research group or lab, not just a single author
> - Query rewriting: expand ambiguous queries before retrieval to improve recall
> - Full-text ingestion: chunk and embed full paper PDFs rather than title + abstract + TLDR only
> - Answer evaluation harness: RAGAS or LLM-as-judge scoring on a held-out query set

## Validation

### Key Assumptions and Risks

| # | Assumption | Risk | Status |
| --- | --- | --- | --- |
| A1 | Hybrid search (dense + BM42 + RRF) outperforms pure dense on this corpus | Medium | Not validated — no labeled query set |
| A2 | ReAct agent terminates reliably with the current stopping criteria | Medium | **Partially validated** ✓ |
| A3 | Mistral Medium 3.5 produces accurate cited answers without hallucinating paper metadata | Medium | Not validated — qualitative observation only |
| A4 | BM42 outperforms BM25 for this academic corpus | Low | Not validated |
| A5 | Forward citations (1-hop) provide sufficient retrieval surface for typical queries | Medium | Not validated |

---

### A2 — Findings

**Verdict: partially validated.** The ReAct agent does not terminate reliably by default. During development, a failure mode was observed: the agent called `get_paper_details` redundantly on papers already retrieved, extending the loop without adding new information. The root cause was an underspecified agent prompt — no explicit rules for when to stop or which tool to call next.

**Fix:** explicit tool selection rules and stopping criteria added to the agent prompt. The rules specify:

- call `search_corpus` first for any new query;
- call `get_paper_details` only for papers not yet seen in the scratchpad;
- stop when sufficient context is available to answer the query without calling another tool.

After prompt optimization, the agent terminates correctly on typical research queries. Adversarial or edge-case inputs (very long queries, queries with no relevant results, ambiguous intent) have not been stress-tested.

---

### Open Risks

**A1 — Retrieval quality (highest priority)**
Hybrid search is expected to outperform pure dense retrieval on exact terminology, but this has not been measured. No labeled query set exists. The system may silently underperform on query types not tested manually — retrieval errors propagate directly into the answer without a grounding check. An evaluation harness (RAGAS or LLM-as-judge on a held-out query set) is the first production requirement that would close this gap.

**A3 — Answer hallucination**
Mistral Medium 3.5 was observed to produce well-grounded cited answers on tested queries. No systematic evaluation has been run. The citation-forcing instruction in the responder prompt is the primary grounding mechanism; its reliability under adversarial prompts or low-quality retrieval is unknown.

**A4 — BM42 vs BM25**
The choice of BM42 over BM25 is theoretically motivated (attention-weighted term importance) but not empirically validated on this corpus. At this scale the practical difference is likely small. Validating this would require A/B comparison on a labeled query set — deferred until A1 is addressed first.

**A5 — Corpus coverage**
1-hop forward citation expansion produces ~60 papers from ~12 seeds. Whether this surface is sufficient for the range of queries a researcher would realistically ask has not been tested. A query that targets a paper outside the 1-hop neighbourhood will miss unless the live S2 API tools cover it.

---

> - Working hypothesis: portfolio demonstration — 2026-05-28
> - Format: Delibera Solution Proposal — incomplete: decisions layer only
> - Full proposal (including complete Problem Frame and Validation sections): planned
