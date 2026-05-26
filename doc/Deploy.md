# Deployment — Option A: Managed Free Tiers

## Philosophy

The local `docker-compose.yml` runs Qdrant and Langfuse as containers. For cloud deployment we swap those two containers for managed services and point the app at them via environment variables. **The application code does not change.**

```
Local:                          Cloud (Option A):
─────────────────────           ─────────────────────────────────
docker compose (Qdrant)    →    Qdrant Cloud   (free tier)
docker compose (Langfuse)  →    Langfuse Cloud (free tier)
uv run scholar-rag         →    uv run scholar-rag  (runs locally,
                                                     talks to cloud)
```

The app itself remains a local CLI — it just points to cloud-hosted services. This is intentional: the project's purpose is to demonstrate agentic RAG, not cloud infrastructure.

---

## Step 1 — Qdrant Cloud

1. Create a free account at **[cloud.qdrant.io](https://cloud.qdrant.io)**

2. Create a new cluster:
   - **Free tier** — 1 cluster, 1 GB storage, hosted on GCP (sufficient for this corpus)
   - Choose a region close to you (Frankfurt is closest for Austria)
   - Cluster name: e.g. `scholar-rag`

3. Once the cluster is ready, go to **Cluster → API Keys** and create a key.

4. Copy the connection details — you'll need:
   - **Cluster URL**: `https://<cluster-id>.<region>.gcp.cloud.qdrant.io:6333`
   - **API Key**: a long random string

5. Update `.env`:
   ```bash
   QDRANT_URL=https://<cluster-id>.<region>.gcp.cloud.qdrant.io:6333
   QDRANT_API_KEY=<your-api-key>
   ```

> The `QdrantClient` in both `ingest.py` and `tools/qdrant_search.py` reads `QDRANT_API_KEY` from the environment and passes it as the `api_key` argument. Local runs with no key set work as before.

---

## Step 2 — Langfuse Cloud

1. Create a free account at **[cloud.langfuse.com](https://cloud.langfuse.com)**

2. Create a new project (e.g. `scholar-rag`)

3. Go to **Settings → API Keys** → create a key pair

4. Update `.env`:
   ```bash
   LANGFUSE_PUBLIC_KEY=pk-lf-...
   LANGFUSE_SECRET_KEY=sk-lf-...
   LANGFUSE_HOST=https://cloud.langfuse.com
   ```

> Tracing is opt-in — if the keys are not set the app runs without errors, just without traces.

---

## Step 3 — Run ingest against the cloud

With `.env` updated, run ingest exactly as before:

```bash
uv run scholar-rag ingest
```

The app embeds locally (sentence-transformers runs on your machine) and upserts to Qdrant Cloud over HTTPS. Ingesting ~60 papers takes 2–3 minutes — the same as local, since the bottleneck is Semantic Scholar API rate limiting, not the vector store.

Verify in the **Qdrant Cloud dashboard**:
- Open your cluster → Collections → `papers`
- You should see ~60 points with `dense` and `sparse` vectors

---

## Step 4 — Ask questions

```bash
uv run scholar-rag ask "What are the main contributions of the evolutionary RBF paper?" --verbose
```

The LangGraph pipeline runs locally; tool calls hit Qdrant Cloud and the Semantic Scholar API. Traces appear in **Langfuse Cloud** at `https://cloud.langfuse.com`.

---

## What the free tiers include

| Service | Free tier limits | Enough for this demo? |
|---|---|---|
| Qdrant Cloud | 1 cluster, 1 GB storage, 1M vectors | ✅ (~60 papers × tiny vectors) |
| Langfuse Cloud | Unlimited traces, 30-day retention | ✅ |

Both free tiers are **persistent** — data survives between sessions, unlike local Docker volumes which reset on `docker compose down -v`.

---

## Environment variable reference

```bash
# .env — cloud configuration
MISTRAL_API_KEY=...           # required

S2_API_KEY=                             # optional
QDRANT_URL=https://<id>.<region>.gcp.cloud.qdrant.io:6333
QDRANT_API_KEY=<qdrant-cloud-api-key>
QDRANT_COLLECTION=papers

LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com

LLM_MODEL=mistralai:mistral-medium-3-5
```

---

## Switching back to local Qdrant

To return to the local Docker Compose Qdrant, restore these lines in `.env`:

```bash
QDRANT_URL=http://host.docker.internal:6333
QDRANT_API_KEY=
```

The collection in Qdrant Cloud is separate from the local one — you'd need to run `ingest` again to populate the local Qdrant.
