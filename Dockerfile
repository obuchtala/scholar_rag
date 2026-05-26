FROM python:3.12-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files and source
COPY pyproject.toml uv.lock ./
COPY src/ src/

# Install dependencies from lockfile (CPU-only torch, no dev deps)
ENV UV_SYSTEM_PYTHON=1
RUN uv sync --no-dev --frozen

# Pre-download the embedding model so the container is self-contained
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

ENTRYPOINT ["scholar-rag"]
