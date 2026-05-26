FROM python:3.12-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files first (layer caching)
COPY pyproject.toml .
COPY src/ src/

# Install dependencies (no dev deps in prod image)
RUN uv pip install --system --no-cache .

# Pre-download the embedding model so the container is self-contained
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

ENTRYPOINT ["scholar-rag"]
