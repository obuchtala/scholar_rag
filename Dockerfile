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

ENTRYPOINT ["uv", "run", "scholar-rag"]
