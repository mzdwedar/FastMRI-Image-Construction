# Build from the repo root:  docker build -f deploy/services/fastapi/Dockerfile .
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.9.5 /uv /usr/local/bin/uv

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app:/app/src \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH=/app/.venv/bin:$PATH

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first (cached layer): CPU-only torch from the lockfile
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --extra train --extra export --extra gateway --no-install-project

COPY configs/ configs/
COPY src/ src/
COPY monitoring/__init__.py monitoring/monitor_metrics.py monitoring/
COPY deploy/ deploy/

RUN useradd --create-home --uid 1000 app
USER app

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

# The checkpoint is mounted at runtime (see docker-compose.yml), not baked into the image
CMD ["uvicorn", "deploy.services.fastapi.main:app", "--host", "0.0.0.0", "--port", "8000"]
