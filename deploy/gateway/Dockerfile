# Build from the repo root:  docker build -f deploy/gateway/Dockerfile .
# Torch-free: only the `gateway` extra is installed (numpy, h5py, FastAPI, tritonclient, ...).

FROM python:3.15-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.9.5 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --extra gateway --no-install-project

FROM python:3.15-slim
ARG GIT_SHA=unknown
ARG VERSION=0.1.0
LABEL org.opencontainers.image.title="fastmri-knee gateway" \
      org.opencontainers.image.revision=$GIT_SHA \
      org.opencontainers.image.version=$VERSION

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PATH=/app/.venv/bin:$PATH \
    GATEWAY_ENV=prod \
    PROMETHEUS_MULTIPROC_DIR=/dev/shm/prometheus

RUN useradd --no-create-home --uid 10001 --shell /usr/sbin/nologin gateway
WORKDIR /app
COPY --from=build /app/.venv /app/.venv
COPY src/ src/
COPY deploy/gateway/gunicorn.conf.py gunicorn.conf.py
USER 10001

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/healthz', timeout=3)"]

CMD ["gunicorn", "-c", "gunicorn.conf.py", "fastmri_knee.gateway.app:create_app()"]
