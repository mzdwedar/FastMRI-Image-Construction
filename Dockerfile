FROM python:3.12-slim

WORKDIR /app

ARG checkpoint_path
ARG experiment_id
ARG run_id

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app \
    checkpoint_path=${checkpoint_path}


# System dependencies (curl for healthcheck)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       curl \
       ca-certificates \
    && rm -rf /var/lib/apt/lists/*



# Copy pyproject.toml and optional poetry.lock or requirements.txt
COPY pyproject.toml /app/

# Install dependencies from pyproject.toml
RUN pip install --upgrade pip \
    && pip install .

# Copy project
COPY configs/ /app/configs/
COPY scripts/ /app/scripts/
COPY src/ /app/src/
COPY mlruns/${experiment_id}/${run_id} /app/mlruns/${experiment_id}/${run_id}

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD curl -fsS http://127.0.0.1:8000/health/ || exit 1

# Start the FastAPI server
CMD python -m scripts.serve \
    --host 0.0.0.0 \
    --port 8000 \
    --checkpoint_path ${checkpoint_path}



