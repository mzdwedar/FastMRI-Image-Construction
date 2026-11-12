.DEFAULT_GOAL := help
SHELL := /bin/bash

help: ## List targets
	@grep -E '^[a-zA-Z0-9_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-18s %s\n", $$1, $$2}'

setup: ## Create .venv from uv.lock (frozen)
	uv sync --frozen --extra train --extra export --extra gateway

lint: ## ruff check + format check
	uv run ruff check . && uv run ruff format --check . && uv run python scripts/check_notebook_outputs.py

format: ## Auto-format and fix lint
	uv run ruff check --fix . && uv run ruff format .

typecheck: ## mypy (strict on preprocess.py; more modules added as they land)
	uv run mypy

test: ## Unit tests (no Docker, no data)
	uv run pytest -m "not integration and not e2e and not training"

test-integration: ## Tests that need Docker (real Triton)
	uv run pytest -m integration

test-e2e: ## Tests against the running Compose stack
	uv run pytest -m e2e

ci: lint typecheck test ## What CI runs (lint, typecheck, unit tests)

MLFLOW_TRACKING_URI ?= http://localhost:5001

model-random: ## Register an untrained model (random_init=true) in MLflow; no dataset needed
	uv run python -m fastmri_knee.packaging register --random-init --tracking-uri $(MLFLOW_TRACKING_URI)

register: ## Register a trained checkpoint: make register CKPT=best.ckpt [SUMMARY=run_summary.json]
	@test -n "$(CKPT)" || { echo "usage: make register CKPT=<best.ckpt> [SUMMARY=<run_summary.json>]"; exit 2; }
	uv run python -m fastmri_knee.packaging register --checkpoint $(CKPT) \
		$(if $(SUMMARY),--summary $(SUMMARY)) --tracking-uri $(MLFLOW_TRACKING_URI)

sync: ## Load the production model into Triton (needs `make up-dev`)
	uv run python -m fastmri_knee.packaging sync --tracking-uri $(MLFLOW_TRACKING_URI)

ALIAS ?= production

promote: ## Gated promote: make promote VERSION=3 [ALIAS=production] [FORCE=1] [ALLOW_RANDOM=1]
	@test -n "$(VERSION)" || { echo "usage: make promote VERSION=<n>"; exit 2; }
	uv run python -m fastmri_knee.packaging promote --version $(VERSION) --alias $(ALIAS) \
		$(if $(FORCE),--force) $(if $(ALLOW_RANDOM),--allow-random) --tracking-uri $(MLFLOW_TRACKING_URI)

rollback: ## Restore previous_production (skips the PSNR gate)
	uv run python -m fastmri_knee.packaging rollback --tracking-uri $(MLFLOW_TRACKING_URI)

api-key: ## Mint a gateway API key: prints it once, appends its hash to secrets/api_keys
	@uv run python -m fastmri_knee.gateway.auth secrets/api_keys 2>/dev/null

up: ## Start the Compose stack (MLflow :5001 plus serving)
	mkdir -p deploy/model_repository
	docker compose up -d

up-dev: ## Same, and publish Triton HTTP :8100 / gRPC :8101 on the host
	mkdir -p deploy/model_repository
	docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d

down: ## Stop the Compose stack
	docker compose down

data: ## Download/prepare FastMRI data (added by F9)
	uv run python -m fastmri_knee.data.prepare --root data

train: ## Train: make train ARGS="trainer.train.max_epochs=2"
	uv run python -m fastmri_knee.train $(ARGS)

clean: ## Remove caches and build output (not data, runs or venvs)
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist htmlcov .coverage
	find . -name __pycache__ -not -path './.venv*' -prune -exec rm -rf {} +

.PHONY: help setup api-key lint format typecheck test ci test-integration test-e2e model-random register sync promote rollback up up-dev down data train clean
