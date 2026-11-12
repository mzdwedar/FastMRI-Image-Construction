.DEFAULT_GOAL := help
SHELL := /bin/bash

help: ## List targets
	@grep -E '^[a-zA-Z0-9_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-18s %s\n", $$1, $$2}'

setup: ## Create .venv from uv.lock (frozen)
	uv sync --frozen --extra train --extra export --extra gateway

lint: ## ruff check + format check
	uv run ruff check . && uv run ruff format --check .

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

up: ## Start the Compose stack (MLflow :5001 plus serving)
	docker compose up -d

down: ## Stop the Compose stack
	docker compose down

data: ## Download/prepare FastMRI data (added by F9)
	uv run python -m fastmri_knee.data.prepare --root data

train: ## Train: make train ARGS="trainer.train.max_epochs=2"
	uv run python -m fastmri_knee.train $(ARGS)

clean: ## Remove caches and build output (not data, runs or venvs)
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist htmlcov .coverage
	find . -name __pycache__ -not -path './.venv*' -prune -exec rm -rf {} +

.PHONY: help setup lint format typecheck test ci test-integration test-e2e up down data train clean
