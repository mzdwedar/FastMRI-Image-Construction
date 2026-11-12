# Spec Index: fastmri-knee production ML system

Living document. This file is the capability map and the index of module specs; each module has its own
`SPEC-<module-id>.md`. Update the spec before changing direction in code.

## Objective

Turn the fastmri-knee repo (U-Net knee MRI reconstruction) from a working prototype into a production-grade ML
serving system on a single host, with the emphasis on infrastructure: model packaging and promotion, NVIDIA Triton
inference, a FastAPI gateway, observability (metrics, logs, traces, alerts, SLOs), model monitoring, and CI/CD.

**Primary user:** an ML/platform engineer who trains a model, promotes it, and operates the service.
**Secondary user:** an API client (e.g. a viewer) that uploads a single-coil k-space `.h5` volume and receives a
reconstruction.

## Decisions (approved 2026-11-12)

| Decision | Choice |
|---|---|
| Deployment target | Docker Compose on a single host, hardened (no Kubernetes/cloud in scope) |
| Request path | FastAPI = public gateway (validation, preprocessing, auth) → Triton over gRPC. Triton is the only inference engine; the PyTorch path in FastAPI is removed |
| Hardware | CPU for development and CI; GPU variant available via a Compose override, not exercised in CI |
| Model registry | MLflow (Compose service), aliases drive promotion |
| Training | Google Colab (GPU). The repo is installed from git at a pinned commit; data and outputs live on Google Drive; the checkpoint + `run_summary.json` are registered locally (Colab can't reach the local MLflow) |
| Observability stack | Prometheus, Alertmanager, Grafana, Loki, Tempo, Grafana Alloy (log shipping + OTLP), node-exporter, cAdvisor |
| CI | GitHub Actions; images published to GHCR |
| Baseline | Commit `d9abbd6` on branch `refactor/prod-infra` |

## Capability Map

| Module id | Responsibility | Depends on | Spec |
|---|---|---|---|
| `foundation` | Package layout, single env, task runner, test markers, repo cleanup, preprocessing correctness | — | [SPEC-foundation.md](SPEC-foundation.md) |
| `model-packaging` | Checkpoint → ONNX → MLflow registry → versioned Triton model repository; promote / rollback | foundation | [SPEC-model-packaging.md](SPEC-model-packaging.md) |
| `inference-server` | Triton configuration, model control, readiness, batching, resource limits, GPU profile, perf budgets | model-packaging | [SPEC-inference-server.md](SPEC-inference-server.md) |
| `gateway` | Public FastAPI API, numpy preprocessing, Triton gRPC client, auth, limits, backpressure, probes | inference-server | [SPEC-gateway.md](SPEC-gateway.md) |
| `observability` | Metrics, structured logs, traces, alert rules, SLOs, dashboards-as-code | gateway, inference-server | [SPEC-observability.md](SPEC-observability.md) |
| `model-monitoring` | Input/output statistics, drift vs training baseline, label-free quality proxy | observability | [SPEC-model-monitoring.md](SPEC-model-monitoring.md) |
| `ci-cd` | GitHub Actions quality gates, image build/scan/publish, Compose e2e smoke, benchmark regression | foundation (gains a gate per module) | [SPEC-ci-cd.md](SPEC-ci-cd.md) |

Build order: `foundation` → `model-packaging` → `inference-server` → `gateway` → `observability` → `model-monitoring`.
`ci-cd` starts right after `foundation` and gains a gate as each module lands.

## Target architecture

```
                 ┌──────────────────────── docker compose ────────────────────────────┐
 client ──HTTP──▶│ gateway (FastAPI) ──gRPC──▶ triton ◀── model_repository/ (generated)  │
                 │   │  /metrics  logs  OTLP     │ /metrics  OTLP          ▲            │
                 │   ▼                           ▼                         │            │
                 │ alloy ──▶ loki / tempo    prometheus ──▶ alertmanager   │            │
                 │                               ▲  ▲                      │            │
                 │              node-exporter ───┘  └── cadvisor           │            │
                 │ grafana (dashboards as code)                            │            │
                 │ mlflow (registry) ──── `make promote` ──────────────────┘            │
                 └─────────────────────────────────────────────────────────────────────┘
```

## Target repository layout

```
src/fastmri_knee/
  preprocess.py          numpy-only k-space → model input; single source of truth (training + gateway)
  data/                  datasets, datamodule, subject split, download
  model.py, metrics.py   LightningModule, metrics
  train.py evaluate.py tune.py predict.py   Hydra entrypoints
  packaging/             export.py (ONNX + parity), registry.py (MLflow), triton_repo.py (materialize/sync)
  gateway/               app.py, settings.py, triton_client.py, schemas.py, telemetry.py, stats.py
configs/                 Hydra configs (training) + gateway/packaging settings
deploy/
  gateway/Dockerfile
  triton/templates/      config.pbtxt templates (cpu, gpu, nobatch)
  model_repository/      generated by packaging, gitignored
monitoring/
  prometheus/ (prometheus.yml, rules/, tests/)  alertmanager/  alloy/  loki/  tempo/
  grafana/ (provisioning/, dashboards/ generated, build_dashboards.py)
benchmarks/              load_test.py, baselines/*.json
tests/
  unit/                  fast, no network, no data, no Docker
  integration/           real Triton container (Docker required)
  e2e/                   full Compose stack
  training/              needs data/, marker `training`, never in CI
docker-compose.yml       full stack; docker-compose.gpu.yml override; .env.example
Makefile                 every command below
.github/workflows/       ci.yml, e2e.yml, bench.yml, release.yml
```

## Commands (target; the source of truth is `make help`)

```
make setup              uv sync --frozen --extra dev --extra train --extra export --extra gateway
make lint               uv run ruff check . && uv run ruff format --check .
make typecheck          uv run mypy src/fastmri_knee/gateway src/fastmri_knee/packaging src/fastmri_knee/preprocess.py
make test               uv run pytest -m "not integration and not e2e and not training" --cov
make test-integration   uv run pytest -m integration
make test-e2e           uv run pytest -m e2e        (expects `make up` running)
make train              uv run python -m fastmri_knee.train  [hydra overrides]
make register CKPT=...  export ONNX, check parity, register a new MLflow model version
make model-random       register an untrained model (no dataset needed; used by CI and benchmarks)
make promote VERSION=n [ALIAS=production]   point alias at version n and sync Triton
make rollback           re-point `production` at the previous version and sync Triton
make up / make down     docker compose up -d --wait / docker compose down
make up-gpu             docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --wait
make bench              uv run python -m benchmarks.load_test ... ; compare to benchmarks/baselines/
make check-configs      promtool check config/rules, promtool test rules, amtool check-config, compose config
make dashboards         regenerate Grafana JSON (CI fails if the generated JSON differs from what's committed)
```

## Code style

Python 3.12 (see foundation open question), ruff (`E,F,I,B,UP,SIM`), line length 120, type hints required in
`gateway/`, `packaging/`, `preprocess.py` (mypy strict there). Settings via `pydantic-settings`, never `os.environ`
scattered through code. Logging via `structlog` (JSON), never `print`. Comments say *why*, not *what*.

```python
class TritonReconstructor:
    """Sends preprocessed slices to Triton; retries only on UNAVAILABLE, never on bad input."""

    def __init__(self, client: InferenceServerClient, settings: TritonSettings) -> None:
        self._client = client
        self._settings = settings

    async def infer(self, slices: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        if slices.ndim != 4 or slices.shape[1:] != MODEL_INPUT_SHAPE:
            raise InvalidInput(f"expected (N, {', '.join(map(str, MODEL_INPUT_SHAPE))}), got {slices.shape}")
        chunks = np.array_split(slices, math.ceil(len(slices) / self._settings.max_batch_size))
        # Concurrent chunks let Triton's dynamic batcher merge them with other clients' requests
        outputs = await asyncio.gather(*(self._infer_chunk(c) for c in chunks))
        return np.concatenate(outputs)
```

Naming: modules `snake_case`, Prometheus metrics `<component>_<thing>_<unit>` (`gateway_request_duration_seconds`),
Triton model name `unet_model`, MLflow registered model `fastmri-unet`, aliases `production` / `staging`.

## Testing strategy (all modules)

| Level | Marker | Scope | Runs in |
|---|---|---|---|
| Unit | (none) | Pure functions, gateway with a fake Triton client, packaging with a temp MLflow file store | every PR |
| Integration | `integration` | Gateway ↔ real Triton container on a random-init model; registry ↔ Triton sync | every PR (Docker job) |
| E2E | `e2e` | Full Compose stack: request → metrics in Prometheus, trace in Tempo, log in Loki, alert fires | every PR (Docker job) |
| Config | — | `promtool test rules`, `amtool check-config`, `docker compose config`, dashboards regenerate diff | every PR |
| Performance | — | `benchmarks/load_test.py` vs committed baseline | nightly + manual |
| Training | `training` | Real FastMRI data | locally only |

Tests use synthetic FastMRI-shaped `.h5` files and an untrained model, so CI needs neither data nor GPU.
Coverage gate: ≥ 85% line coverage on `gateway/`, `packaging/`, `preprocess.py`; no gate on training code.

## Boundaries (all modules)

- **Always:** run `make lint test` before committing; update the module spec before changing its contract; keep
  the gateway image free of torch; pin every container image to a tag (digest pinning in `release`); write configs
  so `make check-configs` validates them; add a test with every bug fix.
- **Ask first:** changing preprocessing semantics (requires retraining), bumping Triton/Python/MLflow major
  versions, adding a new service to Compose, adding runtime dependencies, deleting user files outside the
  cleanup list, changing public API paths, changing SLO targets.
- **Never:** commit data, checkpoints, ONNX files, `.env`, or secrets; bake models into images; expose Triton
  ports publicly in the prod profile; return stack traces to clients; skip or delete a failing test without
  approval; push to `main` directly.

## System success criteria

1. From a clean clone with Docker and uv: `make setup model-random up` gives a healthy stack in ≤ 10 min, and
   `curl` of `/v1/reconstructions` with a synthetic `.h5` returns a reconstruction.
2. `make promote VERSION=n` and `make rollback` each switch the served model in ≤ 60 s with zero failed requests
   during a concurrent load test, without restarting the gateway or Triton.
3. Stopping Triton fires `TritonDown` in Alertmanager within 2 min, and `/readyz` returns 503 within 10 s.
4. One request appears in all three signals: metric increments in Prometheus, a JSON log line with `request_id`
   and `trace_id` in Loki, and a trace with gateway and Triton spans in Tempo.
5. CI on every PR: lint, typecheck, unit, integration, e2e, config checks, image scan all pass.
6. Gateway image contains no `torch`, and is ≤ 500 MB.

## Open questions (cross-cutting)

1. Preprocessing FFT shift (see SPEC-foundation §Open questions). Fixing it requires retraining; decision blocks
   `gateway` parity tests.
2. Python 3.10 reached end of life in Oct 2026; proposal is 3.12 (blocked on verifying `segmentation-models-pytorch`).
3. `/evaluate` is removed from the serving API and becomes an offline CLI (it needs ground truth and torchmetrics).
   Confirm.
