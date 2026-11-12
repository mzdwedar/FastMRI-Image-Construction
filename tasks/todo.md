# Task List: fastmri-knee production ML system

Plan: [plan.md](plan.md). Sizes: XS 1 file · S 1–2 · M 3–5. Every task also meets the Definition of Done:
`make lint test` green, no regressions, behaviour verified, spec updated if the contract changed.
Until F6 lands, "make test" means `uv run pytest` with the current config.
Decisions (`Q`) and human setup steps (`S`) are listed at the [end](#decisions-and-setup) and picked up when the
task that needs them is reached.

---

## Phase 1 — Foundation (SPEC-foundation)

- [x] **F1: Python 3.12 compatibility spike** — S
  - Do: In a throwaway env, install the train deps on 3.12 (smp 0.3.3, then 0.3.4+ if needed); save a
    random-init checkpoint under 3.10 and load it under 3.12, one forward pass each; run the serving tests; note
    Colab's current Python version (Q2).
  - Acceptance: decision (3.12 / 3.11 / stay) with evidence recorded in SPEC-foundation open question 2.
  - Verify: forward-pass output from the 3.10-saved checkpoint matches the 3.10 output (`atol=1e-5`).
  - Deps: S1 · Files: `SPEC-foundation.md`

- [ ] **F2: Verify FFT shift on a real slice** — S
  - Do: For one real `.h5` file (local `data/` is empty: get one from Drive per S11, or run this check in Colab), plot input from current `FastMRITransform` vs shifted variant vs
    `reconstruction_rss`; compute normalized cross-correlation of each with the target.
  - Acceptance: figure `docs/img/fftshift-check.png` + NCC numbers recorded in SPEC-foundation; decision request to user.
  - Verify: manual inspection of the figure.
  - Deps: S11 · Files: `docs/img/fftshift-check.png`, `SPEC-foundation.md`

- [ ] **F3: Repo cleanup and test discovery** — S
  - Do: delete `build/ preds/ results/ triton_models/ lightning_logs/ unet_model.onnx src/*.egg-info split_data.py`;
    untrack `grafana_latency.png` (→ `docs/img/` if README uses it); move pytest config into `pyproject.toml`
    with markers `integration`, `e2e`, `training`, `testpaths=["tests"]`; mark data-dependent tests `training`.
  - Acceptance: all three test dirs collected; `-m "not training"` passes without `data/`.
  - Verify: `uv run pytest --collect-only -q`; `uv run pytest -m "not training"`.
  - Deps: Q3 · Files: `pyproject.toml`, `pytest.ini`(del), `tests/data/test_data.py`, `tests/code/test_train.py`, `.gitignore`

- [x] **F4: Move library code into `src/fastmri_knee/`** — M (pure refactor)
  - Do: `git mv` `src/{data,model,metrics}.py` → `src/fastmri_knee/{data/__init__.py…}`; update imports in
    scripts, deploy, tests; hatchling build config so `uv sync` installs the package.
  - Acceptance: no behaviour change; `import fastmri_knee` works without `pythonpath`.
  - Verify: full test suite passes with `pythonpath` removed.
  - Deps: none · Files: `src/fastmri_knee/*`, `pyproject.toml`, imports in `scripts/*`, `tests/*/conftest.py`

- [x] **F5: Move scripts into package entrypoints** — M (pure refactor)
  - Do: `scripts/{train,evaluate,tune,predict}.py` → `fastmri_knee/{train,…}.py` (Hydra `config_path` absolute
    to repo `configs/`); `scripts/convert_to_onnx.py` → `fastmri_knee/packaging/export.py`; delete
    `scripts/predict_fastapi.py`, `predict_triton.py` (superseded by gateway), `scripts/` removed.
  - Acceptance: `python -m fastmri_knee.train --help` and `python -m fastmri_knee.packaging.export --random-init` work.
  - Verify: serving + unit tests pass; `training` test still collects.
  - Deps: F4 · Files: `src/fastmri_knee/{train,evaluate,tune,predict}.py`, `src/fastmri_knee/packaging/export.py`, `tests/serving/*`

- [ ] **F6: Dependency extras, single env, Makefile** — M
  - Do: split extras `train/export/gateway`, core `numpy,h5py`, `[dependency-groups] dev`; set Python per F1;
    regenerate `uv.lock`; Makefile targets from SPEC-foundation; `.env.example`; delete `.venv-cpu/-dev/-gpu` (ask).
  - Acceptance: `rm -rf .venv && make setup` works from the lockfile; `make help` lists targets.
  - Verify: `make setup lint test` on a clean checkout (macOS) and in `python:3.12` Docker (Linux).
  - Deps: F1, F4, Q2, Q4 · Files: `pyproject.toml`, `uv.lock`, `.python-version`, `Makefile`, `.env.example`

- [x] **F7: numpy preprocessing, shared by training** — M
  - Do: `fastmri_knee/preprocess.py` (`PREPROCESS_VERSION="1"`, exact current semantics incl. normalize-then-crop,
    optional mask); `FastMRITransform` calls it; tests for parity and no-torch import.
  - Acceptance: parity with the old torch transform `atol=1e-6`; imports with torch blocked; float32, `(1,320,320)`, range [0,1], no NaN on zero k-space.
  - Verify: `uv run pytest tests/unit/test_preprocess*.py`.
  - Deps: F4 · Files: `src/fastmri_knee/preprocess.py`, `src/fastmri_knee/data/…`, `tests/unit/test_preprocess.py`, `tests/unit/test_preprocess_no_torch.py`

- [ ] **F8: Fix FFT shift → `PREPROCESS_VERSION="2"`** — S · **GATED on Q1**
  - Do: add `ifftshift`/`fftshift`; phantom test (centre of mass inside crop); update parity test to v2.
  - Acceptance: synthetic phantom 100% inside crop; v1 path deleted (no dual support).
  - Verify: `uv run pytest tests/unit/test_preprocess*.py tests/unit/test_fftshift.py`.
  - Deps: F2, F7, Q1 · Files: `src/fastmri_knee/preprocess.py`, `tests/unit/test_fftshift.py`, `tests/unit/test_preprocess.py`

- [ ] **F9: Data acquisition CLI out of the training path** — S
  - Do: move download/extract/split from `FastMRIDataModule.prepare_data` to `fastmri_knee.data.prepare`
    (`make data`); `prepare_data` only checks presence and raises a clear error.
  - Acceptance: training never shells out to `aria2c`; missing data → actionable error message.
  - Verify: unit test of the missing-data error; `make data --help`.
  - Deps: F4 · Files: `src/fastmri_knee/data/prepare.py`, `src/fastmri_knee/data/…`, `tests/unit/test_data_prepare.py`, `Makefile`

- [ ] **F10: Lint/format/type tooling and pre-commit** — S
  - Do: ruff rules `E,F,I,B,UP,SIM` + `ruff format` (drop black); mypy config (strict on `preprocess.py`,
    `packaging/`, later `gateway/`); `.pre-commit-config.yaml` with ruff + gitleaks; fix findings.
  - Acceptance: `make lint typecheck` clean.
  - Verify: `make lint typecheck`; `pre-commit run --all-files`.
  - Deps: F6 · Files: `pyproject.toml`, `.pre-commit-config.yaml`, files with lint fixes

### ✅ Checkpoint: Foundation
- [ ] `rm -rf .venv && make setup lint typecheck test` green on macOS and Linux
- [ ] `git status` clean after tests; FFT + Python decisions recorded in SPEC-foundation
- [ ] Human review; PR `foundation` → `refactor/prod-infra`

---

## Phase 2 — CI bootstrap (SPEC-ci-cd)

- [ ] **C1: `ci.yml` with lint, typecheck, unit, security** — M
  - Do: setup-uv with lock-keyed cache; jobs call `make lint|typecheck|test`; `security` job: gitleaks +
    `pip-audit` on exported requirements; `actionlint`; SHA-pinned actions; `contents: read`; concurrency group.
  - Acceptance: green on the branch; unit job ≤ 5 min.
  - Verify: push; drill PR with a failing test and a lint error fails the right jobs.
  - Deps: F6, F10, S4 · Files: `.github/workflows/ci.yml`, `Makefile` (`ci` target)

- [ ] **C2: Dependency update automation** — XS
  - Do: Dependabot (or Renovate) for actions, Docker, uv; weekly, grouped.
  - Acceptance/Verify: config validates (GitHub UI shows it parsed).
  - Deps: C1, S7 · Files: `.github/dependabot.yml`

### ✅ Checkpoint: CI bootstrap — `ci.yml` green; drills fail correctly

---

## Phase 3 — Model packaging + Triton base (SPEC-model-packaging, SPEC-inference-server §1–3)

- [ ] **P1: MLflow tracking/registry service** — S
  - Do: `mlflow` service in Compose (pinned, SQLite + artifact volume, host :5001); bump MLflow to current 2.x;
    training uses `MLFLOW_TRACKING_URI` when set (file store fallback kept).
  - Acceptance: `make up` starts MLflow; a 1-step training run on synthetic data logs to it.
  - Verify: `curl localhost:5001/health`; run appears in UI.
  - Deps: F6, Q6 · Files: `docker-compose.yml`, `.env.example`, `pyproject.toml`/`uv.lock`, `configs/trainer.yaml`

- [ ] **P2: `register` command with model card** — M
  - Do: `packaging/registry.py`: export → parity (refuse > 1e-4) → `model_card.json` (all spec fields) →
    new version of `fastmri-unet` with onnx, ckpt, card; tags `preprocess_version`, `git_sha`.
  - Acceptance: unit tests on a temp file store: artifacts present, card fields correct, parity failure refuses.
  - Verify: `uv run pytest tests/unit/packaging/test_register.py`.
  - Deps: F5, F7 · Files: `src/fastmri_knee/packaging/{registry,card,__main__}.py`, `tests/unit/packaging/test_register.py`

- [ ] **P3: `make model-random`** — S
  - Do: register untrained weights tagged `random_init=true`; Makefile target.
  - Acceptance: works with no dataset; version tagged.
  - Verify: unit test; `make model-random` against Compose MLflow.
  - Deps: P2 · Files: `src/fastmri_knee/packaging/registry.py`, `Makefile`, `tests/unit/packaging/test_register.py`

- [ ] **P4: Triton config templates and renderer** — S
  - Do: `deploy/triton/templates/{cpu,gpu,nobatch}.pbtxt` from current configs with variables (version, threads,
    instances, queue delay); `packaging/triton_repo.render()`; golden-file tests; remove `deploy/services/triton/models/`.
  - Acceptance: rendered cpu config equals today's `unet_model` config plus `version_policy specific`.
  - Verify: `uv run pytest tests/unit/packaging/test_render.py`.
  - Deps: F5 · Files: `deploy/triton/templates/*.pbtxt`, `src/fastmri_knee/packaging/triton_repo.py`, `tests/unit/packaging/test_render.py`, `tests/unit/packaging/golden/*`

- [ ] **T1: Triton explicit mode on an empty repo (+ arm64 spike)** — S
  - Do: check native arm64 manifest for the pinned tag (record result); flags per SPEC-inference-server §1;
    mount `deploy/model_repository` read-only; `TRITON_IMAGE/CPUS/MEM` in `.env`; `dev` profile publishes 8100/8101;
    default profile publishes none.
  - Acceptance: empty repo → live 200, ready 503; arm64 finding + fallback recorded in plan risks.
  - Verify: `make up`; `docker compose exec triton curl -s -o /dev/null -w '%{http_code}' localhost:8000/v2/health/ready` → 503.
  - Deps: P4, Q7, Q8, S2 · Files: `docker-compose.yml`, `.env.example`, `.gitignore`, `Makefile`

- [ ] **P5: `sync` command** — M
  - Do: resolve `production` alias → version; download + sha256 check; render config; atomic write; prune to
    current + previous; Triton load API; wait ready (60 s); on failure restore previous config, reload, exit 1.
  - Acceptance: unit tests (Triton HTTP mocked with `respx`): idempotent, sha mismatch aborts, load failure restores.
  - Verify: `uv run pytest tests/unit/packaging/test_sync.py`.
  - Deps: P2, P4, T1 · Files: `src/fastmri_knee/packaging/{triton_repo,sync}.py`, `tests/unit/packaging/test_sync.py`

- [ ] **P6: `promote` and `rollback` with gates** — M
  - Do: gates (preprocess mismatch, PSNR −0.5 dB, random-init → production unless `--allow-random`), `--force`
    logged; `previous_production` tag; rollback skips PSNR gate; Makefile targets.
  - Acceptance: each gate refuses with a clear message; rollback restores previous.
  - Verify: `uv run pytest tests/unit/packaging/test_promote.py`.
  - Deps: P5 · Files: `src/fastmri_knee/packaging/promote.py`, `src/fastmri_knee/packaging/__main__.py`, `Makefile`, `tests/unit/packaging/test_promote.py`

- [ ] **P7: Packaging integration tests against Triton** — M
  - Do: session fixture runs Triton (`docker run`, empty repo, explicit mode) + temp MLflow; tests: promote v1 → v2
    → rollback; corrupt ONNX → v2 keeps serving.
  - Acceptance: tests pass locally (or in CI if arm64 fallback applies).
  - Verify: `make test-integration`.
  - Deps: P6 · Files: `tests/integration/conftest.py`, `tests/integration/test_packaging_triton.py`

### ✅ Checkpoint: Packaging — `make up model-random && make promote VERSION=1 ALIAS=production FORCE=1` → Triton serves v1; P7 green

---

## Phase 4 — Inference server (SPEC-inference-server)

- [ ] **T2: Model config hardening** — S
  - Do: templates gain `default_queue_policy` (REJECT, 5 s), `max_queue_size 256`, `model_warmup` (batch 1 and
    16), instance-count variable; threads from `TRITON_CPUS`.
  - Acceptance: golden files updated; Triton loads the config.
  - Verify: `uv run pytest tests/unit/packaging/test_render.py`; `make sync` loads.
  - Deps: P4 · Files: `deploy/triton/templates/*.pbtxt`, `tests/unit/packaging/golden/*`

- [ ] **T3: Triton integration tests** — M
  - Do: tests: strict readiness (empty → 503), output equals onnxruntime (`atol=1e-5`), overload (512 concurrent)
    rejected while server stays live.
  - Acceptance/Verify: `make test-integration` green.
  - Deps: T2, P7 · Files: `tests/integration/test_triton.py`, `tests/integration/conftest.py`

- [ ] **T4: Benchmark baseline and comparison** — M
  - Do: adapt `benchmarks/load_test.py` to gRPC/HTTP via dev profile; `make bench`, `make bench-compare`;
    compare 1×4 vs 2×2 instances×threads and nobatch; commit `benchmarks/baselines/cpu.json`; set default.
  - Acceptance: batching ≥ 1.5× nobatch at concurrency 16; budgets in SPEC-inference-server filled with numbers.
  - Verify: `make bench bench-compare`; results attached to PR.
  - Deps: T2, P5 · Files: `benchmarks/load_test.py`, `benchmarks/compare.py`, `benchmarks/baselines/cpu.json`, `Makefile`, `SPEC-inference-server.md`

- [ ] **T5: GPU Compose override** — S
  - Do: `docker-compose.gpu.yml` (nvidia device reservation, `TRITON_PROFILE=gpu`); README manual check section.
  - Acceptance/Verify: `docker compose -f docker-compose.yml -f docker-compose.gpu.yml config` validates.
  - Deps: T1 (manual GPU check: S8, deferred) · Files: `docker-compose.gpu.yml`, `README.md`

- [ ] **C3: Integration job in CI** — S
  - Do: `e2e.yml` with `integration` job (Docker available, image cache).
  - Acceptance: green; required check.
  - Verify: push; drill with a broken template fails it.
  - Deps: T3, C1 · Files: `.github/workflows/e2e.yml`

### ✅ Checkpoint: Inference server — integration in CI; baseline committed; overload test passes

---

## Phase 5 — Gateway (SPEC-gateway)

- [ ] **G1: Gateway skeleton** — M
  - Do: `gateway/app.py` (app factory, lifespan), `settings.py` (pydantic-settings, `GATEWAY_` prefix), request-id
    middleware, problem+json handlers, `/healthz`, structlog JSON baseline.
  - Acceptance: invalid settings fail startup; unknown errors → 500 problem+json without internals; `X-Request-ID` echoed/generated.
  - Verify: `uv run pytest tests/unit/gateway/test_app.py`.
  - Deps: F6, F10 · Files: `src/fastmri_knee/gateway/{app,settings,errors,logging}.py`, `tests/unit/gateway/test_app.py`

- [ ] **G2: Triton client and `/v1/infer` (first end-to-end slice)** — M
  - Do: `triton_client.py` (grpc.aio, chunking by max_batch_size from model config, retry UNAVAILABLE only,
    deadlines, finite-output check); `/v1/infer` route; `FakeTritonClient` for tests.
  - Acceptance: chunking 1/16/17/64; retry vs no-retry codes; deadline → 504; non-finite → 500; N > 64 → 413.
  - Verify: `uv run pytest tests/unit/gateway/test_infer.py`; manual curl against `make up` Triton.
  - Deps: G1, P6 · Files: `src/fastmri_knee/gateway/{triton_client,routes}.py`, `tests/unit/gateway/{fakes,test_infer}.py`

- [ ] **G3: Readiness loop, `/readyz`, `/v1/model`** — S
  - Do: background task polls model readiness + version every 2 s; `/readyz` reads cache with `{reason}`;
    `/v1/model` returns version + card (from registry tag or Triton metadata); `X-Model-Version` header.
  - Acceptance: not-ready reasons `triton_unreachable` / `model_not_ready`; flips within 10 s.
  - Verify: `uv run pytest tests/unit/gateway/test_readiness.py`.
  - Deps: G2 · Files: `src/fastmri_knee/gateway/{readiness,routes}.py`, `tests/unit/gateway/test_readiness.py`

- [ ] **G4: `/v1/reconstructions`** — M
  - Do: streamed upload to temp file with byte limit; h5 validation (dataset, rank, dtype, slice count); numpy
    preprocess in threadpool; optional `acceleration`; response h5 (`reconstruction` + attrs); cleanup on
    success/error/disconnect.
  - Acceptance: 400/413/415/422 cases; 300 MB stream rejected without buffering; temp dir empty after every case.
  - Verify: `uv run pytest tests/unit/gateway/test_reconstructions.py`.
  - Deps: G2, F7, Q5, Q10 · Files: `src/fastmri_knee/gateway/{uploads,routes,schemas}.py`, `tests/unit/gateway/test_reconstructions.py`, `tests/fixtures/` helper

- [ ] **G5: API-key auth and backpressure** — S
  - Do: hashed keys file, constant-time compare, dev-only bypass (startup error otherwise); global semaphore
    `MAX_INFLIGHT` → 503 + `Retry-After: 1`.
  - Acceptance: 401 cases; bypass outside dev fails startup; saturation returns 503 immediately.
  - Verify: `uv run pytest tests/unit/gateway/test_auth.py tests/unit/gateway/test_backpressure.py`.
  - Deps: G4, S6 · Files: `src/fastmri_knee/gateway/{auth,limits}.py`, `tests/unit/gateway/test_auth.py`, `tests/unit/gateway/test_backpressure.py`

- [ ] **G6: Gateway image and Compose wiring; remove old backend** — M · needs Q9
  - Do: `deploy/gateway/Dockerfile` (multi-stage uv, 3.12-slim, uid 10001, labels), gunicorn+uvicorn workers,
    Compose service (read-only rootfs, tmpfs `/tmp`, secrets file, healthcheck); delete
    `deploy/services/fastapi/`, `tests/serving/test_api.py`, `monitoring/monitor_metrics.py` usage by old app.
  - Acceptance: `make up` → gateway healthy; `curl /v1/reconstructions` works through Triton.
  - Verify: `make up && curl …` with synthetic h5; `docker compose ps` all healthy.
  - Deps: G5, Q9, S6 · Files: `deploy/gateway/Dockerfile`, `docker-compose.yml`, `.env.example`, `deploy/services/fastapi/*`(del), `tests/serving/test_api.py`(del)

- [ ] **G7: Gateway integration, contract, and image tests** — M
  - Do: integration: reconstruction equals local ORT on same input (`atol=1e-5`), `/readyz` flips when Triton
    stops; OpenAPI snapshot; image checks (no torch, ≤ 500 MB, non-root).
  - Acceptance/Verify: `make test test-integration` green; coverage ≥ 85% on `gateway/`.
  - Deps: G6 · Files: `tests/integration/test_gateway.py`, `tests/unit/gateway/test_openapi.py`, `tests/unit/openapi.json`, `tests/integration/test_image.py`

- [ ] **C4: e2e job and first e2e test** — M
  - Do: `make ci-e2e` (build, up --wait, model-random, promote, test-e2e, down -v); `tests/e2e/test_smoke.py`
    (reconstruct a synthetic volume); upload compose logs on failure; measure wall time.
  - Acceptance: e2e green in CI ≤ 12 min.
  - Verify: push; drill with a broken gateway route fails it.
  - Deps: G6, C3 · Files: `.github/workflows/e2e.yml`, `Makefile`, `tests/e2e/conftest.py`, `tests/e2e/test_smoke.py`

- [ ] **P8: E2E promotion under load** — S
  - Do: e2e test: load test against the gateway (20 s, concurrency 8) while promoting v2 then rolling back.
  - Acceptance: zero failed requests; each switch ≤ 60 s.
  - Verify: `make test-e2e`.
  - Deps: C4, P6 · Files: `tests/e2e/test_promotion.py`

### ✅ Checkpoint: Gateway — SPEC.md system criteria 1, 2, 6 pass; Triton-kill test (in-flight get 503/504, recovers); human review

---

## Phase 6 — Observability (SPEC-observability)

- [ ] **O1: Gateway metrics (multiprocess)** — M
  - Do: metrics table from spec in `gateway/telemetry.py`; `PROMETHEUS_MULTIPROC_DIR` on tmpfs; `/metrics`;
    remove `prometheus-fastapi-instrumentator`, delete `monitoring/monitor_metrics.py` (psutil).
  - Acceptance: names/labels match spec exactly; correct counts with 2 workers; no unbounded labels.
  - Verify: `uv run pytest tests/unit/gateway/test_metrics.py`.
  - Deps: G6 · Files: `src/fastmri_knee/gateway/telemetry.py`, `src/fastmri_knee/gateway/app.py`, `monitoring/monitor_metrics.py`(del), `pyproject.toml`, `tests/unit/gateway/test_metrics.py`

- [ ] **O2: Structured access log** — S
  - Do: one JSON line per request with spec fields; uvicorn access log off; key id hash only.
  - Acceptance: exactly one line per request with matching `request_id`; no key/filename leakage.
  - Verify: `uv run pytest tests/unit/gateway/test_logging.py`.
  - Deps: O1 · Files: `src/fastmri_knee/gateway/logging.py`, `tests/unit/gateway/test_logging.py`

- [ ] **O3: Alertmanager, alert-sink, node-exporter, cAdvisor; config checks** — M · needs Q11
  - Do: Compose services (pinned); Prometheus scrape jobs + `rule_files`; `alertmanager.yml` (sink default,
    Slack when env set); `make check-configs` (promtool, amtool, `docker compose config`).
  - Acceptance: all targets `up` in Prometheus; `make check-configs` passes.
  - Verify: `make up check-configs`; Prometheus `/targets`.
  - Deps: G6, Q11, S2 (S5 if Slack) · Files: `docker-compose.yml`, `monitoring/prometheus/prometheus.yml`, `monitoring/alertmanager/alertmanager.yml`, `monitoring/alert-sink/` , `Makefile`

- [ ] **O4: Symptom and infra alerts with rule tests** — M
  - Do: rules for GatewayDown, TritonDown, ModelNotReady, TritonQueueSaturation, GatewayRejectingLoad,
    ContainerMemoryHigh, ContainerCPUThrottled, meta-monitoring; promtool tests firing + not firing each.
  - Acceptance/Verify: `promtool test rules monitoring/prometheus/tests/*.yml` green.
  - Deps: O1, O3 · Files: `monitoring/prometheus/rules/{service,infra}.yml`, `monitoring/prometheus/tests/{service,infra}_test.yml`

- [ ] **O5: SLO recording rules and burn-rate alerts** — M
  - Do: SLI recording rules (5m/30m/1h/6h) for the three SLOs; fast/slow burn alerts; promtool tests with
    synthetic series.
  - Acceptance: burn-rate math verified by tests at 14.4× and 6×.
  - Verify: `make check-configs`.
  - Deps: O4, Q12 · Files: `monitoring/prometheus/rules/slo.yml`, `monitoring/prometheus/tests/slo_test.yml`

- [ ] **O6: Runbooks** — S
  - Do: `docs/runbooks/<Alert>.md` for every alert (symptoms, dashboards, causes, commands); `runbook_url` annotation
    on every rule; test that every alert has an existing runbook file.
  - Acceptance/Verify: `uv run pytest tests/unit/test_runbooks.py`.
  - Deps: O5 · Files: `docs/runbooks/*.md`, `tests/unit/test_runbooks.py`

- [ ] **O7: Loki and Alloy log pipeline** — S
  - Do: Loki + Alloy services (Docker discovery, labels `service`,`level`); Grafana Loki datasource.
  - Acceptance: gateway log lines queryable in Loki by `request_id`.
  - Verify: `make up`; `logcli`/Grafana Explore query.
  - Deps: O2, O3 · Files: `docker-compose.yml`, `monitoring/alloy/config.alloy`, `monitoring/loki/config.yml`, `monitoring/grafana/provisioning/datasources/*.yml`

- [ ] **O9: Spike — Triton OpenTelemetry context propagation** — S
  - Do: enable `--trace-config mode=opentelemetry` to a local collector; send gRPC request with `traceparent`
    metadata; confirm Triton spans share the trace id.
  - Acceptance: finding recorded in SPEC-observability (works / fallback chosen).
  - Verify: manual, trace viewed in Tempo or collector logs.
  - Deps: T1 · Files: `SPEC-observability.md`

- [ ] **O8: Tracing: gateway OTel, Tempo, tail sampling, trace ids in logs** — M
  - Do: OTel SDK + FastAPI instrumentation + manual spans; propagate to Triton (per O9); Tempo service; Alloy OTLP
    receiver + tail sampling (errors, > 2 s); `trace_id`/`span_id` in log lines; Grafana Tempo datasource with
    trace↔log links.
  - Acceptance: one request → trace with gateway spans (+ Triton spans if O9 succeeded); log line carries its trace_id.
  - Verify: `uv run pytest tests/unit/gateway/test_tracing.py` (in-memory exporter); manual Tempo check.
  - Deps: O7, O9 · Files: `src/fastmri_knee/gateway/telemetry.py`, `docker-compose.yml`, `monitoring/tempo/config.yml`, `monitoring/alloy/config.alloy`, `tests/unit/gateway/test_tracing.py`

- [ ] **O10: Dashboards as code** — M
  - Do: refactor `build_dashboards.py`; Service overview, Triton, Infrastructure dashboards with spec panels and
    cross-links; `make dashboards`; remove old dashboards.
  - Acceptance: dashboards load with live data; regenerated JSON identical to committed.
  - Verify: `make dashboards && git diff --exit-code`; open Grafana.
  - Deps: O1, O3 · Files: `monitoring/grafana/build_dashboards.py`, `monitoring/grafana/dashboards/*.json`, `monitoring/grafana/provisioning/dashboards/dashboards.yml`, `Makefile`

- [ ] **O11: Observability e2e tests** — M
  - Do: e2e: 10 requests → counter ≥ 10; Loki has the log for a known request_id; Tempo has its trace; stop Triton →
    `TritonDown` at alert-sink ≤ 2 min, `/readyz` 503 ≤ 10 s; every dashboard query returns no error.
  - Acceptance/Verify: `make test-e2e` green locally and in CI.
  - Deps: O4, O8, O10 · Files: `tests/e2e/test_observability.py`, `tests/e2e/conftest.py`

- [ ] **C5: Configs job in CI** — S
  - Do: `configs` job: `make check-configs`, dashboards diff, hadolint, actionlint; make it required.
  - Acceptance/Verify: green; drill with a broken alert rule fails it.
  - Deps: O3, C1 · Files: `.github/workflows/ci.yml`

### ✅ Checkpoint: Observability — SPEC.md criteria 3, 4 pass in e2e; all alerts have tests + runbooks; human review

---

## Phase 7 — Model monitoring (SPEC-model-monitoring)

- [ ] **M1: Stats module** — M
  - Do: `gateway/stats.py`: input/output stats and data-consistency error (numpy); golden-value tests;
    performance test (≤ 2% of a 35-slice request).
  - Acceptance: DC error ≈ 0 for perfect reconstruction, monotonic with noise.
  - Verify: `uv run pytest tests/unit/gateway/test_stats.py`.
  - Deps: O1, F7, Q13 · Files: `src/fastmri_knee/gateway/stats.py`, `tests/unit/gateway/test_stats.py`

- [ ] **M2: Stats metrics in the request path** — S
  - Do: histograms `model_input_stat`, `model_output_stat`, `model_data_consistency_error`, shape-other counter;
    labelled by `model_version`.
  - Acceptance: emitted per slice; bounded labels.
  - Verify: `uv run pytest tests/unit/gateway/test_metrics.py`.
  - Deps: M1 · Files: `src/fastmri_knee/gateway/{telemetry,routes}.py`, `tests/unit/gateway/test_metrics.py`

- [ ] **M3: Baseline stats at registration; baseline gauges in gateway** — M
  - Do: `register` computes `baseline_stats.json` (val split or synthetic, `baseline_source` tag); gateway
    loads it for the served version and exports `model_baseline_quantile`.
  - Acceptance: every new version has the artifact; gauges match file after promote.
  - Verify: `uv run pytest tests/unit/packaging/test_register.py tests/unit/gateway/test_baseline.py`.
  - Deps: P2, M1 · Files: `src/fastmri_knee/packaging/{registry,baseline}.py`, `src/fastmri_knee/gateway/readiness.py`, `tests/unit/gateway/test_baseline.py`

- [ ] **M4: Drift and quality alert rules** — S
  - Do: recording rules for live p50; `ModelInputDrift`, `ModelOutputDrift`, `ModelOutputNonFinite`,
    `ModelQualityProxyDegraded`; promtool tests incl. < 100 slices quiet; runbooks.
  - Acceptance/Verify: `make check-configs` green.
  - Deps: M2, M3, O6 · Files: `monitoring/prometheus/rules/model.yml`, `monitoring/prometheus/tests/model_test.yml`, `docs/runbooks/Model*.md`

- [ ] **M5: Model dashboard and promotion annotations** — S
  - Do: "Model" dashboard (live vs baseline bands, DC error, per-version slices); packaging posts Grafana
    annotations on promote/rollback.
  - Acceptance: dashboard loads; annotation appears after `make promote`.
  - Verify: `make dashboards && git diff --exit-code`; manual promote check.
  - Deps: M4, O10 · Files: `monitoring/grafana/build_dashboards.py`, `monitoring/grafana/dashboards/model.json`, `src/fastmri_knee/packaging/promote.py`

- [ ] **M6: Drift demo and e2e** — S
  - Do: `make drift-demo` sends scaled/noisy synthetic volumes; e2e with shortened `for:` override asserts
    `ModelInputDrift` at alert-sink.
  - Acceptance/Verify: `make test-e2e` green.
  - Deps: M5 · Files: `benchmarks/drift_demo.py`, `Makefile`, `tests/e2e/test_drift.py`, `monitoring/prometheus/rules/overrides-test.yml`

### ✅ Checkpoint: Model monitoring — drift e2e green; `make bench` shows ≤ 2% p95 regression from stats

---

## Phase 8 — Release and hardening (SPEC-ci-cd, docs)

- [ ] **C6: `release.yml`** — M · needs Q14
  - Do: on tag `v*.*.*`: build amd64 gateway image, trivy (fail CRITICAL/HIGH with fix), syft SBOM to release,
    push `ghcr.io/<owner>/fastmri-gateway:{version,sha}`, git-cliff changelog; `packages: write` only here.
  - Acceptance: `v0.2.0-rc1` tag publishes; `GATEWAY_IMAGE=ghcr.io/…:0.2.0-rc1 make up` runs it.
  - Verify: tag on the branch; pull and run.
  - Deps: G6, C4, Q14, S9 · Files: `.github/workflows/release.yml`, `cliff.toml`, `.env.example`

- [ ] **C7: `bench.yml` nightly (advisory)** — S
  - Do: nightly + manual; runs `make bench bench-compare`; job summary; fails only on > 30% regression.
  - Acceptance/Verify: manual dispatch succeeds and posts a summary.
  - Deps: T4, C4 · Files: `.github/workflows/bench.yml`

- [ ] **C8: Drills and contribution docs** — S
  - Do: run broken-PR drills for every required job (test, rule, Dockerfile lint, fake secret, broken e2e) and
    record results; `docs/contributing.md` with branch protection settings and required checks.
  - Acceptance: every drill fails the expected job.
  - Verify: links to drill runs in the PR.
  - Deps: C1–C6, S10 · Files: `docs/contributing.md`

- [ ] **D1: README rewrite** — S
  - Do: architecture diagram (target), quickstart (`make setup model-random up`), operations (promote/rollback,
    dashboards, alerts, runbooks), GPU section, links to specs; remove stale sections.
  - Acceptance: a new user follows the quickstart on a clean machine to a working reconstruct call.
  - Verify: follow it in a fresh clone.
  - Deps: all · Files: `README.md`

### ✅ Checkpoint: Complete
- [ ] All SPEC.md system success criteria 1–6 verified (evidence linked in PR)
- [ ] All workflows green; every module spec's success criteria checked off
- [ ] Human review → merge `refactor/prod-infra` into `main`

---

## Training track — Google Colab (SPEC-foundation req 10, SPEC-model-packaging req 2)

Runs alongside the infra phases; K1 after F5/F7, K3 after P2. I can't execute Colab, so Colab runs are verified by
you (manual check) and everything testable is covered locally on synthetic data.

- [ ] **K1: Colab-ready training entrypoint and `run_summary.json`** — M
  - Do: `accelerator: auto`; data/checkpoint/output paths overridable (Drive); resume from `last.ckpt` in the
    checkpoint dir (Colab disconnects); write `run_summary.json` next to the best checkpoint (git sha,
    `preprocess_version`, hparams, val metrics, checkpoint sha256, torch/CUDA versions, GPU name).
  - Acceptance: 2-step CPU run on synthetic `.h5` writes a valid summary; rerun with same dir resumes, not restarts.
  - Verify: `uv run pytest tests/unit/test_train_summary.py`.
  - Deps: F5, F7, Q2 · Files: `src/fastmri_knee/train.py`, `configs/trainer.yaml`, `configs/datamodule.yaml`, `tests/unit/test_train_summary.py`

- [ ] **K2: Colab training notebook (run from VS Code)** — S
  - Context: you open the repo notebook in VS Code and connect it to a Colab GPU kernel with the Google Colab
    extension. The kernel is remote: it can't see local files or uncommitted changes, so it installs the package
    from a pushed commit, and outputs come back via Drive.
  - Do: first verify `google.colab.drive.mount` works under the VS Code kernel (if not, record the fallback, e.g.
    `google.colab.auth` + Drive API or rclone, in `docs/training-colab.md`); then `notebooks/train_colab.ipynb`:
    parameter cell (git sha, Drive paths, epochs, batch size), GPU check, sha-is-pushed check, mount Drive, `pip install "fastmri-knee[train] @ git+https://github.com/mzdwedar/fastmri@<sha>"`, copy data from Drive
    to `/content` (Drive FUSE is slow for h5 random reads), run `python -m fastmri_knee.train …`, outputs to Drive;
    `docs/training-colab.md` with the steps. Outputs stripped before commit.
  - Acceptance: smoke run (1 epoch, small `num_samples`) on a Colab GPU writes `best.ckpt` + `run_summary.json` to Drive.
  - Verify: manual — you run it on Colab and share the summary.
  - Deps: K1, Q17, Q18, S4 (commit must be pushed), S11 · Files: `notebooks/train_colab.ipynb`, `docs/training-colab.md`

- [ ] **K3: Register a Colab run locally** — S
  - Do: `register --checkpoint … --summary …` (`make register CKPT=… SUMMARY=…`); refuse on checkpoint hash or
    `preprocess_version` mismatch; card `source=colab` with the summary's metrics so the PSNR promote gate works.
  - Acceptance: unit tests for both refusals and for card fields from a summary.
  - Verify: `uv run pytest tests/unit/packaging/test_register.py`.
  - Deps: P2, K1 · Files: `src/fastmri_knee/packaging/registry.py`, `Makefile`, `tests/unit/packaging/test_register.py`

- [ ] **K4: Notebook hygiene in CI** — XS
  - Do: `nbstripout` in pre-commit; ruff lints the notebook; CI fails if outputs are committed.
  - Acceptance/Verify: committing a notebook with outputs fails pre-commit and CI.
  - Deps: K2, C1 · Files: `.pre-commit-config.yaml`

---

## Gated (outside infra scope)

- [ ] **R1: Retrain on preprocessing v2 (Colab)** — you, via K2 · Deps: F8, Q1, K2, K3, S11 · download `best.ckpt` + `run_summary.json` from Drive, then `make register CKPT=… SUMMARY=… && make promote VERSION=n ALIAS=production`

---

## Decisions and setup

Tackle each item when its **trigger** task is reached. Record the outcome inline (`→ Decided YYYY-MM-DD: …`) and
in the referenced spec, then check it off. Items marked *deferred* are out of scope for this pass; they are kept
so the decision isn't lost.

### Decisions (Q)

- [ ] **Q1: Fix the FFT shift and retrain?**
  - Context: confirmed on a synthetic phantom; `|ifft2(k)|` is the correct image shifted by half a frame, so the
    crop misses the anatomy. Every existing model was trained on misaligned inputs.
  - Options: (a) fix → `PREPROCESS_VERSION="2"`, retrain (R1); (b) keep v1.
  - Recommendation: (a). Evidence: F2 figure + NCC numbers.
  - Trigger: after F2 · Blocks: F8, R1 · Spec: SPEC-foundation OQ1

- [x] **Q2: Python version** → decided 2026-11-12: 3.12 (requires-python >=3.10,<3.14)
  - Context: 3.10 reached end of life in Oct 2026. Risk: `segmentation-models-pytorch==0.3.3` / `timm==0.9.2` on 3.12.
    Hard constraint: `requires-python` must include Colab's runtime Python, or the package won't install there.
  - Options: 3.12 · 3.11 · stay on 3.10.
  - Recommendation: 3.12 if F1 passes, else 3.11. Evidence: F1 results.
  - Trigger: after F1 · Blocks: F6 · Spec: SPEC-foundation OQ2

- [ ] **Q3: Delete `latest_version.txt`?**
  - Context: untracked, contents `10.30.0,2025-09-01T16:14:47Z`, no references in code.
  - Recommendation: delete unless you know what writes it.
  - Trigger: F3 · Blocks: F3 (only that file)

- [ ] **Q4: Delete the extra venvs `.venv-cpu`, `.venv-dev`, `.venv-gpu`?**
  - Context: replaced by one `.venv` with extras; `.venv-dev` already points to a missing interpreter.
  - Recommendation: delete.
  - Trigger: F6 · Blocks: F6 (cleanup step only)

- [ ] **Q5: Per-slice max normalization and the response format**
  - Context: preprocessing divides each slice by its max, so outputs are on a relative [0, 1] scale.
  - Options: (a) keep, return `scale` attribute so clients can restore intensity; (b) change normalization (retrain).
  - Recommendation: (a).
  - Trigger: F7 / G4 · Blocks: G4 (response schema) · Spec: SPEC-foundation OQ3

- [ ] **Q6: MLflow version and backend**
  - Options: latest 2.x + SQLite (spec default) · 3.x · Postgres backend · MinIO/S3 artifacts.
  - Recommendation: latest 2.x + SQLite + local volume; revisit if multi-user.
  - Trigger: P1 · Blocks: P1 · Spec: SPEC-model-packaging req 1

- [ ] **Q7: Triton model control mode**
  - Options: `explicit` (sync calls the load API; failures visible) · `poll` (Triton watches the repo).
  - Recommendation: `explicit`.
  - Trigger: T1 · Blocks: T1, P5 · Spec: SPEC-model-packaging OQ2

- [ ] **Q8: Triton image version**
  - Context: pinned to `24.08-py3`; also must confirm an arm64 build exists (T1 spike).
  - Options: stay on 24.08 · move to a current 2026 release (revalidate ORT parity).
  - Recommendation: current release if it has arm64 and passes T3; decide with T1 evidence.
  - Trigger: T1 · Blocks: T1 · Spec: SPEC-inference-server req 6

- [ ] **Q9: Drop `/evaluate` (and `/predict`) from the serving API?**
  - Context: needs ground truth and torchmetrics; the gateway is torch-free. Becomes offline `make evaluate`.
  - Recommendation: drop.
  - Trigger: G6 · Blocks: G6 · Spec: SPEC.md OQ3

- [ ] **Q10: Keep the `acceleration` demo parameter on `/v1/reconstructions`?**
  - Context: lets fully sampled files simulate undersampled scanner input.
  - Recommendation: keep, documented as a demo feature.
  - Trigger: G4 · Blocks: G4 · Spec: SPEC-gateway OQ2

- [ ] **Q11: Alertmanager receivers**
  - Options: local `alert-sink` only (used by e2e) · plus Slack (needs S5) · plus email (needs SMTP).
  - Recommendation: `alert-sink` default, Slack enabled only when `SLACK_WEBHOOK_URL` is set.
  - Trigger: O3 · Blocks: O3 · Spec: SPEC-observability OQ1

- [ ] **Q12: Calibrate SLO targets**
  - Context: initial targets (99.5% availability, `/v1/infer` < 250 ms @ 99%, `/v1/reconstructions` < 5 s @ 95%)
    are guesses until the gateway baseline exists. Changing SLOs requires approval.
  - Recommendation: set from G7/C4 measured p95/p99 with ~2× headroom.
  - Trigger: O5 · Blocks: O5 · Spec: SPEC-observability §SLOs

- [ ] **Q13: Data-consistency error as the quality proxy with fully sampled input**
  - Context: with no mask, it measures fidelity to the input rather than undersampled consistency.
  - Recommendation: accept, label clearly on the dashboard.
  - Trigger: M1 · Blocks: M1 · Spec: SPEC-model-monitoring OQ1

- [ ] **Q14: GHCR image visibility**
  - Options: public · private (pulls need a token).
  - Trigger: C6 · Blocks: C6 · Spec: SPEC-ci-cd OQ1

- [ ] **Q15: Serve `staging` side by side (shadow/canary)?** — *deferred*
  - Recommendation: not this pass; revisit after Phase 7. Spec: SPEC-model-packaging OQ1

- [ ] **Q16: Async job API for large volumes?** — *deferred*
  - Recommendation: only if `/v1/reconstructions` p95 > 10 s. Spec: SPEC-gateway OQ1

- [ ] **Q17: Experiment tracking from Colab**
  - Context: Colab can't reach the local Compose MLflow.
  - Options: (a) MLflow file store on Drive for curves + register locally from `best.ckpt` + `run_summary.json`;
    (b) tunnel to local MLflow (ngrok/cloudflared; exposes a local service, needs auth); (c) hosted MLflow
    (DagsHub, Databricks).
  - Recommendation: (a): no exposed services, and registration stays a local, gated step.
  - Trigger: K2 · Blocks: K2

- [ ] **Q18: Dataset on Drive**
  - Context: local `data/` is empty. Full single-coil knee train set is ~88 GB; Colab local disk is ~100 GB and
    sessions are time-limited.
  - Options: full set · fixed subset (e.g. N train volumes + full val) recorded in a manifest with file hashes.
  - Recommendation: a committed subset manifest (`configs/data/subset.yaml`), copied to `/content` at session start.
  - Trigger: K2 · Blocks: K2, S11

### Setup (S) — steps that need you, an account, or a machine

- [x] **S1: Rebuild the local environment** (done 2026-11-12 with `uv sync --extra dev --extra cpu`; 15 tests pass on 3.10)
  - Context: `.venv` was broken and is now empty.
  - Do: `uv sync --extra dev` (before F6) / `make setup` (after F6).
  - Trigger: F1 (first task that runs code)

- [ ] **S2: Docker resources for the full stack**
  - Context: Docker Desktop has 8 GB RAM / 11 CPUs. Triton (4 GB limit) + gateway + MLflow + 8 observability
    services will be tight.
  - Do: raise Docker Desktop memory to ≥ 12 GB, or accept an observability Compose profile that is off by default.
  - Trigger: T1 (tight), O3 (required)

- [x] **S3: A trained checkpoint for compatibility checks**
  - → Decided 2026-11-12: none needed up front. F1 saves a random-init checkpoint under 3.10 and loads it under the
    new Python; real checkpoints come from Colab (K2).

- [ ] **S4: GitHub Actions enabled on `mzdwedar/fastmri`, branch pushed**
  - Do: confirm Actions enabled; `git push -u origin refactor/prod-infra` (I'll ask before pushing).
  - Trigger: C1 · Blocks: C1, C3, C4, C5, C7

- [ ] **S5: Slack webhook (optional)**
  - Do: create an incoming webhook; store as `SLACK_WEBHOOK_URL` in local `.env` and as a repo secret if CI should use it.
  - Trigger: O3 (only if Q11 includes Slack)

- [ ] **S6: Secrets for local and prod runs**
  - Do: `make api-key` (added in G5) generates a key and appends its hash to `secrets/api_keys`; set
    `GF_ADMIN_PASSWORD` in `.env`. `secrets/` and `.env` are gitignored.
  - Trigger: G5 / G6

- [ ] **S7: Dependency bot choice**
  - Options: Dependabot (no install) · Renovate (install the GitHub app).
  - Recommendation: Dependabot.
  - Trigger: C2

- [ ] **S8: GPU Docker host for the Triton GPU check** — *deferred*
  - Context: training moved to Colab (S11). Colab can't run Docker, so the T5 manual GPU check of Triton needs a
    separate NVIDIA host with Docker + NVIDIA Container Toolkit (e.g. a short-lived cloud GPU VM).
  - Trigger: T5 manual check only; T5's config validation doesn't need it.

- [ ] **S9: GHCR publishing permissions**
  - Do: allow workflows to write packages (repo Settings → Actions → Workflow permissions), set package visibility per Q14.
  - Trigger: C6

- [ ] **S10: Branch protection on `main`**
  - Do: require PRs, linear history, and the required checks listed in SPEC-ci-cd (GitHub Settings → Branches).
  - Trigger: C8

- [ ] **S11: Colab (via VS Code) and Google Drive**
  - Do: install the official Google Colab extension in VS Code and sign in (Command Palette → "Google Colab: Sign
    In"); confirm you can pick a GPU runtime (T4) from the notebook kernel picker; pick a Colab tier (free T4 has session limits; Pro gives longer sessions/A100); make sure Drive has space for
    the dataset subset (Q18) and checkpoints; upload data to `MyDrive/fastmri/{train,val,test}`; download one real
    val `.h5` into local `data/val/` for F2 and realistic local tests.
  - Trigger: F2 (one file), K2 (full setup)
