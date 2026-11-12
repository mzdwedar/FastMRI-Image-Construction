# Spec: gateway

Module of [SPEC.md](SPEC.md). Depends on: `inference-server`, `foundation` (numpy preprocessing).

## Objective

Replace the FastAPI + PyTorch backend with a torch-free gateway that is the only public entry point: it accepts
k-space volumes, validates and preprocesses them, calls Triton over gRPC, and returns reconstructions, with auth,
limits, backpressure, honest health probes, and a stable versioned API.

## API contract (v1)

All errors are `application/problem+json` (RFC 9457): `{type, title, status, detail, request_id}`. Every response
carries `X-Request-ID` (echoed if the client sent a valid one, else generated) and `X-Model-Version`.

| Method | Path | Auth | Request | Response |
|---|---|---|---|---|
| POST | `/v1/reconstructions` | API key | multipart `file`: single-coil `.h5` with `kspace` (S, H, W) complex64; optional query `acceleration` (int, simulates undersampling for demos; default none) | 200 `application/x-hdf5`: dataset `reconstruction` (S, 320, 320) float32, attrs `model_version`, `preprocess_version`, `request_id`, `scale` (per-slice max) |
| POST | `/v1/infer` | API key | `application/octet-stream` float32 bytes, N×1×320×320, N ≤ 64 | 200 octet-stream, same shape (benchmark/low-level path) |
| GET | `/v1/model` | API key | — | 200 JSON: served version, model card fields |
| GET | `/healthz` | none | — | 200 if the process is up (liveness; never checks dependencies) |
| GET | `/readyz` | none | — | 200 if Triton is reachable and `unet_model` is ready; 503 with `{reason}` otherwise |
| GET | `/metrics` | none (network-restricted) | — | Prometheus exposition |

Status codes: 400 malformed input (not HDF5, missing `kspace`, wrong dtype/rank); 401 missing/invalid key;
413 upload > `MAX_UPLOAD_MB` (default 256) or slices > `MAX_SLICES` (default 64); 415 wrong content type;
422 parameter validation; 503 not ready or overloaded (with `Retry-After`); 504 Triton deadline exceeded;
500 unexpected (no internals in body). `/evaluate` and `/predict` are removed (see SPEC.md open question 3).

## Requirements

1. **Torch-free.** Depends only on the `gateway` extra and `fastmri_knee.preprocess`; a test fails if `torch` is
   importable in the gateway image.
2. **Settings** via `pydantic-settings` (`GATEWAY_` prefix): Triton URL, timeouts, limits, API keys file, OTLP
   endpoint, log level, workers. Invalid settings fail at startup.
3. **Upload handling.** Stream the upload to a temp file with a running byte count (abort at the limit, no full
   read into memory first); open with h5py read-only; validate dataset names, rank, dtype, slice count before
   any compute; temp files always removed (also on client disconnect).
4. **Preprocessing** runs in a thread pool (numpy FFT releases the GIL), one call per volume.
5. **Triton client** (`tritonclient.grpc.aio`, one channel per worker, created in lifespan):
   - per-call deadline: `TRITON_TIMEOUT_S` (default 10 s for a chunk);
   - split volumes into chunks of ≤ `max_batch_size` (read from Triton model config at startup), sent
     concurrently;
   - retry only `UNAVAILABLE`, max 2 retries, jittered exponential backoff (50 ms base), within the overall
     request deadline; never retry `INVALID_ARGUMENT`;
   - output validated (shape, finite); non-finite output → 500 and an error metric, never returned to clients.
6. **Backpressure.** A global in-flight limit (`MAX_INFLIGHT`, default 32 volumes) via semaphore; when full,
   respond 503 + `Retry-After: 1` immediately rather than queueing in the gateway.
7. **Readiness loop.** A background task polls Triton model readiness every 2 s and caches the result; `/readyz`
   reads the cache (fast, no per-probe gRPC). It also refreshes the served model version for `X-Model-Version`.
8. **Auth.** API keys from a mounted file (`/run/secrets/api_keys`, one SHA-256 hash per line), compared in
   constant time. `GATEWAY_AUTH_DISABLED=true` allowed only when `GATEWAY_ENV=dev` (startup error otherwise).
9. **Process model.** `gunicorn -k uvicorn.workers.UvicornWorker`, workers from env (default 2), graceful timeout
   30 s; SIGTERM drains in-flight requests. Prometheus multiprocess mode for metrics across workers.
10. **Image.** `deploy/gateway/Dockerfile`: multi-stage uv build, `python:3.12-slim` runtime, non-root uid 10001,
    read-only root filesystem in Compose (tmpfs for `/tmp`), healthcheck on `/healthz`, labels with git sha/version.
11. **Instrumentation hooks** for observability (metrics, structured logs, spans) are defined in
    SPEC-observability; this module must expose the spans `upload.read`, `h5.validate`, `preprocess`,
    `triton.infer` (per chunk), `response.encode`.

## Commands

```
make up                         # gateway on :8000
make test                       # unit tests (fake Triton client)
make test-integration           # gateway ↔ real Triton
curl -H "X-API-Key: $KEY" -F file=@tests/fixtures/synthetic.h5 localhost:8000/v1/reconstructions -o out.h5
```

## Testing strategy

- Unit (`FakeTritonClient` implementing the same protocol): every status code in the table; upload limit enforced
  without buffering the whole body (memory assertion on a 300 MB stream); request-id propagation; chunking (1, 16,
  17, 64 slices); retry on UNAVAILABLE but not on INVALID_ARGUMENT; deadline → 504; semaphore full → 503 with
  Retry-After; non-finite output → 500; auth disabled outside dev → startup error; temp files removed after
  success, error, and disconnect.
- Contract: OpenAPI schema snapshot test (`tests/unit/openapi.json`); changes need a spec update.
- Integration: real Triton + random-init model; `/v1/reconstructions` output equals ONNX Runtime run locally on
  the same preprocessed input (`atol=1e-5`); `/readyz` flips to 503 when Triton is stopped and back after start.
- Image: `docker run --rm gateway python -c "import torch"` must fail; image size ≤ 500 MB.

## Boundaries

- Always: validate before compute; keep the API versioned under `/v1`.
- Ask first: new endpoints; changing response format; adding per-key rate limiting (needs Redis, a new service).
- Never: log request bodies or file contents; return exception messages from internal failures; load a model in
  the gateway.

## Success criteria

- [ ] All unit, contract, and integration tests pass; coverage ≥ 85% on `gateway/`.
- [ ] A 35-slice synthetic volume round-trips via `curl` on the CPU stack; p95 end-to-end latency recorded as the
      baseline for the SLO in SPEC-observability.
- [ ] Gateway image has no torch, ≤ 500 MB, runs as non-root with a read-only root filesystem.
- [ ] Killing Triton mid-load: in-flight requests get 503/504 (no 500s, no hangs past the deadline); gateway stays
      live and recovers without restart.

## Open questions

1. Async job API (`POST` returns 202 + job id) for large volumes? Proposal: not needed at ≤ 64 slices and seconds
   of latency; revisit if p95 > 10 s.
2. Keep the `acceleration` demo parameter? It lets a fully sampled dataset show realistic undersampled input.
   Proposal: keep, marked as a demo feature in the OpenAPI description.
