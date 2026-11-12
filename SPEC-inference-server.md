# Spec: inference-server

Module of [SPEC.md](SPEC.md). Depends on: `model-packaging`.

## Objective

Run NVIDIA Triton as the only inference engine: private to the Compose network, ready only when the production
model is loaded, sized to its CPU budget, observable, and with explicit performance budgets that CI can check.
GPU is a drop-in profile.

## Current state (baseline)

Triton `24.08-py3` in Compose, `--model-repository=/models` (default `none` control mode, all models loaded at
start), `unet_model` with dynamic batching (preferred 4/8/16, 2 ms queue delay, max batch 16), `unet_model_nobatch`
baseline, ORT intra-op threads 4, ports 8100/8101/8102 published, healthcheck on `/v2/health/ready`.

## Requirements

1. **Server flags.**
   `--model-control-mode=explicit --load-model=unet_model --strict-readiness=true --exit-on-error=false
   --metrics-config=summary_latencies=true --allow-metrics=true --log-format=default --log-verbose=0`
   plus OpenTelemetry tracing flags (`--trace-config mode=opentelemetry`, rate from env; details in
   SPEC-observability). Startup with an empty repository must succeed (the server is healthy, the model is not
   ready) so `make sync` can load into it.
2. **Network exposure.** Default profile publishes only the metrics port to the Compose network; HTTP/gRPC are
   reachable from the gateway and packaging only. A `dev` Compose profile publishes 8100/8101 for debugging and
   benchmarks.
3. **Resources.** `cpus` and `mem_limit` from `.env` (`TRITON_CPUS=4`, `TRITON_MEM=4g`); ORT thread params are
   rendered from the same value (no drift between the container limit and ORT pool). `shm_size: 1g`. Restart
   policy `unless-stopped`.
4. **Model config** (rendered by packaging; values here are the contract):
   - `max_batch_size: 16`; input `input` FP32 `[1,320,320]`, output `output` FP32 `[1,320,320]`;
   - dynamic batching `preferred_batch_size [4, 8, 16]`, `max_queue_delay_microseconds` from template var
     (default 2000), `default_queue_policy { timeout_action: REJECT, default_timeout_microseconds: 5000000 }` and
     `max_queue_size: 256` so overload is rejected fast instead of queueing unbounded;
   - `model_warmup` with one batch of 1 and 16 zero inputs, so the first real request doesn't pay ORT init;
   - `instance_group` count template var (default 1 on CPU).
5. **GPU profile.** `docker-compose.gpu.yml` adds `deploy.resources.reservations.devices` (nvidia, count 1) and sets
   `TRITON_PROFILE=gpu`; template uses `KIND_GPU`, ORT CUDA EP with optional TensorRT accelerator
   (`precision_mode` FP16 behind a flag). Not run in CI; documented manual check in README.
6. **Image.** Pin the Triton image to a single version in `.env` (`TRITON_IMAGE`). Upgrading from 24.08: ask first;
   proposal is to move to a 2026 release during this module, revalidating ORT parity.
7. **Performance budgets** (CPU profile, 4 vCPU, random-init weights, single-slice requests via
   `benchmarks/load_test.py`, recorded in `benchmarks/baselines/cpu.json`):
   - establish baseline first (Task 1), then budgets are: throughput at concurrency 16 ≥ 0.9× baseline;
     p99 latency at concurrency 1 ≤ 1.15× baseline; error rate 0;
   - dynamic batching must beat `nobatch` throughput at concurrency 16 by ≥ 1.5× (validates the batching config);
   - overload (concurrency 512) returns gRPC `UNAVAILABLE`/HTTP 503 within the queue timeout; no OOM, no crash.

## Commands

```
make up                                  # triton starts empty-but-healthy, then `make sync` loads production
make bench BACKEND=triton MODEL=unet_model
make bench BACKEND=triton MODEL=unet_model_nobatch   # TRITON_PROFILE=nobatch loads the baseline model
make bench-compare                       # diff latest results vs benchmarks/baselines/cpu.json
```

## Testing strategy

- Integration (`integration`): container starts with an empty repo → `/v2/health/live` 200, `/v2/health/ready` 503
  (strict readiness, no model); after sync, ready 200; inference output shape/dtype correct and equal to
  onnxruntime on the same input (`atol=1e-5`); queue overflow → requests rejected, server stays live.
- Config: rendered `config.pbtxt` files parse (Triton loads them in the integration test); golden-file tests in
  packaging.
- Performance: `make bench` nightly in CI (advisory, shared runners are noisy) and required locally before
  merging changes to batching/thread settings.

## Boundaries

- Always: change batching or thread parameters only with a before/after benchmark attached to the PR.
- Ask first: Triton image upgrade; enabling response cache; adding the Python backend or ensembles.
- Never: publish Triton HTTP/gRPC ports in the default profile; mount the model repository read-write into Triton.

## Success criteria

- [ ] Fresh `make up` with an empty registry: Triton healthy, gateway `/readyz` 503 with reason `model_not_ready`;
      after `make model-random promote VERSION=1`: ready.
- [ ] Integration tests pass.
- [ ] Baseline recorded; batching ≥ 1.5× nobatch at concurrency 16; overload test shows fast rejection, no crash.
- [ ] `docker compose -f docker-compose.yml -f docker-compose.gpu.yml config` validates; GPU manual check documented.

## Open questions

1. Instance count vs threads on CPU: 1 instance × 4 threads (current) vs 2 × 2. Proposal: benchmark both in Task 1,
   keep the winner as the default.
