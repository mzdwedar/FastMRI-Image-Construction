# Implementation Plan: fastmri-knee production ML system

Specs: [SPEC.md](../SPEC.md) and `SPEC-<module>.md`. Task details (acceptance, verification, files): [todo.md](todo.md).
Branch: `refactor/prod-infra` (baseline `d9abbd6`). Each task = one or more atomic commits; PR per phase.

## Overview

Seven modules, 60 tasks (+1 gated retrain on Colab), 8 phases. Infra work runs entirely on random-init models and synthetic `.h5` files, so it
does not wait on the preprocessing fix or retraining. CI is introduced right after the foundation and gains a
gate in every later phase.

## Architecture decisions made while planning

- **Triton explicit-mode setup (T1) is pulled ahead of packaging sync (P5).** Sync must call Triton's load API, so
  the server flags land before it; the rest of `inference-server` follows packaging as specced.
- **Gateway is built in vertical slices:** `/v1/infer` through a real Triton first (thinnest end-to-end path), then
  readiness, then the h5 upload route, then auth/backpressure, then the image. The old FastAPI service and its
  tests stay working until G6 replaces them, so the stack is never broken mid-phase.
- **Preprocessing ships as `PREPROCESS_VERSION="1"` (current behaviour) first**, behind a parity test with the
  torch transform. The FFT-shift fix (F8) is a separate, gated task that bumps to `"2"`. Packaging gates promote on
  matching versions, so a model trained on v1 can never be served with v2 preprocessing.
- **High-risk spikes first in each phase:** Python 3.12 compatibility (F1), FFT on real data (F2), Triton on arm64 +
  empty explicit repo (T1), Triton OpenTelemetry propagation (O9 is a spike before O8 depends on it).
- **Integration tests run Triton via `docker run` from a pytest session fixture** (no testcontainers dependency);
  e2e tests run against `make up`.

## Phases and checkpoints

| Phase | Tasks | Checkpoint (must pass before next phase) |
|---|---|---|
| 1. Foundation | F1–F10 | `rm -rf .venv && make setup lint typecheck test` green; all test dirs collected; FFT decision recorded |
| 2. CI bootstrap | C1–C2 | `ci.yml` green on the branch; a broken lint/test PR fails |
| 3. Model packaging (+ Triton base) | P1–P4, T1, P5–P7 | `make model-random promote` serves v1; promote/rollback integration tests pass |
| 4. Inference server | T2–T5, C3 | Triton integration tests in CI; baseline recorded; batching ≥ 1.5× nobatch |
| 5. Gateway | G1–G7, C4, P8 | System criteria 1, 2, 6 pass; e2e job in CI; old FastAPI backend removed |
| 6. Observability | O1–O11, C5 | System criteria 3, 4 pass; `make check-configs` in CI |
| 7. Model monitoring | M1–M6 | Drift demo fires alert in e2e; stats overhead ≤ 2% |
| Training track (parallel) | K1–K4 | Colab smoke run writes `best.ckpt` + `run_summary.json`; `make register CKPT=… SUMMARY=…` passes gates |
| 8. Release and hardening | C6–C8, D1 | All workflows green; tagged image runs via `GATEWAY_IMAGE=...`; README rewritten; human review → merge to `main` |

Gated, outside infra scope: **R1** retrain on preprocessing v2 on Colab, after F8 and K2–K3.

## Dependency graph (critical path in bold)

```
**F1** → F6 ─┬→ C1 → C2
**F4** → F5 ─┤
F4 → F7 → F8(gated) → R1(gated)
F3, F9, F10 (independent after F4)
F6 → **P1 → P2 → P3**;  P4 → **T1 → P5 → P6 → P7**
P4 → T2 → T3 → C3;  P5 → T4 → C7;  T5 independent
T1,P6 → **G1 → G2 → G3 → G4 → G5 → G6** → G7, **C4** → P8
G6 → O1 → O2 → O8;  O9 → O8;  O3 → O4 → O5 → O6;  O7 → O8;  O1,O3 → O10;  all → **O11**;  O3 → C5
O1 → M1 → M2;  P2 → M3;  M2,M3 → M4 → M5 → M6
G6 → C6;  C1..C6 → C8;  all → D1
F5,F7 → K1 → K2 (Colab, manual);  P2,K1 → K3;  K2,C1 → K4;  F8,K2,K3 → R1(gated)
```

## Parallelization

- After F4: F3, F7, F9, F10 are independent.
- Phase 3/4: T5 (GPU override) and C2 in parallel with anything.
- Phase 6: tracks {O1→O2}, {O3→O4→O5→O6}, {O7}, {O9 spike} are parallel; join at O8/O10/O11.
- Phase 7: M3 (packaging side) parallel with M1→M2 (gateway side).
- Training track K1–K4 runs alongside Phases 3–6.
- Shared contracts that must land before parallel work: metric names (SPEC-observability table), API schema
  (SPEC-gateway table), model card fields (SPEC-model-packaging).

## Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Triton image slow or unavailable on arm64 (Apple Silicon); emulation makes local integration/bench unusable | High | T1 spike first: verify a native arm64 manifest for the pinned tag; fall back to running Triton integration only in CI (amd64) and benchmarks on CI/Linux; document |
| `segmentation-models-pytorch==0.3.3` incompatible with Python 3.12 | Med | F1 spike; options: smp ≥ 0.3.4 with checkpoint-load test, or stay on 3.11 |
| FFT fix invalidates all existing checkpoints | Med (ML), Low (infra) | Infra uses random-init; versioned preprocessing + promote gate prevents mismatched serving |
| Triton OpenTelemetry context propagation over gRPC doesn't link to gateway traces | Med | O9 spike before O8; fallback: gateway-only spans with Triton timing from `nv_*` metrics (spec amendment) |
| Prometheus multiprocess mode with gunicorn: gauge semantics, stale files | Med | `PROMETHEUS_MULTIPROC_DIR` on tmpfs, cleaned at start; `multiprocess_mode` set per gauge; unit test with 2 workers in G7 |
| GitHub-hosted runners too slow for full Compose e2e within 15 min | Med | Pre-pull images with cache, CPU-light random-init model, shortened alert `for:` via test override; measure in C4 |
| Benchmark noise on shared runners | Low | `bench.yml` advisory with 30% threshold; local `make bench` required for batching changes |
| Colab sessions disconnect; Drive FUSE is slow for h5 reads | Med | Resume from `last.ckpt` on Drive (K1); copy data to `/content` at start (K2) |
| Colab Python/torch/CUDA drift from the local lockfile | Med | Install at a pinned git sha; `requires-python` covers Colab's version (Q2); torch/CUDA recorded in `run_summary.json` |
| Scope creep (16 services in Compose) | Med | Each new service needs spec approval (Ask first); observability services behind a Compose profile if dev startup > 2 min |

## Open questions (block the tasks noted)

All open questions and human setup steps are tracked as tasks in
[todo.md § Decisions and setup](todo.md#decisions-and-setup): decisions `Q1–Q18`, setup `S1–S11`. Each lists its
trigger (the task that needs it) and is tackled when that task is reached, not up front. A task that lists a `Q`/`S`
dependency must not start until that item is checked off.
