# Spec: ci-cd

Module of [SPEC.md](SPEC.md). Depends on: `foundation`; gains a gate as each module lands.

## Objective

Every PR proves it doesn't break the system (code, configs, container stack), and every tagged release produces
scanned, signed-off images on GHCR that the Compose stack can run by tag. No dataset or GPU needed anywhere in CI.

## Workflows (`.github/workflows/`)

| Workflow | Trigger | Jobs (all required unless noted) |
|---|---|---|
| `ci.yml` | PR, push to `main` | `lint` (ruff check + format check), `typecheck` (mypy), `unit` (pytest unit, coverage gate), `configs` (`make check-configs`, `make dashboards` + `git diff --exit-code`, hadolint on Dockerfiles, `docker compose config`), `security` (gitleaks, pip-audit on `uv.lock` export) |
| `e2e.yml` | PR, push to `main` | `integration` (build gateway image, pytest `-m integration` with Triton container), `e2e` (`make up` with images built in-job, `make model-random promote VERSION=1`, pytest `-m e2e`; on failure upload `docker compose logs` and Prometheus snapshot as artifacts; always `make down -v`) |
| `bench.yml` | nightly, manual | load test (CPU profile) on random-init model, compare to `benchmarks/baselines/cpu.json`, publish results artifact and job summary; **advisory** (fails only on > 30% regression, since shared runners are noisy) |
| `release.yml` | tag `v*.*.*` | build gateway image (linux/amd64), trivy scan (fail on CRITICAL/HIGH with fix available), SBOM (syft) attached to release, push `ghcr.io/<owner>/fastmri-gateway:{version, sha}`, update `.env.example` image tag in release notes; git-cliff changelog |

Shared setup: `astral-sh/setup-uv` with cache keyed on `uv.lock`, Python from `.python-version`; Docker layer cache
via `type=gha`. Concurrency group per ref cancels superseded runs. All third-party actions pinned to commit SHA.
Workflow permissions default to `contents: read`; only `release.yml` gets `packages: write`.

## Requirements

1. Required status checks on `main`: `lint`, `typecheck`, `unit`, `configs`, `security`, `integration`, `e2e`.
   Branch protection: PR required, linear history (configured in GitHub; documented in `docs/contributing.md`).
2. PR CI wall time ≤ 15 min (unit jobs ≤ 5 min; e2e ≤ 12 min).
3. `make` targets are the single interface: workflows call `make <target>`, never duplicate commands, so local
   and CI runs are identical.
4. Images are never pushed from PRs. Compose references the gateway image by `${GATEWAY_IMAGE}` (local build by
   default, GHCR tag in prod `.env`).
5. Renovate or Dependabot for GitHub Actions, Docker images, and `uv.lock` (weekly, grouped).
6. Phasing: `ci.yml` lands with `foundation`; `integration` with `inference-server`; `e2e` with `gateway`;
   observability/model-monitoring add e2e cases and rule tests; `release.yml` after `gateway`.

## Commands

```
make ci            # lint typecheck test check-configs  (what ci.yml runs)
make ci-e2e        # build, up, model-random, promote, test-e2e, down  (what e2e.yml runs)
act -j unit        # optional local run of a workflow job
```

## Testing strategy

- The pipeline is tested by running it: a deliberately broken PR (failing test, bad alert rule, Dockerfile lint
  error, committed fake secret) must fail the matching job. Done once when each workflow lands; recorded in the PR.
- `actionlint` runs in the `configs` job.

## Boundaries

- Always: pin action SHAs and image tags; upload logs on e2e failure.
- Ask first: adding paid runners/GPU runners; changing required checks; publishing to registries other than GHCR.
- Never: store secrets in workflow files; run `pull_request_target` with checkout of PR code; disable a required
  check to merge.

## Success criteria

- [ ] All workflows green on `refactor/prod-infra` before merge to `main`.
- [ ] Broken-PR drills each fail the expected job.
- [ ] A `v0.2.0` tag publishes a scanned gateway image with SBOM; `GATEWAY_IMAGE=ghcr.io/...:0.2.0 make up` runs it.
- [ ] PR wall time ≤ 15 min.

## Open questions

1. Is the repo on GitHub with Actions enabled, and should images be public on GHCR? (Needed before `release.yml`.)
