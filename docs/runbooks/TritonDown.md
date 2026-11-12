# TritonDown

Severity: page. Prometheus cannot scrape Triton (`up{job="triton"} == 0`) for 1 minute.

## Symptoms

- `/readyz` on the gateway returns 503 within about 10 s and `ModelNotReady` follows.
- Inference requests fail with a Triton error (`gateway_triton_errors_total` rises).

## Dashboards

[Triton](http://localhost:3000/d/fastmri-triton), [Infrastructure](http://localhost:3000/d/fastmri-infra).

## Likely causes

- Triton container exited (OOM against its 4 GB limit is the usual one).
- The model repository is empty or the model failed to load; Triton is live but not ready.
- ONNX Runtime crash on a bad input.

## Remediation

```
docker compose ps triton
docker compose logs --tail 100 triton
docker compose up -d triton
docker compose exec triton curl -fsS localhost:8000/v2/health/ready
```
After it is back, the gateway needs a few seconds to see the model ready; `/readyz` is checked per worker, so send a handful of `/v1/infer` calls before declaring recovery.
