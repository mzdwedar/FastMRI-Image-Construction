# ModelNotReady

Severity: page. `gateway_model_ready == 0` for 2 minutes: the readiness loop does not see the model loaded in Triton.

## Symptoms

- Reconstruction requests are refused with 503 until the loop sees the model.
- `/readyz` is 503 while `/healthz` is 200.

## Dashboards

[Service overview](http://localhost:3000/d/fastmri-service) (Model ready, Served model), [Triton](http://localhost:3000/d/fastmri-triton).

## Likely causes

- Triton is down or still loading (check `TritonDown`).
- The model repository does not contain the expected version, or `config.pbtxt` does not match the ONNX file.
- gRPC connection problem between gateway and Triton.

## Remediation

```
docker compose logs --tail 100 triton | grep -i -E "model|error|fail"
docker compose exec gateway python -c "import urllib.request as u; print(u.urlopen('http://127.0.0.1:8000/readyz').status)"
ls -R deploy/model_repository
```
If the repository is empty or wrong, load the production model with `make sync` (needs `make up-dev`); if a bad promotion caused it, `make rollback`, then restart Triton. If Triton is fine, restart the gateway: `docker compose restart gateway`.
