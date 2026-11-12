# GatewayDown

Severity: page. Prometheus cannot scrape the gateway (`up{job="gateway"} == 0`) for 1 minute.

## Symptoms

- Every `/v1/*` call fails or times out; `/healthz` does not answer.
- The service-overview panels stop updating (no series for `gateway_*`).

## Dashboards

[Service overview](http://localhost:3000/d/fastmri-service) (flat lines), [Infrastructure](http://localhost:3000/d/fastmri-infra) (is the container restarting?).

## Likely causes

- Container exited or is restart-looping (crash at start-up, bad config, missing `secrets/api_keys`).
- Out of memory kill (see `ContainerMemoryHigh`).
- Port 8000 busy, or the Compose network is broken so Prometheus cannot reach `gateway:8000`.

## Remediation

```
docker compose ps gateway
docker compose logs --tail 100 gateway
docker compose up -d gateway          # start it if it exited
curl -fsS localhost:8000/healthz
```
If the logs show a Python traceback at start-up, fix the cause (config or secret) before restarting. Once it answers, confirm `up{job="gateway"}` is 1 in Prometheus (`localhost:9090`).
