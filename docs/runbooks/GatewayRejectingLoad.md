# GatewayRejectingLoad

Severity: ticket. The gateway has been refusing requests with 503 `Server busy` (`gateway_rejected_total{reason="overloaded"}`) for 10 minutes.

## Symptoms

- Clients get 503 with `Retry-After: 1`.
- In-flight requests sit at `GATEWAY_MAX_INFLIGHT` (default 32).

## Dashboards

[Service overview](http://localhost:3000/d/fastmri-service) (In-flight, Rejections).

## Likely causes

- Real traffic above capacity.
- Requests are slow (see Triton queue and CPU throttling), so they hold slots longer.
- A client retrying in a tight loop.

## Remediation

```
docker compose logs --tail 200 gateway | grep -c '"status": 503'
docker stats --no-stream gateway triton
```
- If latency is also bad, fix that first (`TritonQueueSaturation`).
- If traffic is legitimate, raise capacity (`GATEWAY_WORKERS`, `TRITON_CPUS`) before raising `GATEWAY_MAX_INFLIGHT`; a higher limit without capacity only moves the queue.
- Identify a noisy client by the `client` key-id hash in the access log (never the key itself).
