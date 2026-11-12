# LatencySLOBurnFast

Severity: page. A route is spending its 30 d latency error budget more than 14.4x too fast over both 1 h and 5 m. `/v1/infer` target: < 500 ms for 99 %; `/v1/reconstructions`: < 5 s for 95 %.

## Symptoms

- p95 or p99 for the named `route` is above its threshold right now.
- Failed requests do not count here: if requests are failing, `TritonDown`, `ModelNotReady` or `GatewayDown` should also be firing.

## Dashboards

[Service overview](http://localhost:3000/d/fastmri-service) (latency SLI, p50/p95/p99 by route, in-flight), [Triton](http://localhost:3000/d/fastmri-triton) (queue vs compute time).

## Likely causes

- Triton queueing: more concurrent requests than it can compute (`TritonQueueSaturation`).
- CPU throttling of Triton or the gateway (`ContainerCPUThrottled`), or a noisy neighbour on the host.
- A larger input than usual (more slices per volume) or a changed model that is slower.
- Cold start after a restart: the first requests are slow.

## Remediation

In Grafana, open the slow request's trace (panel data link into Tempo) and see which span is long: `preprocess`, `triton` (queue vs compute) or `encode`.
```
docker compose logs --tail 200 gateway | grep '"route": "/v1/infer"'
docker stats --no-stream
```
- Queue time dominant: lower the load or raise Triton CPUs/instances (`TRITON_CPUS`).
- Throttling: raise `SERVER_CPUS` / `TRITON_CPUS` or stop other work on the host.
- If a recent deploy caused it, roll back the model or image and confirm the burn rate falls below 14.4 on the 5 m window.
