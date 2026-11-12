# ContainerCPUThrottled

Severity: ticket. More than 25 % of a container's CFS periods were throttled over 15 minutes.

## Symptoms

- The container is hitting its CPU cap, so latency rises without the host being busy.

## Dashboards

[Infrastructure](http://localhost:3000/d/fastmri-infra) (container CPU and throttling), [Triton](http://localhost:3000/d/fastmri-triton).

## Likely causes

- The `cpus` limit (`TRITON_CPUS`, `SERVER_CPUS`) is below what the workload needs.
- A burst of `/v1/reconstructions` volume requests.

## Remediation

```
docker stats --no-stream
```
Raise the CPU limit for the throttled container, or reduce load. Throttling often precedes `LatencySLOBurnSlow` and `TritonQueueSaturation`.
