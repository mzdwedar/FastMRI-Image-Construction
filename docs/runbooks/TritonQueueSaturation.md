# TritonQueueSaturation

Severity: ticket. Average time a request waits in Triton's queue is more than half its compute time over 10 minutes (`nv_inference_queue_duration_us` / `nv_inference_compute_infer_duration_us`).

## Symptoms

- Latency grows with load while compute time per request stays flat.
- Pending requests on the Triton dashboard stay above zero.

## Dashboards

[Triton](http://localhost:3000/d/fastmri-triton) (queue vs compute time, pending requests), [Service overview](http://localhost:3000/d/fastmri-service) (in-flight).

## Likely causes

- More concurrent requests than model instances can compute (CPU-only Triton with 4 CPUs).
- Batch delay settings holding requests back to form larger batches.
- Heavy `/v1/reconstructions` volumes crowding out interactive calls.

## Remediation

```
docker compose logs --tail 100 triton
docker stats --no-stream triton
```
- Raise `TRITON_CPUS` or raise the instance count (`instances`) in the Triton config template under `deploy/triton/templates/`, then `make sync` and restart Triton.
- Lower `GATEWAY_MAX_INFLIGHT` so the gateway sheds load (503 with `Retry-After`) instead of queueing it behind Triton.
- Check `queue_delay_us` (rendered as `max_queue_delay_microseconds`) in the model config if the queue time is close to it.
