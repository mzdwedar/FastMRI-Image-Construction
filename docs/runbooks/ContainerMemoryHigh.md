# ContainerMemoryHigh

Severity: ticket. A container's working set is above 90 % of its memory limit for 5 minutes (cAdvisor).

## Symptoms

- The container is close to being OOM-killed; restarts may follow.
- cAdvisor identifies containers by cgroup id here; match it with `docker ps --no-trunc`.

## Dashboards

[Infrastructure](http://localhost:3000/d/fastmri-infra) (container memory).

## Likely causes

- Triton holding large batches or a bigger model than the 4 GB limit allows.
- Gateway memory growth from large uploads held in memory (check `gateway_upload_bytes`).
- Loki or Tempo filling their 384 MB limit with a busy log or trace load.

## Remediation

```
docker ps --no-trunc --format '{{.ID}} {{.Names}}' | grep <first 12 chars of the id>
docker stats --no-stream
```
Raise the container's `mem_limit` in `docker-compose.yml` if the usage is legitimate, or find the leak from the logs. Within Docker Desktop's total memory (8 GB here) limits cannot sum past what the VM has.
