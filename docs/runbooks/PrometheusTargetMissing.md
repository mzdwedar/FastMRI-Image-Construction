# PrometheusTargetMissing

Severity: ticket. A scrape job other than the gateway and Triton has been down for 5 minutes (node-exporter, cAdvisor, Alertmanager, Prometheus itself).

## Symptoms

- Gaps in the infrastructure panels; alerts that depend on those series stop working.

## Dashboards

[Infrastructure](http://localhost:3000/d/fastmri-infra), Prometheus Targets page (`localhost:9090/targets`).

## Likely causes

- The target container exited or restarted.
- A scrape URL or port changed in `prometheus.yml`.

## Remediation

```
curl -s localhost:9090/api/v1/targets | python3 -m json.tool | grep -E '"job"|"health"|lastError'
docker compose ps
docker compose up -d <service>
```
