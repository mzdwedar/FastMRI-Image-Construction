"""Generate the provisioned Grafana dashboards (run after editing a query).

    python monitoring/grafana/build_dashboards.py
"""

import json
from pathlib import Path

OUT = Path(__file__).parent / "dashboards"
W = "[30s]"  # 6 scrapes at 5 s: smooth enough, still reacts within a benchmark step


def panel(pid, title, targets, unit, x, y, w=12, h=8, desc="", stack=False, minv=0):
    return {
        "id": pid,
        "type": "timeseries",
        "title": title,
        "description": desc,
        "datasource": {"type": "prometheus", "uid": "prometheus"},
        "gridPos": {"x": x, "y": y, "w": w, "h": h},
        "fieldConfig": {
            "defaults": {
                "unit": unit,
                "min": minv,
                "custom": {"lineWidth": 2, "fillOpacity": 10 if stack else 0,
                           "stacking": {"mode": "normal" if stack else "none"}, "spanNulls": True},
            },
            "overrides": [],
        },
        "options": {"legend": {"displayMode": "list", "placement": "bottom"},
                    "tooltip": {"mode": "multi", "sort": "desc"}},
        "targets": [{"refId": chr(65 + i), "expr": expr, "legendFormat": legend}
                    for i, (expr, legend) in enumerate(targets)],
    }


def dashboard(uid, title, panels):
    return {
        "uid": uid, "title": title, "tags": ["fastmri", "serving"], "timezone": "browser",
        "schemaVersion": 39, "version": 1, "refresh": "5s",
        "time": {"from": "now-15m", "to": "now"}, "panels": panels,
    }


def per_req(metric):
    """Average of a Triton cumulative-microseconds counter per successful request, in seconds."""
    return (f"sum by (model) (rate({metric}{W})) / sum by (model) (rate(nv_inference_request_success{W})) / 1e6")


triton = dashboard("triton-serving", "Triton: U-Net serving", [
    panel(1, "Throughput (requests/s)", [
        (f"sum by (model) (rate(nv_inference_request_success{W}))", "{{model}}")], "reqps", 0, 0,
        desc="Successful inference requests per second, per model."),
    panel(2, "Mean request latency", [(per_req("nv_inference_request_duration_us"), "{{model}}")], "s", 12, 0,
        desc="rate(duration) / rate(count). The raw nv_*_duration_us metrics are cumulative counters, "
             "so plotting them directly only ever goes up."),
    panel(3, "Request latency p50 / p99 (server side)", [
        ('max by (model) (nv_inference_request_summary_us{quantile="0.5"}) / 1e6', "{{model}} p50"),
        ('max by (model) (nv_inference_request_summary_us{quantile="0.99"}) / 1e6', "{{model}} p99")], "s", 0, 8,
        desc="Triton summary quantiles (needs --metrics-config=summary_latencies=true)."),
    panel(4, "Where the time goes: queue vs compute (mean per request)", [
        (per_req("nv_inference_queue_duration_us"), "{{model}} queue"),
        (per_req("nv_inference_compute_infer_duration_us"), "{{model}} compute")], "s", 12, 8,
        desc="Queue time rising while compute stays flat means the instance is saturated: add instances "
             "or batch harder. Compute rising means bigger batches."),
    panel(5, "Average batch size", [
        (f"sum by (model) (rate(nv_inference_count{W})) / sum by (model) (rate(nv_inference_exec_count{W}))",
         "{{model}}")], "none", 0, 16,
        desc="Inferences per model execution. 1.0 means dynamic batching is not merging requests."),
    panel(6, "Failed requests/s", [
        (f"sum by (model, reason) (rate(nv_inference_request_failure{W}))", "{{model}} {{reason}}")], "reqps", 12, 16),
    panel(7, "Triton CPU utilisation", [("avg(nv_cpu_utilization)", "cpu")], "percentunit", 0, 24),
    panel(8, "Triton memory", [("avg(nv_cpu_memory_used_bytes)", "used")], "bytes", 12, 24),
])



def q(p):
    return f'histogram_quantile({p}, sum by (le) (rate(inference_latency_seconds_bucket{{endpoint="infer"}}{W})))'


fastapi = dashboard("fastapi-serving", "FastAPI: U-Net serving (PyTorch)", [
    panel(1, "Throughput (requests/s)", [
        (f"sum by (endpoint) (rate(inference_requests_total{W}))", "{{endpoint}}")], "reqps", 0, 0),
    panel(2, "/infer latency p50 / p95 / p99", [(q(0.5), "p50"), (q(0.95), "p95"), (q(0.99), "p99")], "s", 12, 0,
        desc="From the inference_latency_seconds histogram (model time only)."),
    panel(3, "Slices reconstructed/s", [
        (f"sum by (endpoint) (rate(inference_slices_total{W}))", "{{endpoint}}")], "none", 0, 8),
    panel(4, "Errors/s", [
        (f"sum by (endpoint) (rate(inference_failures_total{W}))", "{{endpoint}} model errors"),
        (f'sum(rate(http_requests_total{{status="5xx"}}{W}))', "HTTP 5xx"),
        (f'sum(rate(http_requests_total{{status="4xx"}}{W}))', "HTTP 4xx")], "reqps", 12, 8),
    panel(5, "Host CPU", [("max(host_cpu_percent)", "cpu %")], "percent", 0, 16),
    panel(6, "Service memory (RSS)", [("max(process_rss_bytes)", "rss")], "bytes", 12, 16),
])

OUT.mkdir(parents=True, exist_ok=True)
for name, d in [("triton-dashboard.json", triton), ("fastapi-dashboard.json", fastapi)]:
    (OUT / name).write_text(json.dumps(d, indent=2) + "\n")
    print("wrote", OUT / name)
