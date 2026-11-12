"""Generate the provisioned Grafana dashboards. Output is committed; CI fails if it drifts.

    make dashboards        # or: python monitoring/grafana/build_dashboards.py

Three dashboards (the fourth, model monitoring, is added by SPEC-model-monitoring M5):
service-overview, triton, infrastructure. They link to each other, and the request-rate panel links to the
matching Loki logs.
"""

import json
from pathlib import Path
from urllib.parse import quote

OUT = Path(__file__).parent / "dashboards"
W = "[$__rate_interval]"
PROM = {"type": "prometheus", "uid": "prometheus"}
ALL_V1 = '{route=~"/v1/.*"}'
# SPEC-observability SLOs (latency only; thresholds are histogram bucket edges). The burn-rate rules (O5) replace
# these inline 30 d queries.
SLOS = {
    "/v1/infer": {"le": "0.5", "target": 0.99, "label": "< 500 ms"},
    "/v1/reconstructions": {"le": "5.0", "target": 0.95, "label": "< 5 s"},
}

NAV = [
    {"title": "Service overview", "uid": "fastmri-service"},
    {"title": "Triton", "uid": "fastmri-triton"},
    {"title": "Infrastructure", "uid": "fastmri-infra"},
]


def target(expr, legend="", ref="A", instant=False):
    t = {"refId": ref, "datasource": PROM, "expr": expr, "legendFormat": legend}
    if instant:
        t.update(instant=True, range=False)
    return t


def timeseries(title, targets, unit, x, y, w=12, h=8, desc="", stack=False, minv=0, links=None, thresholds=None):
    defaults = {
        "unit": unit,
        "min": minv,
        "custom": {
            "lineWidth": 2,
            "fillOpacity": 10 if stack else 0,
            "stacking": {"mode": "normal" if stack else "none"},
            "spanNulls": True,
        },
    }
    if thresholds:
        defaults["thresholds"] = {"mode": "absolute", "steps": thresholds}
        defaults["custom"]["thresholdsStyle"] = {"mode": "line"}
    if links:
        defaults["links"] = links
    return {
        "type": "timeseries",
        "title": title,
        "description": desc,
        "datasource": PROM,
        "gridPos": {"x": x, "y": y, "w": w, "h": h},
        "fieldConfig": {"defaults": defaults, "overrides": []},
        "options": {
            "legend": {"displayMode": "list", "placement": "bottom"},
            "tooltip": {"mode": "multi", "sort": "desc"},
        },
        "targets": [target(expr, legend, chr(65 + i)) for i, (expr, legend) in enumerate(targets)],
    }


def stat(title, expr, unit, x, y, w=4, h=4, desc="", steps=None, text_mode="auto", minv=None, maxv=None):
    defaults = {
        "unit": unit,
        "thresholds": {"mode": "absolute", "steps": steps or [{"color": "green", "value": None}]},
    }
    if minv is not None:
        defaults["min"] = minv
    if maxv is not None:
        defaults["max"] = maxv
    return {
        "type": "stat",
        "title": title,
        "description": desc,
        "datasource": PROM,
        "gridPos": {"x": x, "y": y, "w": w, "h": h},
        "fieldConfig": {"defaults": defaults, "overrides": []},
        "options": {
            "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
            "colorMode": "background",
            "graphMode": "none",
            "textMode": text_mode,
        },
        "targets": [target(expr, "", instant=True)],
    }


def dashboard(uid, title, panels):
    for i, p in enumerate(panels, start=1):
        p["id"] = i
    return {
        "uid": uid,
        "title": title,
        "tags": ["fastmri"],
        "timezone": "browser",
        "schemaVersion": 39,
        "version": 1,
        "refresh": "10s",
        "time": {"from": "now-1h", "to": "now"},
        "links": [
            {"title": n["title"], "type": "link", "url": f"/d/{n['uid']}", "icon": "dashboard", "keepTime": True}
            for n in NAV
            if n["uid"] != uid
        ],
        "panels": panels,
    }


def explore_link(title, logql, datasource_uid="loki"):
    """Data link into Explore. `logql` may use ${__field.labels.x}; the time range follows the dashboard."""
    state = {
        "datasource": datasource_uid,
        "queries": [{"refId": "A", "expr": logql}],
        "range": {"from": "${__from}", "to": "${__to}"},
    }
    return {"title": title, "url": "/explore?left=" + quote(json.dumps(state, separators=(",", ":")), safe="${}_.")}


def hq(p, metric, by="route", extra=""):
    return f"histogram_quantile({p}, sum by (le, {by}) (rate({metric}_bucket{extra}{W})))"


# ---------------------------------------------------------------- service overview


def sli(route, le):
    """Share of requests on `route` that finished under the bucket edge `le`, over 30 d."""
    return (
        f'sum(increase(gateway_request_duration_seconds_bucket{{route="{route}",le="{le}"}}[30d])) '
        f'/ sum(increase(gateway_request_duration_seconds_count{{route="{route}"}}[30d]))'
    )


def slo_stats(x0):
    """Per SLO: the 30 d SLI, and the error budget left (1 minus the share of the allowed slow requests used)."""
    out = []
    for i, (route, slo) in enumerate(SLOS.items()):
        expr = sli(route, slo["le"])
        out.append(
            stat(
                f"{route} {slo['label']} (30 d)",
                expr,
                "percentunit",
                x0 + 6 * i,
                0,
                w=3,
                desc=f"Share of {route} requests under {slo['label']}; SLO {slo['target']:.0%}. Covers the data "
                "that exists if Prometheus holds less than 30 d.",
                steps=[{"color": "red", "value": None}, {"color": "green", "value": slo["target"]}],
                minv=0.8,
                maxv=1,
            )
        )
        out.append(
            stat(
                f"{route} budget left",
                f"1 - ((1 - ({expr})) / {round(1 - slo['target'], 4)})",
                "percentunit",
                x0 + 6 * i + 3,
                0,
                w=3,
                desc=f"1 minus the share of the {1 - slo['target']:.0%} slow-request allowance already used. "
                "Negative = SLO breached.",
                steps=[
                    {"color": "red", "value": None},
                    {"color": "orange", "value": 0.1},
                    {"color": "green", "value": 0.5},
                ],
            )
        )
    return out


service = dashboard(
    "fastmri-service",
    "Service overview",
    [
        *slo_stats(0),
        stat(
            "Model ready",
            "min(gateway_model_ready)",
            "none",
            12,
            0,
            w=3,
            desc="1 when Triton is reachable and the model is ready (alert: ModelNotReady).",
            steps=[{"color": "red", "value": None}, {"color": "green", "value": 1}],
        ),
        stat(
            "Served model",
            "max by (version) (gateway_model_info)",
            "none",
            15,
            0,
            w=3,
            desc="Model version the gateway reports (label of gateway_model_info).",
            text_mode="name",
        ),
        stat(
            "In flight",
            "sum(gateway_inflight_requests)",
            "none",
            18,
            0,
            w=3,
            desc="Requests holding a backpressure slot.",
        ),
        stat(
            "Rejections (5 m)",
            "sum(increase(gateway_rejected_total[5m])) or vector(0)",
            "none",
            21,
            0,
            w=3,
            steps=[{"color": "green", "value": None}, {"color": "orange", "value": 1}],
        ),
        timeseries(
            "Request rate by route",
            [("sum by (route) (rate(gateway_requests_total" + W + "))", "{{route}}")],
            "reqps",
            0,
            4,
            desc="Click a series: Logs for this route opens the matching access-log lines in Loki.",
            links=[
                explore_link(
                    "Logs for this route",
                    '{service="gateway"} | json | route="${__field.labels.route}"',
                )
            ],
        ),
        timeseries(
            "Error ratio (5xx / all) by route",
            [
                (
                    f'sum by (route) (rate(gateway_requests_total{{status_class="5xx"}}{W})) '
                    f"/ sum by (route) (rate(gateway_requests_total{W}))",
                    "{{route}}",
                )
            ],
            "percentunit",
            12,
            4,
            desc="Errors are not an SLO (SLOs are latency only); see GatewayDown, TritonDown and ModelNotReady for outages.",
        ),
        timeseries(
            "Latency p50 / p95 / p99 by route",
            [
                (hq(0.5, "gateway_request_duration_seconds"), "{{route}} p50"),
                (hq(0.95, "gateway_request_duration_seconds"), "{{route}} p95"),
                (hq(0.99, "gateway_request_duration_seconds"), "{{route}} p99"),
            ],
            "s",
            0,
            12,
            desc="Includes upload read time. SLO: /v1/infer < 500 ms, /v1/reconstructions < 5 s.",
        ),
        timeseries(
            "Stage time p95",
            [(hq(0.95, "gateway_stage_duration_seconds", by="stage"), "{{stage}}")],
            "s",
            12,
            12,
            desc="Where a request spends its time: preprocess, Triton call, response encode. "
            "Open a trace in Tempo for per-request detail.",
            links=[
                {
                    "title": "Find slow traces in Tempo",
                    "url": "/explore?left="
                    + quote(
                        json.dumps(
                            {
                                "datasource": "tempo",
                                "queries": [
                                    {
                                        "refId": "A",
                                        "queryType": "traceql",
                                        "query": '{ resource.service.name = "gateway" && duration > 1s }',
                                    }
                                ],
                                "range": {"from": "${__from}", "to": "${__to}"},
                            },
                            separators=(",", ":"),
                        ),
                        safe="${}_.",
                    ),
                }
            ],
        ),
        timeseries(
            "In-flight requests",
            [("sum(gateway_inflight_requests)", "in flight")],
            "none",
            0,
            20,
            desc="Compare with GATEWAY_MAX_INFLIGHT: at the limit the gateway answers 503 overloaded.",
        ),
        timeseries(
            "Rejections by reason",
            [("sum by (reason) (rate(gateway_rejected_total" + W + "))", "{{reason}}")],
            "reqps",
            12,
            20,
            stack=True,
        ),
        timeseries(
            "Slices processed/s",
            [("sum by (route) (rate(gateway_slices_total" + W + "))", "{{route}}")],
            "none",
            0,
            28,
        ),
        timeseries(
            "Triton call failures/s by gRPC code",
            [("sum by (code) (rate(gateway_triton_errors_total" + W + "))", "{{code}}")],
            "reqps",
            12,
            28,
            stack=True,
        ),
    ],
)


# ---------------------------------------------------------------- Triton
def per_req(metric):
    """Mean of a Triton cumulative-microseconds counter per successful request, in seconds."""
    return f"sum by (model) (rate({metric}{W})) / sum by (model) (rate(nv_inference_request_success{W})) / 1e6"


triton = dashboard(
    "fastmri-triton",
    "Triton",
    [
        timeseries(
            "Inference rate (successful requests/s)",
            [(f"sum by (model) (rate(nv_inference_request_success{W}))", "{{model}}")],
            "reqps",
            0,
            0,
        ),
        timeseries(
            "Failed requests/s",
            [(f"sum by (model, reason) (rate(nv_inference_request_failure{W}))", "{{model}} {{reason}}")],
            "reqps",
            12,
            0,
        ),
        timeseries(
            "Queue vs compute (mean per request)",
            [
                (per_req("nv_inference_queue_duration_us"), "{{model}} queue"),
                (per_req("nv_inference_compute_infer_duration_us"), "{{model}} compute"),
            ],
            "s",
            0,
            8,
            desc="Queue time rising while compute stays flat means the instance is saturated "
            "(alert: TritonQueueSaturation). The nv_*_duration_us metrics are cumulative, so they are rates "
            "divided by the request rate.",
        ),
        timeseries(
            "Request latency p50 / p99 (Triton summary)",
            [
                ('max by (model) (nv_inference_request_summary_us{quantile="0.5"}) / 1e6', "{{model}} p50"),
                ('max by (model) (nv_inference_request_summary_us{quantile="0.99"}) / 1e6', "{{model}} p99"),
            ],
            "s",
            12,
            8,
            desc="Needs --metrics-config=summary_latencies=true.",
        ),
        timeseries(
            "Average batch size",
            [
                (
                    f"sum by (model) (rate(nv_inference_count{W})) / sum by (model) (rate(nv_inference_exec_count{W}))",
                    "{{model}}",
                )
            ],
            "none",
            0,
            16,
            desc="nv_inference_count / nv_inference_exec_count. 1.0 means dynamic batching is not merging requests.",
        ),
        timeseries(
            "Pending requests",
            [("sum by (model) (nv_inference_pending_request_count)", "{{model}}")],
            "none",
            12,
            16,
        ),
        timeseries(
            "Inferences and executions per version",
            [
                (f"sum by (model, version) (rate(nv_inference_count{W}))", "{{model}} v{{version}} inferences"),
                (f"sum by (model, version) (rate(nv_inference_exec_count{W}))", "{{model}} v{{version}} executions"),
            ],
            "ops",
            0,
            24,
        ),
        timeseries(
            "Triton CPU and memory",
            [
                ("avg(nv_cpu_utilization)", "cpu (fraction)"),
            ],
            "percentunit",
            12,
            24,
        ),
    ],
)

# ---------------------------------------------------------------- infrastructure
DOCKER = 'id=~"/docker/.+"'
LIMIT = f"(container_spec_memory_limit_bytes{{{DOCKER}}} > 0)"
infra = dashboard(
    "fastmri-infra",
    "Infrastructure",
    [
        timeseries(
            "Container CPU (cores)",
            [(f"sum by (name) (rate(container_cpu_usage_seconds_total{{{DOCKER}}}{W}))", "{{name}}")],
            "short",
            0,
            0,
            desc="cAdvisor per-container CPU. cAdvisor on Docker Desktop reports cgroup ids; names may be empty.",
        ),
        timeseries(
            "Container memory (working set)",
            [(f"container_memory_working_set_bytes{{{DOCKER}}}", "{{name}} {{id}}")],
            "bytes",
            12,
            0,
        ),
        timeseries(
            "Memory vs limit",
            [(f"container_memory_working_set_bytes{{{DOCKER}}} / on (id) {LIMIT}", "{{id}}")],
            "percentunit",
            0,
            8,
            thresholds=[{"color": "transparent", "value": None}, {"color": "red", "value": 0.9}],
            desc="Alert ContainerMemoryHigh fires above 90 %. Containers without a limit are omitted.",
        ),
        timeseries(
            "CPU throttling ratio",
            [
                (
                    f"rate(container_cpu_cfs_throttled_periods_total{{{DOCKER}}}{W}) "
                    f"/ rate(container_cpu_cfs_periods_total{{{DOCKER}}}{W})",
                    "{{id}}",
                )
            ],
            "percentunit",
            12,
            8,
            thresholds=[{"color": "transparent", "value": None}, {"color": "red", "value": 0.25}],
            desc="Alert ContainerCPUThrottled fires above 25 % for 15 min.",
        ),
        timeseries(
            "Host CPU",
            [('1 - avg(rate(node_cpu_seconds_total{mode="idle"}' + W + "))", "busy")],
            "percentunit",
            0,
            16,
        ),
        timeseries(
            "Host memory",
            [
                ("node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes", "used"),
                ("node_memory_MemTotal_bytes", "total"),
            ],
            "bytes",
            12,
            16,
        ),
        timeseries(
            "Docker data disk used",
            [
                (
                    '1 - (node_filesystem_avail_bytes{mountpoint="/var/lib/docker"} '
                    '/ node_filesystem_size_bytes{mountpoint="/var/lib/docker"})',
                    "/var/lib/docker",
                ),
            ],
            "percentunit",
            0,
            24,
        ),
        timeseries(
            "Scrape targets up",
            [("up", "{{job}}")],
            "none",
            12,
            24,
            desc="1 = Prometheus scraped the target (alerts: GatewayDown, TritonDown, PrometheusTargetMissing).",
        ),
    ],
)

FILES = {
    "service-overview.json": service,
    "triton.json": triton,
    "infrastructure.json": infra,
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for stale in OUT.glob("*.json"):
        if stale.name not in FILES:
            stale.unlink()  # dashboards the generator no longer owns (fastapi-dashboard.json, triton-dashboard.json)
    for name, d in FILES.items():
        (OUT / name).write_text(json.dumps(d, indent=2) + "\n")
        print("wrote", OUT / name)


if __name__ == "__main__":
    main()
