"""gunicorn settings. Workers come from `GATEWAY_WORKERS` (same variable the app's Settings reads)."""

import os
import shutil

bind = "0.0.0.0:8000"
worker_class = "uvicorn.workers.UvicornWorker"
workers = int(os.environ.get("GATEWAY_WORKERS", "2"))
graceful_timeout = 30  # SIGTERM drains in-flight requests for up to this long
timeout = 60
keepalive = 5
accesslog = None  # the app logs structured JSON itself
worker_tmp_dir = "/dev/shm"  # heartbeat files; keeps the read-only rootfs untouched


# Prometheus multiprocess mode: per-worker metric files live on tmpfs and are wiped at start, so a restarted
# container never reports a previous run's counters. A dead worker's live gauges are dropped; counters stay.
_metrics_dir = os.environ.get("PROMETHEUS_MULTIPROC_DIR")


def on_starting(server):
    if _metrics_dir:
        shutil.rmtree(_metrics_dir, ignore_errors=True)
        os.makedirs(_metrics_dir)


def child_exit(server, worker):
    if _metrics_dir:
        from prometheus_client import multiprocess

        multiprocess.mark_process_dead(worker.pid)
