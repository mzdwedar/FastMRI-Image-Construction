"""gunicorn settings. Workers come from `GATEWAY_WORKERS` (same variable the app's Settings reads)."""

import os

bind = "0.0.0.0:8000"
worker_class = "uvicorn.workers.UvicornWorker"
workers = int(os.environ.get("GATEWAY_WORKERS", "2"))
graceful_timeout = 30  # SIGTERM drains in-flight requests for up to this long
timeout = 60
keepalive = 5
accesslog = None  # the app logs structured JSON itself
worker_tmp_dir = "/dev/shm"  # heartbeat files; keeps the read-only rootfs untouched
