"""Prometheus metrics for the FastAPI service.

Triton exports its own metrics on :8002/metrics; these are the equivalents for the
Python service, so both backends can be compared on the same Grafana dashboards.
"""

import threading

import psutil
from prometheus_client import Counter, Gauge, Histogram

# 5 ms .. 30 s: covers a single-slice /infer call up to a full-volume /predict
LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.25, 0.5, 0.75, 1, 2.5, 5, 10, 30)
BATCH_BUCKETS = (1, 2, 4, 8, 16, 32, 64)


class InferenceMonitor:
    """Request, latency, batch-size and error metrics, labelled by endpoint."""

    def __init__(self) -> None:
        self.requests = Counter("inference_requests_total", "Inference requests received", ["endpoint"])
        self.failures = Counter("inference_failures_total", "Inference requests that failed", ["endpoint"])
        self.latency = Histogram(
            "inference_latency_seconds",
            "Wall time spent in model inference, excluding upload and response",
            ["endpoint"],
            buckets=LATENCY_BUCKETS,
        )
        self.batch_size = Histogram(
            "inference_batch_size", "Slices per forward pass", ["endpoint"], buckets=BATCH_BUCKETS
        )
        self.slices = Counter("inference_slices_total", "MRI slices reconstructed", ["endpoint"])


class HardwareMonitor:
    """Samples host CPU/memory and process RSS in a background thread."""

    def __init__(self, interval_s: float = 5.0) -> None:
        self.host_cpu_percent = Gauge("host_cpu_percent", "Host CPU percent")
        self.host_mem_percent = Gauge("host_mem_percent", "Host memory percent")
        self.process_rss_bytes = Gauge("process_rss_bytes", "Service process resident memory in bytes")
        self.interval_s = interval_s
        self._process = psutil.Process()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        psutil.cpu_percent(interval=None)  # prime: the first call always returns 0.0

    def _run(self) -> None:
        while not self._stop.is_set():
            self.host_cpu_percent.set(psutil.cpu_percent(interval=None))
            self.host_mem_percent.set(psutil.virtual_memory().percent)
            self.process_rss_bytes.set(self._process.memory_info().rss)
            self._stop.wait(self.interval_s)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="hardware-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_s)
