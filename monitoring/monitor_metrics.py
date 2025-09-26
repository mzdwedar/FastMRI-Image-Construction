from prometheus_client import Counter, Gauge
import psutil
import time
from prometheus_client import Histogram


class inferenceMonitor:
    """Class representing an inference monitor that tracks the number of inferences made."""

    def __init__(self):
        """
        Initializes the metrics for monitoring prediction and evaluation requests:
        - predict_requests_total (Counter): A counter to track the total number of prediction requests.
        - evaluate_requests_total (Counter): A counter to track the total number of evaluation requests.
        - inference_latency (Histogram): A histogram to measure the latency of model predictions, 
                                           with specified buckets for latency in seconds.
        """

        self.predict_requests_total = Counter("predict_requests_total", "Total number of predict requests")
        self.evaluate_requests_total = Counter("evaluate_requests_total", "Total number of evaluate requests")
        self.inference_latency = Histogram(
                                    name="predict_inference_latency_seconds",
                                    documentation="Time spent in model.predict() for /predict endpoint",
                                    buckets=(0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10)  # 10ms → 10s
                                    )
    
    def increment_predict_count(self) -> None:
        """
        Increment the inference counter by one.
        
        ---
        Returns:
            None
        """
        self.predict_requests_total.inc()
    
    def increment_evaluate_count(self) -> None:
        """
        Increment the evaluation counter by one.

        ---
        Returns:
            None
        """
        self.evaluate_requests_total.inc()

    def observe_latency(self, duration: float) -> None:
        """
        Observe and record the latency of an inference operation.
        Args:
            duration (float): The time taken for the inference operation in seconds.
            
        ---
        Returns:
            None
        """
        self.inference_latency.observe(duration)

class hardwareMonitor:
    """Class representing a hardware monitor that tracks CPU and memory usage."""

    def __init__(self):
        """
        Initialize following metrics:
        - `host_cpu_percent`: Gauge to track the percentage of CPU usage on the host.
        - `host_mem_percent`: Gauge to track the percentage of memory usage on the host.
        - `process_rss_bytes`: Gauge to track the Resident Set Size (RSS) in bytes of the application process.
        Additionally, it initializes a `psutil.Process` instance to monitor the current process and primes the CPU percent calculation to avoid an initial value of 0.0.
        """
        self.host_cpu_percent = Gauge("host_cpu_percent", "Host CPU percent")
        self.host_mem_percent = Gauge("host_mem_percent", "Host memory percent")
        self.process_rss_bytes = Gauge("process_rss_bytes", "App process RSS bytes")
        self.process = psutil.Process()

        # Prime cpu_percent to avoid initial 0.0
        psutil.cpu_percent(interval=None)

    def sample(self) -> None:
        """
        Sample system the following process metrics at regular intervals:
        - Host CPU usage percentage
        - Host memory usage percentage
        - Resident Set Size (RSS) of the process memory

        ---
        Returns:
            None
        """

        while True:
            try:
                self.host_cpu_percent.set(psutil.cpu_percent(interval=None))
                self.host_mem_percent.set(psutil.virtual_memory().percent)
                self.process_rss_bytes.set(self.process.memory_info().rss)
            except Exception:
                # Do not crash the sampler; just continue
                pass
            time.sleep(5)