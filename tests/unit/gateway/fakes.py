import numpy as np

from fastmri_knee.gateway.triton_client import TritonError


class FakeTritonClient:
    """Implements the `TritonClient` protocol. `script` entries are consumed per `infer` call: an Exception is
    raised, a callable transforms the batch, anything else is ignored (identity-plus-one output)."""

    def __init__(self, max_batch_size: int = 16, script: list | None = None, delay_s: float = 0.0):
        self.max_batch_size = max_batch_size
        self.script = list(script or [])
        self.delay_s = delay_s
        self.calls: list[int] = []  # batch size of every infer call, including retries
        self.closed = False

    async def get_max_batch_size(self) -> int:
        return self.max_batch_size

    async def infer(self, batch: np.ndarray, timeout_s: float) -> np.ndarray:
        import asyncio

        self.calls.append(len(batch))
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        step = self.script.pop(0) if self.script else None
        if isinstance(step, Exception):
            raise step
        if callable(step):
            return step(batch)
        return batch + 1.0

    async def close(self) -> None:
        self.closed = True


def unavailable() -> TritonError:
    return TritonError("UNAVAILABLE")
