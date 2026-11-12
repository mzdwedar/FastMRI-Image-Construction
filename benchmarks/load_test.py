"""Closed-loop load test: same payload, same client, every backend.

    make up-dev && make sync                      # Triton HTTP on :8100
    make bench NAME=cpu_1x4                       # writes benchmarks/results/cpu_1x4.json
    make bench NAME=nobatch MODEL=unet_model_nobatch   # after `TRITON_PROFILE=nobatch make sync`

Talks to Triton directly (no gateway), so the numbers are the model server's own.
Each request carries ONE preprocessed 320x320 slice, the shape of interactive traffic
(a viewer requesting slices as the radiologist scrolls). N workers each keep exactly
one request in flight, so N is the concurrency. Per step: warm-up, then a timed window.
Requests use the binary tensor extension of the KServe v2 protocol, so nobody pays for
JSON-encoding 400 KB of floats.
"""

import argparse
import asyncio
import json
import platform
import time
from pathlib import Path

import httpx
import numpy as np

SHAPE = (1, 1, 320, 320)


def triton_request(base: str, model: str, payload: bytes):
    header = json.dumps(
        {
            "inputs": [
                {
                    "name": "input",
                    "shape": list(SHAPE),
                    "datatype": "FP32",
                    "parameters": {"binary_data_size": len(payload)},
                }
            ],
            "outputs": [{"name": "output", "parameters": {"binary_data": True}}],
        }
    ).encode()
    return dict(
        url=f"{base}/v2/models/{model}/infer",
        content=header + payload,
        headers={"Content-Type": "application/octet-stream", "Inference-Header-Content-Length": str(len(header))},
    )


def check_response(resp: httpx.Response) -> None:
    resp.raise_for_status()
    expected = int(np.prod(SHAPE)) * 4
    body = resp.content[int(resp.headers["Inference-Header-Content-Length"]) :]
    if len(body) != expected:
        raise ValueError(f"unexpected output size {len(body)} != {expected}")


async def run_step(client, req, concurrency, warmup_s, duration_s):
    latencies, errors, glitches, error_kinds = [], 0, 0, set()
    t_start = time.perf_counter()
    t_measure, t_end = t_start + warmup_s, t_start + warmup_s + duration_s

    async def worker():
        nonlocal errors, glitches
        while (now := time.perf_counter()) < t_end:
            try:
                resp = await client.post(**req)
                check_response(resp)
                done = time.perf_counter()
                if now >= t_measure:
                    latencies.append(done - now)
            except Exception as exc:
                if "illegal header line" in str(exc):
                    glitches += 1  # Triton's HTTP binary path mis-frames ~1 response in 20k; the gateway uses gRPC
                    continue
                errors += 1
                error_kinds.add(f"{type(exc).__name__}: {str(exc)[:120]}")

    await asyncio.gather(*(worker() for _ in range(concurrency)))
    lat_ms = np.array(latencies) * 1000
    return {
        "concurrency": concurrency,
        "requests": len(latencies),
        "errors": errors,
        "http_framing_glitches": glitches,
        "error_kinds": sorted(error_kinds),
        "throughput_rps": len(latencies) / duration_s,
        "latency_ms": {
            "mean": float(lat_ms.mean()) if len(lat_ms) else None,
            **{f"p{p}": float(np.percentile(lat_ms, p)) if len(lat_ms) else None for p in (50, 90, 95, 99)},
        },
    }


async def main_async(args):
    payload = np.random.default_rng(0).random(SHAPE, dtype=np.float32).tobytes()
    req = triton_request(args.url, args.model, payload)

    limits = httpx.Limits(
        max_connections=max(args.concurrency) + 8, max_keepalive_connections=max(args.concurrency) + 8
    )
    async with httpx.AsyncClient(timeout=120, limits=limits) as client:
        resp = await client.post(**req)  # fail fast if the backend isn't up
        check_response(resp)
        steps = []
        for c in args.concurrency:
            step = await run_step(client, req, c, args.warmup, args.duration)
            lat = step["latency_ms"]
            print(
                f"c={c:>3}  {step['throughput_rps']:7.1f} req/s   p50 {lat['p50']:8.1f} ms   "
                f"p95 {lat['p95']:8.1f} ms   p99 {lat['p99']:8.1f} ms   errors {step['errors']}",
                flush=True,
            )
            steps.append(step)
            await asyncio.sleep(args.cooldown)
    return {
        "model": args.model,
        "payload": {"slices_per_request": SHAPE[0], "shape": list(SHAPE), "dtype": "float32"},
        "step": {"warmup_s": args.warmup, "duration_s": args.duration},
        "client_host": {"machine": platform.machine(), "system": platform.system()},
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "steps": steps,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="unet_model", help="Triton model name")
    parser.add_argument("--url", default="http://localhost:8100", help="Triton HTTP base URL (dev overlay port)")
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--warmup", type=float, default=5)
    parser.add_argument("--duration", type=float, default=30)
    parser.add_argument("--cooldown", type=float, default=3)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = asyncio.run(main_async(args))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")
        print("wrote", args.out)


if __name__ == "__main__":
    main()
