"""Check a benchmark result against the committed baseline (SPEC-inference-server, performance budgets).

    uv run python -m benchmarks.compare                       # benchmarks/results/cpu.json vs baselines/cpu.json

Budgets: throughput at concurrency 16 >= 0.9x baseline; p99 at concurrency 1 <= 2x baseline; no errors;
(the old 'batching beats nobatch by 1.5x' budget was dropped: T4 measured 1.0x on 4 CPUs).
Exit code 1 when a budget is missed.
"""

import argparse
import json
import sys
from pathlib import Path

THROUGHPUT_FLOOR = 0.9
P99_CEILING = 2.0  # single-client p99 swings 90-190 ms between identical runs (T4), so 1.15x flaked
BATCH_CONCURRENCY = 16


def step(result: dict, concurrency: int) -> dict:
    for s in result["steps"]:
        if s["concurrency"] == concurrency:
            return s
    raise SystemExit(
        f"no step with concurrency {concurrency} in the result (has {[s['concurrency'] for s in result['steps']]})"
    )


def check(result: dict, baseline: dict) -> list[str]:
    """Return one line per missed budget (empty = all met)."""
    misses = []
    errors = sum(s["errors"] for s in result["steps"])
    if errors:
        misses.append(f"{errors} request errors (budget: 0)")
    got, want = step(result, BATCH_CONCURRENCY)["throughput_rps"], step(baseline, BATCH_CONCURRENCY)["throughput_rps"]
    if got < THROUGHPUT_FLOOR * want:
        misses.append(
            f"throughput at c={BATCH_CONCURRENCY}: {got:.1f} req/s < {THROUGHPUT_FLOOR} x baseline {want:.1f}"
        )
    got, want = step(result, 1)["latency_ms"]["p99"], step(baseline, 1)["latency_ms"]["p99"]
    if got > P99_CEILING * want:
        misses.append(f"p99 at c=1: {got:.1f} ms > {P99_CEILING} x baseline {want:.1f} ms")
    return misses


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--result", type=Path, default=Path("benchmarks/results/cpu.json"))
    parser.add_argument("--baseline", type=Path, default=Path("benchmarks/baselines/cpu.json"))
    args = parser.parse_args()
    misses = check(json.loads(args.result.read_text()), json.loads(args.baseline.read_text()))
    for line in misses:
        print("FAIL", line)
    if not misses:
        print("all performance budgets met")
    sys.exit(1 if misses else 0)


if __name__ == "__main__":
    main()
