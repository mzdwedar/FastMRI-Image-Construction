"""Check a benchmark result against the committed baseline (SPEC-inference-server, performance budgets).

    uv run python -m benchmarks.compare                       # benchmarks/results/cpu.json vs baselines/cpu.json

Budgets: throughput at concurrency 16 >= 0.9x baseline; p99 at concurrency 1 <= 2x baseline; no errors;
(the old 'batching beats nobatch by 1.5x' budget was dropped: T4 measured 1.0x on 4 CPUs).
Exit code 1 when a budget is missed.

Nightly mode (bench.yml, advisory; the runner is not the baseline machine, so it needs its own baseline):

    uv run python -m benchmarks.compare --baseline benchmarks/baselines/ci.json --max-regression 0.3 --markdown out.md
    uv run python -m benchmarks.compare --report-only --markdown out.md      # no baseline yet: just the table

`--max-regression 0.3` fails only when c=16 throughput drops, or c=16 p99 rises, by more than 30 %, or on any error.
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


def check_regression(result: dict, baseline: dict, max_regression: float) -> list[str]:
    """Nightly gate: only a regression beyond `max_regression` at c=16, or any error, is a miss."""
    misses = []
    errors = sum(s["errors"] for s in result["steps"])
    if errors:
        misses.append(f"{errors} request errors")
    got, want = step(result, BATCH_CONCURRENCY), step(baseline, BATCH_CONCURRENCY)
    if got["throughput_rps"] < (1 - max_regression) * want["throughput_rps"]:
        misses.append(
            f"throughput at c={BATCH_CONCURRENCY}: {got['throughput_rps']:.1f} req/s is more than "
            f"{max_regression:.0%} below baseline {want['throughput_rps']:.1f}"
        )
    if got["latency_ms"]["p99"] > (1 + max_regression) * want["latency_ms"]["p99"]:
        misses.append(
            f"p99 at c={BATCH_CONCURRENCY}: {got['latency_ms']['p99']:.0f} ms is more than "
            f"{max_regression:.0%} above baseline {want['latency_ms']['p99']:.0f} ms"
        )
    return misses


def markdown(result: dict, baseline: dict | None, misses: list[str]) -> str:
    """A job-summary table: one row per concurrency, with the baseline's numbers beside ours when there is one."""
    head = "| c | req/s | p50 ms | p99 ms | errors |" + (" baseline req/s | baseline p99 ms |" if baseline else "")
    lines = [f"### Triton benchmark ({result['model']})", "", head, "|" + "---|" * (head.count("|") - 1)]
    for s in result["steps"]:
        row = f"| {s['concurrency']} | {s['throughput_rps']:.1f} | {s['latency_ms']['p50']:.0f} | {s['latency_ms']['p99']:.0f} | {s['errors']} |"
        if baseline:
            b = step(baseline, s["concurrency"])
            row += f" {b['throughput_rps']:.1f} | {b['latency_ms']['p99']:.0f} |"
        lines.append(row)
    lines.append("")
    lines += [f"**FAIL** {m}" for m in misses] or [
        "Gate: " + ("within budget." if baseline else "no baseline for this runner yet, so nothing was compared.")
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--result", type=Path, default=Path("benchmarks/results/cpu.json"))
    parser.add_argument("--baseline", type=Path, default=Path("benchmarks/baselines/cpu.json"))
    parser.add_argument("--max-regression", type=float, help="nightly gate: allowed fraction at c=16 (e.g. 0.3)")
    parser.add_argument("--report-only", action="store_true", help="no baseline: print/write the table, never fail")
    parser.add_argument("--markdown", type=Path, help="also write a job-summary table here")
    args = parser.parse_args()
    result = json.loads(args.result.read_text())
    baseline = None if args.report_only else json.loads(args.baseline.read_text())
    if baseline is None:
        misses = ["request errors"] if any(s["errors"] for s in result["steps"]) else []
    elif args.max_regression is not None:
        misses = check_regression(result, baseline, args.max_regression)
    else:
        misses = check(result, baseline)
    if args.markdown:
        args.markdown.write_text(markdown(result, baseline, misses))
    for line in misses:
        print("FAIL", line)
    if not misses:
        print("all performance budgets met")
    sys.exit(1 if misses else 0)


if __name__ == "__main__":
    main()
