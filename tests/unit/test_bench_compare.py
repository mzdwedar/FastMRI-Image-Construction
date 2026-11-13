from benchmarks.compare import check, check_regression, markdown


def result(rps16=100.0, p99_1=20.0, errors=0):
    return {
        "steps": [
            {"concurrency": 1, "errors": errors, "throughput_rps": 50.0, "latency_ms": {"p99": p99_1}},
            {"concurrency": 16, "errors": 0, "throughput_rps": rps16, "latency_ms": {"p99": 200.0}},
        ]
    }


def test_equal_result_meets_all_budgets():
    assert check(result(), result()) == []


def test_throughput_just_inside_and_outside_the_floor():
    assert check(result(rps16=90.5), result()) == []
    assert "throughput at c=16" in check(result(rps16=89.0), result())[0]


def test_p99_ceiling_and_errors():
    assert check(result(p99_1=39.0), result()) == []
    assert "p99 at c=1" in check(result(p99_1=41.0), result())[0]
    assert "errors" in check(result(errors=2), result())[0]


def test_nightly_gate_fails_only_beyond_thirty_percent_at_c16():
    assert check_regression(result(rps16=71.0), result(), 0.3) == []
    assert "throughput at c=16" in check_regression(result(rps16=69.0), result(), 0.3)[0]
    assert "errors" in check_regression(result(errors=1), result(), 0.3)[0]
    slow = result()
    slow["steps"][1]["latency_ms"]["p99"] = 261.0  # baseline helper's c=16 p99 is 200 ms
    assert "p99 at c=16" in check_regression(slow, result(), 0.3)[0]


def test_summary_table_has_a_row_per_step_and_says_when_nothing_was_compared():
    r = {"model": "unet_model", **result()}
    for s in r["steps"]:
        s["latency_ms"].update(p50=10.0)
    text = markdown(r, None, [])
    assert text.count("\n| 1 |") == 1 and text.count("\n| 16 |") == 1
    assert "no baseline for this runner yet" in text
    assert "**FAIL** boom" in markdown(r, r, ["boom"]) and "baseline req/s" in markdown(r, r, [])
