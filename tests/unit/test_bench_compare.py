from benchmarks.compare import check


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
    assert check(result(p99_1=23.0), result()) == []
    assert "p99 at c=1" in check(result(p99_1=23.5), result())[0]
    assert "errors" in check(result(errors=2), result())[0]


def test_batching_must_beat_nobatch_by_one_and_a_half():
    assert check(result(rps16=150.0), result(rps16=150.0), nobatch=result(rps16=100.0)) == []
    assert "batching gain" in check(result(rps16=140.0), result(rps16=140.0), nobatch=result(rps16=100.0))[0]
