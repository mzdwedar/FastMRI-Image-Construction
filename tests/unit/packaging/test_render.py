from pathlib import Path

import pytest

from fastmri_knee.packaging.triton_repo import render

GOLDEN = Path(__file__).parent / "golden"

CASES = {
    "cpu": {"profile": "cpu"},
    "nobatch": {"profile": "nobatch"},
    "gpu": {"profile": "gpu"},
    "gpu_tensorrt_fp16": {"profile": "gpu", "tensorrt": True, "fp16": True},
}


@pytest.mark.parametrize("name", CASES)
def test_render_matches_golden_file(name):
    assert render(version=3, **CASES[name]) == (GOLDEN / f"{name}.pbtxt").read_text()


def test_version_threads_instances_and_queue_delay_are_substituted():
    text = render("cpu", version=12, threads=8, instances=2, queue_delay_us=500)
    assert "versions: [ 12 ]" in text
    assert 'key: "intra_op_thread_count" value: { string_value: "8" }' in text
    assert "count: 2 }" in text
    assert "max_queue_delay_microseconds: 500" in text
    assert "${" not in text


def test_profiles_differ_where_they_should():
    assert "dynamic_batching" in render("cpu", version=1)
    assert "dynamic_batching" not in render("nobatch", version=1)
    assert 'name: "unet_model_nobatch"' in render("nobatch", version=1)
    assert "KIND_GPU" in render("gpu", version=1) and "KIND_CPU" not in render("gpu", version=1)
    assert "gpu_execution_accelerator" not in render("gpu", version=1)


@pytest.mark.parametrize(
    "kwargs",
    [{"profile": "tpu"}, {"profile": "cpu", "tensorrt": True}, {"profile": "gpu", "fp16": True}],
)
def test_invalid_combinations_are_rejected(kwargs):
    with pytest.raises(ValueError):
        render(version=1, **kwargs)


@pytest.mark.parametrize("profile", ["cpu", "gpu"])
def test_batched_profiles_have_queue_policy_and_warmup(profile):
    text = render(profile, version=1)
    assert "timeout_action: REJECT" in text
    assert "default_timeout_microseconds: 5000000" in text
    assert "max_queue_size: 256" in text
    assert "batch_size: 1\n" in text and "batch_size: 16\n" in text
    assert "zero_data: true" in text


def test_nobatch_baseline_is_left_unhardened():
    text = render("nobatch", version=1)
    assert "default_queue_policy" not in text and "model_warmup" not in text


def test_cpu_never_waits_to_fill_a_batch_but_keeps_load_shedding():
    cpu = render("cpu", version=1)
    assert "max_queue_delay_microseconds: 0" in cpu and "preferred_batch_size" not in cpu
    assert "max_queue_size: 256" in cpu and "timeout_action: REJECT" in cpu
    assert "max_queue_delay_microseconds: 2000" in render("gpu", version=1)  # GPU keeps its delay (not measured)
