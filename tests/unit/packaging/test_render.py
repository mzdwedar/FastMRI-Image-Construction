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
