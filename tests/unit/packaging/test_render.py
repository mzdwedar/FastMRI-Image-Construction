import re
from pathlib import Path

import pytest

from fastmri_knee.packaging.triton_repo import render

GOLDEN = Path(__file__).parent / "golden"
LEGACY = Path("deploy/services/triton/models")  # removed in T1, when Compose switches to the rendered repository
POLICY = re.compile(r"version_policy \{ [^\n]*\}\n")

CASES = {
    "cpu": {"profile": "cpu"},
    "nobatch": {"profile": "nobatch"},
    "gpu": {"profile": "gpu"},
    "gpu_tensorrt_fp16": {"profile": "gpu", "tensorrt": True, "fp16": True},
}


@pytest.mark.parametrize("name", CASES)
def test_render_matches_golden_file(name):
    assert render(version=3, **CASES[name]) == (GOLDEN / f"{name}.pbtxt").read_text()


@pytest.mark.skipif(not LEGACY.exists(), reason="legacy model repository already removed")
@pytest.mark.parametrize(("profile", "legacy_dir"), [("cpu", "unet_model"), ("nobatch", "unet_model_nobatch")])
def test_default_render_equals_todays_config_plus_specific_version_policy(profile, legacy_dir):
    legacy = (LEGACY / legacy_dir / "config.pbtxt").read_text()
    assert "version_policy { latest { num_versions: 1 } }\n" in legacy
    expected = POLICY.sub("version_policy { specific { versions: [ 7 ] } }\n", legacy)
    # the cpu template's comment on max_queue_delay is generic now that the delay is a variable
    expected = expected.replace("# wait up to 2 ms to fill a batch", "# wait up to this long to fill a batch")
    assert render(profile, version=7) == expected


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
