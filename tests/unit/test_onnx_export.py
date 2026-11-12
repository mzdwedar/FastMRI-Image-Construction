import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

from fastmri_knee.packaging.export import check_parity

MODEL_REPO = Path("deploy/services/triton/models")


def test_onnx_matches_pytorch(onnx_model):
    module, path = onnx_model
    assert check_parity(module, path, batch=2) < 1e-4


def test_onnx_has_dynamic_batch_matching_triton_config(onnx_model):
    _, path = onnx_model
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    (inp,), (out,) = session.get_inputs(), session.get_outputs()
    assert (inp.name, out.name) == ("input", "output")
    assert isinstance(inp.shape[0], str) and inp.shape[1:] == [1, 320, 320]
    x = np.zeros((16, 1, 320, 320), dtype=np.float32)  # Triton's max_batch_size
    assert session.run(None, {"input": x})[0].shape == x.shape


def test_triton_configs_agree_on_io_and_threads():
    configs = {p.parent.name: p.read_text() for p in MODEL_REPO.glob("*/config.pbtxt")}
    assert set(configs) == {"unet_model", "unet_model_nobatch"}
    for name, text in configs.items():
        assert f'name: "{name}"' in text
        assert 'platform: "onnxruntime_onnx"' in text
        assert "dims: [ 1, 320, 320 ]" in text
        assert "response_cache" not in text, "a cache would inflate benchmark numbers"
    assert "dynamic_batching" in configs["unet_model"]
    assert "dynamic_batching" not in configs["unet_model_nobatch"]


def test_dashboards_use_rates_not_raw_counters():
    for path in Path("monitoring/grafana/dashboards").glob("*.json"):
        for panel in json.loads(path.read_text())["panels"]:
            for target in panel["targets"]:
                expr = target["expr"]
                if "_duration_us" in expr or "_total" in expr or "_bucket" in expr:
                    assert "rate(" in expr, f"{path.name} / {panel['title']}: counter plotted without rate()"
