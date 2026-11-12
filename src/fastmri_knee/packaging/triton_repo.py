"""Render Triton `config.pbtxt` files from the profile templates in deploy/triton/templates."""

from pathlib import Path
from string import Template

TEMPLATE_DIR = Path(__file__).resolve().parents[3] / "deploy" / "triton" / "templates"
PROFILES = ("cpu", "gpu", "nobatch")
MODEL_NAMES = {"cpu": "unet_model", "gpu": "unet_model", "nobatch": "unet_model_nobatch"}

_TENSORRT = """
optimization {
  execution_accelerators {
    gpu_execution_accelerator [ {
      name: "tensorrt"
      parameters { key: "precision_mode" value: "${precision}" }
    } ]
  }
}
"""


def render(
    profile: str,
    *,
    version: int,
    threads: int = 4,
    instances: int = 1,
    queue_delay_us: int = 2000,
    tensorrt: bool = False,
    fp16: bool = False,
) -> str:
    """Return the config.pbtxt text serving exactly `version`.

    `threads` is the CPU budget (SERVER_CPUS / TRITON_CPUS): ORT intra-op threads follow it, inter-op stays 1.
    `tensorrt` and `fp16` only apply to the gpu profile.
    """
    if profile not in PROFILES:
        raise ValueError(f"unknown Triton profile {profile!r}; expected one of {PROFILES}")
    if (tensorrt or fp16) and profile != "gpu":
        raise ValueError("tensorrt/fp16 are only valid for the gpu profile")
    if fp16 and not tensorrt:
        raise ValueError("fp16 needs tensorrt=True (precision_mode is a TensorRT accelerator setting)")
    optimization = Template(_TENSORRT).substitute(precision="FP16" if fp16 else "FP32") if tensorrt else ""
    values = {
        "model_name": MODEL_NAMES[profile],
        "version": int(version),
        "instances": int(instances),
        "queue_delay_us": int(queue_delay_us),
        "intra_threads": int(threads),
        "inter_threads": 1,
        "optimization": optimization,
    }
    return Template((TEMPLATE_DIR / f"{profile}.pbtxt").read_text()).substitute(values)
