"""Model card: the provenance record stored next to every registered model version."""

import hashlib
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

VAL_METRICS = ("psnr", "ssim", "nmse")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_sha() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return out.stdout.strip()


def pick_val_metrics(metrics: dict[str, Any] | None) -> dict[str, float]:
    """Keep psnr/ssim/nmse if the run has them (MLflow may log them with a val_ prefix)."""
    found: dict[str, float] = {}
    for name in VAL_METRICS:
        for key in (name, f"val_{name}"):
            if metrics and key in metrics:
                found[name] = float(metrics[key])
                break
    return found


def build_card(
    *,
    checkpoint_sha256: str,
    onnx_sha256: str,
    opset: int,
    input_shape: tuple[int, ...],
    preprocess_version: str,
    parity_max_diff: float,
    git_sha: str,
    source_run_id: str | None,
    source: str,
    val_metrics: dict[str, float],
    hyperparameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """registry_version is filled in by `registry.register` once MLflow assigns it."""
    return {
        "registry_version": None,
        "source": source,
        "source_run_id": source_run_id,
        "git_sha": git_sha,
        "checkpoint_sha256": checkpoint_sha256,
        "onnx_sha256": onnx_sha256,
        "opset": opset,
        "input_names": ["input"],
        "output_names": ["output"],
        "input_shape": ["batch", *input_shape],
        "output_shape": ["batch", *input_shape],
        "dtype": "float32",
        "preprocess_version": preprocess_version,
        "parity_max_diff": parity_max_diff,
        "val_metrics": val_metrics,
        "hyperparameters": hyperparameters or {},
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
