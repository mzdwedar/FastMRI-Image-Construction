"""`run_summary.json`: provenance written next to the best checkpoint of a training run.

Colab runs leave only files on Drive behind, so the summary carries everything `register` needs
(checkpoint hash, preprocess version, val metrics) to accept the checkpoint on another machine.
"""

import json
from pathlib import Path
from typing import Any

import torch

from fastmri_knee.packaging import card
from fastmri_knee.preprocess import PREPROCESS_VERSION

SUMMARY_NAME = "run_summary.json"


def gpu_name() -> str | None:
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else None


def build_summary(
    checkpoint: Path,
    *,
    metrics: dict[str, Any],
    hyperparameters: dict[str, Any],
    source: str,
    global_step: int,
    resumed_from: str | None,
) -> dict[str, Any]:
    return {
        "source": source,
        "git_sha": card.git_sha(),
        "preprocess_version": PREPROCESS_VERSION,
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": card.sha256_file(checkpoint),
        "val_metrics": card.pick_val_metrics(metrics),
        "hyperparameters": hyperparameters,
        "global_step": global_step,
        "resumed_from": resumed_from,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": gpu_name(),
    }


def write_summary(checkpoint: Path, summary: dict[str, Any]) -> Path:
    """Write atomically so a Colab disconnect never leaves a half-written file."""
    path = checkpoint.parent / SUMMARY_NAME
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(summary, indent=2))
    tmp.replace(path)
    return path
