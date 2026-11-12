"""Register a checkpoint as a new version of the MLflow model `fastmri-unet`.

Export to ONNX, refuse if ONNX Runtime disagrees with PyTorch, write a model card, then
create the version with model.onnx, model.ckpt and model_card.json attached.
"""

import json
import tempfile
from pathlib import Path
from typing import Any

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from fastmri_knee.model import UnetModel
from fastmri_knee.packaging import card as card_mod
from fastmri_knee.packaging.export import INPUT_SHAPE, check_parity, export
from fastmri_knee.preprocess import PREPROCESS_VERSION

MODEL_NAME = "fastmri-unet"
REGISTRY_EXPERIMENT = "model-registry"
OPSET = 17
PARITY_ATOL = 1e-4


class RegistryError(Exception):
    """Registration was refused; nothing was written to the registry."""


def _validate_summary(summary: dict[str, Any], checkpoint_sha256: str) -> None:
    if summary.get("checkpoint_sha256") != checkpoint_sha256:
        raise RegistryError(
            f"checkpoint sha256 {checkpoint_sha256[:12]}... does not match the run summary "
            f"({str(summary.get('checkpoint_sha256'))[:12]}...)"
        )
    if summary.get("preprocess_version") != PREPROCESS_VERSION:
        raise RegistryError(
            f"run summary preprocess_version={summary.get('preprocess_version')!r}, "
            f"installed package is {PREPROCESS_VERSION!r}"
        )


def register(
    checkpoint: Path,
    *,
    tracking_uri: str | None = None,
    summary: dict[str, Any] | None = None,
    run_id: str | None = None,
    source: str = "local",
) -> int:
    """Return the new registry version number. Raises RegistryError on any refusal."""
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    ckpt_sha = card_mod.sha256_file(checkpoint)
    if summary is not None:
        _validate_summary(summary, ckpt_sha)

    metrics: dict[str, Any] | None = summary.get("val_metrics") if summary else None
    if run_id:
        metrics = {**client.get_run(run_id).data.metrics, **(metrics or {})}
    source_run_id = run_id or (summary or {}).get("run_id")

    module = UnetModel.load_from_checkpoint(checkpoint, map_location="cpu")
    with tempfile.TemporaryDirectory() as tmp:
        onnx_path = Path(tmp) / "model.onnx"
        export(module, onnx_path, opset=OPSET)
        try:
            parity = check_parity(module, onnx_path, batch=4, atol=PARITY_ATOL)
        except AssertionError as exc:
            raise RegistryError(f"refusing to register: {exc}") from exc

        model_card = card_mod.build_card(
            checkpoint_sha256=ckpt_sha,
            onnx_sha256=card_mod.sha256_file(onnx_path),
            opset=OPSET,
            input_shape=INPUT_SHAPE,
            preprocess_version=PREPROCESS_VERSION,
            parity_max_diff=parity,
            git_sha=(summary or {}).get("git_sha") or card_mod.git_sha(),
            source_run_id=source_run_id,
            source=(summary or {}).get("source", source),
            val_metrics=card_mod.pick_val_metrics(metrics),
            hyperparameters=(summary or {}).get("hyperparameters"),
        )
        card_path = Path(tmp) / "model_card.json"
        card_path.write_text(json.dumps(model_card, indent=2))

        mlflow.set_experiment(REGISTRY_EXPERIMENT)
        with mlflow.start_run(run_name=f"register-{ckpt_sha[:8]}") as run:
            for path in (onnx_path, checkpoint, card_path):
                mlflow.log_artifact(str(path), artifact_path="model")
            try:
                client.create_registered_model(MODEL_NAME)
            except MlflowException as exc:
                if exc.error_code != "RESOURCE_ALREADY_EXISTS":
                    raise
            mv = client.create_model_version(
                MODEL_NAME, source=f"runs:/{run.info.run_id}/model", run_id=run.info.run_id
            )
            version = int(mv.version)

            # MLflow assigns the version number only now; write it into the card and re-upload.
            model_card["registry_version"] = version
            card_path.write_text(json.dumps(model_card, indent=2))
            mlflow.log_artifact(str(card_path), artifact_path="model")

    client.set_model_version_tag(MODEL_NAME, str(version), "preprocess_version", PREPROCESS_VERSION)
    client.set_model_version_tag(MODEL_NAME, str(version), "git_sha", model_card["git_sha"])
    return version
