"""Materialize the registry's `production` model into the Triton model repository and load it.

Idempotent. The repository is written only here. Steps: resolve alias -> download model.onnx and verify its
sha256 against the model card -> write version dir and config.pbtxt atomically -> ask Triton to load (explicit
control mode) -> wait until that version is ready -> prune old version directories. A failed load restores the
previous config and version, reloads it and resets the alias.
"""

import json
import os
import re
import shutil
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

import httpx
import mlflow
from mlflow.tracking import MlflowClient

from fastmri_knee.packaging.card import sha256_file
from fastmri_knee.packaging.registry import MODEL_NAME
from fastmri_knee.packaging.triton_repo import MODEL_NAMES, render

DEFAULT_REPO = Path("deploy/model_repository")
READY_TIMEOUT_S = 60.0
_VERSION_RE = re.compile(r"versions:\s*\[\s*(\d+)\s*\]")


class SyncError(Exception):
    """Sync failed; the repository was left (or restored) as it was."""


def served_version(config_text: str) -> int | None:
    match = _VERSION_RE.search(config_text)
    return int(match.group(1)) if match else None


def _atomic_write(path: Path, text: str) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".config-")
    with os.fdopen(fd, "w") as f:
        f.write(text)
    os.replace(tmp, path)


def _is_ready(http: httpx.Client, name: str, version: int) -> bool:
    try:
        return http.get(f"/v2/models/{name}/versions/{version}/ready").status_code == 200
    except httpx.HTTPError:
        return False


def _load_and_wait(
    http: httpx.Client,
    name: str,
    version: int,
    timeout: float,
    sleep: Callable[[float], None],
    clock: Callable[[], float],
) -> None:
    response = http.post(f"/v2/repository/models/{name}/load")
    if response.status_code != 200:
        raise SyncError(f"Triton refused to load {name}: HTTP {response.status_code} {response.text[:200]}")
    deadline = clock() + timeout
    while not _is_ready(http, name, version):
        if clock() >= deadline:
            raise SyncError(f"{name} version {version} not ready after {timeout:.0f}s")
        sleep(1.0)


def sync(
    *,
    tracking_uri: str | None = None,
    alias: str = "production",
    profile: str = "cpu",
    threads: int = 4,
    instances: int = 1,
    repo: Path = DEFAULT_REPO,
    triton_url: str = "http://localhost:8100",
    timeout: float = READY_TIMEOUT_S,
    http: httpx.Client | None = None,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> int:
    """Return the version now served. Raises SyncError on any failure."""
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    try:
        version = int(client.get_model_version_by_alias(MODEL_NAME, alias).version)
    except mlflow.exceptions.MlflowException as exc:
        raise SyncError(f"no model version has alias {alias!r} on {MODEL_NAME}: {exc}") from exc

    name = MODEL_NAMES[profile]
    model_dir = repo / name
    config_path = model_dir / "config.pbtxt"
    version_dir = model_dir / str(version)
    wanted_config = render(profile, version=version, threads=threads, instances=instances)

    with http or httpx.Client(base_url=triton_url, timeout=10.0) as session:
        previous_config = config_path.read_text() if config_path.exists() else None
        previous = served_version(previous_config) if previous_config else None
        if previous == version:
            previous = None  # re-sync of the served version: nothing to fall back to except itself

        card = json.loads(
            Path(mlflow.artifacts.download_artifacts(f"models:/{MODEL_NAME}/{version}/model_card.json")).read_text()
        )
        expected_sha = card["onnx_sha256"]
        onnx_in_repo = version_dir / "model.onnx"
        up_to_date = (
            previous_config == wanted_config and onnx_in_repo.exists() and sha256_file(onnx_in_repo) == expected_sha
        )
        if up_to_date and _is_ready(session, name, version):
            return version  # idempotent: no filesystem change, no reload

        created_version_dir = False
        if not up_to_date:
            with tempfile.TemporaryDirectory() as tmp:
                downloaded = Path(
                    mlflow.artifacts.download_artifacts(f"models:/{MODEL_NAME}/{version}/model.onnx", dst_path=tmp)
                )
                actual = sha256_file(downloaded)
                if actual != expected_sha:
                    raise SyncError(
                        f"model.onnx sha256 {actual[:12]}... does not match the model card ({expected_sha[:12]}...)"
                    )
                model_dir.mkdir(parents=True, exist_ok=True)
                staging = Path(tempfile.mkdtemp(dir=model_dir, prefix=".staging-"))
                try:
                    (staging / "model.onnx").write_bytes(downloaded.read_bytes())
                    created_version_dir = not version_dir.exists()
                    if version_dir.exists():
                        shutil.rmtree(version_dir)
                    os.replace(staging, version_dir)
                finally:
                    shutil.rmtree(staging, ignore_errors=True)
            _atomic_write(config_path, wanted_config)

        try:
            _load_and_wait(session, name, version, timeout, sleep, clock)
        except (SyncError, httpx.HTTPError) as exc:
            _restore(
                session,
                client,
                alias,
                name,
                config_path,
                previous_config,
                previous,
                version_dir,
                created_version_dir,
                timeout,
                sleep,
                clock,
            )
            raise SyncError(f"loading {name} version {version} failed, previous state restored: {exc}") from exc

    keep = {str(version)} | ({str(previous)} if previous is not None else set())
    for child in model_dir.iterdir():
        if child.is_dir() and child.name.isdigit() and child.name not in keep:
            shutil.rmtree(child)
    return version


def _restore(
    session: httpx.Client,
    client: MlflowClient,
    alias: str,
    name: str,
    config_path: Path,
    previous_config: str | None,
    previous: int | None,
    version_dir: Path,
    created_version_dir: bool,
    timeout: float,
    sleep: Callable[[float], None],
    clock: Callable[[], float],
) -> None:
    if previous_config is None:
        config_path.unlink(missing_ok=True)
    else:
        _atomic_write(config_path, previous_config)
    if created_version_dir:
        shutil.rmtree(version_dir, ignore_errors=True)
    if previous is not None:
        client.set_registered_model_alias(MODEL_NAME, alias, str(previous))  # before the reload: it may fail too
    try:
        if previous is not None:
            _load_and_wait(session, name, previous, timeout, sleep, clock)
        else:
            session.post(f"/v2/repository/models/{name}/unload")
    except (SyncError, httpx.HTTPError):
        pass  # the original failure is the one reported
