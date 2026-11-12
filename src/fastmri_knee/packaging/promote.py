"""Promote a registry version to an alias (and, for `production`, deploy it); roll back to the previous one.

Gates (each refuses with a message; nothing is changed when one fires):
  * preprocess mismatch: the version's `preprocess_version` must equal the installed package's. Never bypassable.
  * random-init -> production: refused unless `allow_random` (or `force`, which is logged like the PSNR bypass).
  * PSNR regression: for `production`, val PSNR must be >= current production PSNR - 0.5 dB, unless `force`
    (logged, recorded on the version as the `promote_forced` tag). A version without PSNR can't be compared
    and is refused too, when the current production has one.
Then: tag the new version `previous_production`, move the alias, and (production only) run sync. If sync fails,
the alias and tag are put back. Rollback = promote `previous_production` of the current production, skipping the
PSNR and random-init gates (it already served), but not the preprocess gate.
"""

import json
import logging
from pathlib import Path
from typing import Any

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from fastmri_knee.packaging.registry import MODEL_NAME
from fastmri_knee.packaging.sync import SyncError, sync
from fastmri_knee.preprocess import PREPROCESS_VERSION

log = logging.getLogger(__name__)

PSNR_TOLERANCE_DB = 0.5
PRODUCTION = "production"
PREVIOUS_TAG = "previous_production"


class PromoteError(Exception):
    """Promotion refused or failed; the alias was left (or put back) as it was."""


def _production_version(client: MlflowClient) -> int | None:
    try:
        return int(client.get_model_version_by_alias(MODEL_NAME, PRODUCTION).version)
    except MlflowException:
        return None


def _psnr(version: int) -> float | None:
    card_path = mlflow.artifacts.download_artifacts(f"models:/{MODEL_NAME}/{version}/model_card.json")
    metrics: dict[str, Any] = json.loads(Path(card_path).read_text()).get("val_metrics") or {}
    return float(metrics["psnr"]) if "psnr" in metrics else None


def _check_gates(
    client: MlflowClient,
    version: int,
    alias: str,
    current: int | None,
    *,
    force: bool,
    allow_random: bool,
    skip_quality_gates: bool,
) -> list[str]:
    """Raise PromoteError on a failed gate; return the names of gates bypassed by `force`."""
    try:
        tags = client.get_model_version(MODEL_NAME, str(version)).tags
    except MlflowException as exc:
        raise PromoteError(f"{MODEL_NAME} has no version {version}: {exc}") from exc

    if tags.get("preprocess_version") != PREPROCESS_VERSION:
        raise PromoteError(
            f"version {version} was built with preprocess_version={tags.get('preprocess_version')!r}, "
            f"the installed package is {PREPROCESS_VERSION!r}; serving it would silently give wrong output"
        )
    if skip_quality_gates or alias != PRODUCTION:
        return []
    bypassed: list[str] = []
    if tags.get("random_init") == "true" and not allow_random:
        message = f"version {version} is random-init (untrained)"
        if not force:
            raise PromoteError(f"{message}; pass --allow-random (or --force) to put it in production")
        log.warning("--force: promoting despite failed random-init gate: %s", message)
        bypassed.append("random_init")
    if current is not None and current != version:
        candidate, baseline = _psnr(version), _psnr(current)
        if baseline is not None:
            if candidate is None:
                message = f"version {version} has no val PSNR to compare with production v{current} ({baseline:.2f} dB)"
            elif candidate < baseline - PSNR_TOLERANCE_DB:
                message = (
                    f"version {version} val PSNR {candidate:.2f} dB is more than {PSNR_TOLERANCE_DB} dB below "
                    f"production v{current} ({baseline:.2f} dB)"
                )
            else:
                message = ""
            if message:
                if not force:
                    raise PromoteError(f"{message}; pass --force to override")
                log.warning("--force: promoting despite failed PSNR gate: %s", message)
                bypassed.append("psnr")
    return bypassed


def promote(
    version: int,
    *,
    alias: str = PRODUCTION,
    force: bool = False,
    allow_random: bool = False,
    skip_quality_gates: bool = False,
    do_sync: bool = True,
    tracking_uri: str | None = None,
    **sync_kwargs: Any,
) -> int:
    """Return the version now carrying `alias`. `sync_kwargs` go to `sync` (profile, repo, triton_url, ...)."""
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    current = _production_version(client) if alias == PRODUCTION else None
    bypassed = _check_gates(
        client,
        version,
        alias,
        current,
        force=force,
        allow_random=allow_random,
        skip_quality_gates=skip_quality_gates,
    )

    v = str(version)
    previous = current if current is not None and current != version else None
    old_previous_tag = client.get_model_version(MODEL_NAME, v).tags.get(PREVIOUS_TAG)
    if previous is not None:
        client.set_model_version_tag(MODEL_NAME, v, PREVIOUS_TAG, str(previous))
    if bypassed:
        client.set_model_version_tag(MODEL_NAME, v, "promote_forced", ",".join(bypassed))
    client.set_registered_model_alias(MODEL_NAME, alias, v)
    log.info("alias %s -> v%s (was %s)", alias, version, previous)

    if alias == PRODUCTION and do_sync:
        try:
            sync(tracking_uri=tracking_uri, alias=alias, **sync_kwargs)
        except SyncError as exc:
            # sync resets the alias itself when it had loaded something; make sure in every other case too
            if previous is not None:
                client.set_registered_model_alias(MODEL_NAME, alias, str(previous))
            else:
                client.delete_registered_model_alias(MODEL_NAME, alias)
            if previous is not None:
                if old_previous_tag is None:
                    client.delete_model_version_tag(MODEL_NAME, v, PREVIOUS_TAG)
                else:
                    client.set_model_version_tag(MODEL_NAME, v, PREVIOUS_TAG, old_previous_tag)
            raise PromoteError(f"promotion of v{version} failed, {alias} restored: {exc}") from exc
    return version


def rollback(*, tracking_uri: str | None = None, **kwargs: Any) -> int:
    """Promote the `previous_production` version of the current production. Returns the version now serving."""
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    current = _production_version(client)
    if current is None:
        raise PromoteError("nothing to roll back: no version has the production alias")
    target = client.get_model_version(MODEL_NAME, str(current)).tags.get(PREVIOUS_TAG)
    if not target:
        raise PromoteError(f"production v{current} has no {PREVIOUS_TAG} tag; promote a specific version instead")
    return promote(int(target), alias=PRODUCTION, skip_quality_gates=True, tracking_uri=tracking_uri, **kwargs)
