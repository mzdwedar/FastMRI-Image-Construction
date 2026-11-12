# Spec: model-packaging

Module of [SPEC.md](SPEC.md). Depends on: `foundation`.

## Objective

Make the path from a trained checkpoint to a served model a reproducible, auditable, one-command operation, with
one-command rollback. Today a checkpoint is mounted by hand and ONNX files are copied into the Triton repo by
`scripts/convert_to_onnx.py`; nothing records which model is served or how it was produced.

## Lifecycle

```
train run on Colab (Drive: checkpoint + run_summary.json [+ MLflow file store])
   │  download from Drive, then: make register CKPT=… SUMMARY=…   (or RUN_ID=… for local MLflow runs)
   ▼
export ONNX (opset 17, dynamic batch) → parity check vs PyTorch → model card
   │
   ▼
MLflow registered model "fastmri-unet", version n   (artifacts: model.onnx, model.ckpt, model_card.json)
   │  make promote VERSION=n            (alias production → n)
   ▼
sync: deploy/model_repository/unet_model/{n}/model.onnx + config.pbtxt (version_policy specific [n])
   │  Triton repository API: load unet_model
   ▼
served; gateway reports model_version=n
```

## Requirements

1. **MLflow service.** Compose runs `mlflow server` (pinned image) with SQLite backend and a local artifact volume
   at `http://mlflow:5000` (host port 5001). Training logs to it when `MLFLOW_TRACKING_URI` is set; the file store
   stays the offline fallback. Upgrade MLflow from 2.12.1 to current 2.x (ask first if 3.x).
2. **Register** (`python -m fastmri_knee.packaging register --checkpoint PATH [--summary PATH] | --run-id ID`):
   - `--summary` is the `run_summary.json` written by training (Colab path): source (`colab`), git sha,
     `preprocess_version`, hyperparameters, val metrics, checkpoint sha256. Register refuses if the checkpoint hash
     doesn't match the summary, or if the summary's `preprocess_version` differs from the installed package;
   - exports ONNX (reuse `export()`), runs parity on a fixed seeded batch of 4; **refuses to register** if
     max |torch − onnxruntime| > 1e-4;
   - writes `model_card.json`: registry version, source run id, git sha, checkpoint sha256, onnx sha256, opset,
     input/output names, shapes, dtype, `preprocess_version`, parity max diff, val metrics (psnr, ssim, nmse) if the
     run has them, created_at;
   - creates a new version of `fastmri-unet` with `model.onnx`, `model.ckpt`, `model_card.json`; tags
     `preprocess_version` and `git_sha` on the version.
3. **Random-init model** (`make model-random`): same path with untrained weights, tagged `random_init=true`.
   Used by CI, tests, benchmarks. Must be blocked from the `production` alias unless `--allow-random`.
4. **Promote** (`python -m fastmri_knee.packaging promote --version n [--alias production]`):
   - gate: refuse if the version's `preprocess_version` differs from the gateway's (from the shared package);
   - gate: for `production`, refuse if val PSNR < current production PSNR − 0.5 dB, unless `--force` (logged);
   - records the previous version in alias history (MLflow tag `previous_production` on the new version);
   - sets the alias, then runs **sync**.
5. **Sync** (`python -m fastmri_knee.packaging sync`): idempotent. Resolves alias `production` → version n;
   downloads `model.onnx` into `deploy/model_repository/unet_model/n/` (verifies sha256 against the card);
   renders `config.pbtxt` from `deploy/triton/templates/<profile>.pbtxt` with `version_policy: { specific {
   versions: [n] } }`; writes atomically (temp dir + rename); calls Triton `POST
   /v2/repository/models/unet_model/load`; waits until `/v2/models/unet_model/versions/n/ready`, timeout 60 s;
   removes version directories other than n and the previous version (keeps rollback on disk).
   On load failure: restore the previous config, reload, exit non-zero. Alias is reset to the previous version.
6. **Rollback** (`make rollback`): promote the `previous_production` version, skipping the PSNR gate.
7. **Profiles.** Templates `cpu.pbtxt` (current `unet_model` config), `gpu.pbtxt` (`KIND_GPU`, optional TensorRT
   execution accelerator, FP16 flag), `nobatch.pbtxt` (benchmark baseline). Profile from `TRITON_PROFILE` env,
   default `cpu`. Thread counts are template variables tied to `SERVER_CPUS`.
8. **Gitignore** `deploy/model_repository/`; the committed `deploy/services/triton/models/` tree is replaced by
   templates.

## Commands

```
make register CKPT=artifacts/model.ckpt     # or RUN_ID=<mlflow run id>
make model-random                           # register + promote an untrained model (staging by default)
make promote VERSION=3 [ALIAS=production] [FORCE=1]
make rollback
make sync                                   # re-materialize the repo from the registry (after host restart)
```

## Testing strategy

- Unit (temp MLflow file store, no Docker): register writes a version with all three artifacts and correct card
  fields; parity failure (monkeypatched exporter adding noise) refuses registration; promote gates (preprocess
  mismatch, PSNR regression, random-init to production) each refuse with a clear message; sync renders the expected
  `config.pbtxt` (golden file) and is idempotent (second run makes no filesystem changes); sha mismatch aborts.
- Integration (`integration`, Triton container): register two random-init versions; promote 1 then 2 → Triton
  reports version 2 ready and version 1 unloaded; rollback → version 1 ready; corrupt ONNX → sync fails, version 2
  still served.
- E2E: promotion under concurrent load (`benchmarks/load_test.py`, 20 s, concurrency 8) has zero failed requests.

## Boundaries

- Always: verify checksums before loading; keep the previous version on disk until the next promotion.
- Ask first: MLflow 3.x; Postgres backend instead of SQLite; S3/MinIO artifact store.
- Never: write into the model repository from anywhere except `sync`; promote without a model card.

## Success criteria

- [ ] `make model-random && make promote VERSION=1` on a fresh stack → Triton serves `unet_model` v1 and
      `GET /v1/model` on the gateway returns version 1 with the card.
- [ ] Promote/rollback each complete in ≤ 60 s with zero failed requests under load.
- [ ] Each served model can be traced to git sha, run id, and checkpoint hash from its model card.
- [ ] Unit + integration tests above pass; coverage ≥ 85% on `packaging/`.

## Open questions

1. Should `staging` be served side by side (Triton serves both versions; gateway routes `X-Model-Alias: staging`)?
   Proposal: not in this pass; spec it with model-monitoring if shadow traffic is wanted.
2. Triton model control: `explicit` (sync calls the load API) vs `poll`. Proposal: `explicit`, so loads are
   deliberate and failures are visible to the caller.
