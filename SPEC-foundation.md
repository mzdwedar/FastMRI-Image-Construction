# Spec: foundation

Module of [SPEC.md](SPEC.md). Depends on: —

## Objective

Give every later module a clean, reproducible base: one installable package, one environment, one command runner,
a test suite that actually runs, a repo free of generated clutter, and a preprocessing function verified to be
correct and usable without torch (the gateway depends on this).

## Current state (baseline `d9abbd6`)

- Code split across `src/`, `scripts/`, `deploy/`, `monitoring/`, imported via `pythonpath = .`; a stale
  `src/fastmri_knee.egg-info` and `build/lib/` exist.
- `pytest.ini` sets `testpaths = tests/serving`, so `tests/data` and `tests/code` never run.
- `split_data.py` (untracked) imports the deleted `src.utils`; it duplicates `src/data.py:split_singlecoil_subjects`.
- `FastMRIDataModule.prepare_data` downloads with `aria2c` from an env var `url` inside the training path.
- Preprocessing (`FastMRITransform`) is torch-only.
- Four venvs (`.venv`, `.venv-cpu`, `.venv-dev`, `.venv-gpu`); `.venv` was broken and has been recreated empty.
- Python pinned to `==3.10.*`, which reached end of life in Oct 2026.

## Requirements

1. **Package layout.** Move code into `src/fastmri_knee/` per the layout in SPEC.md, installed by `uv sync`
   (no `pythonpath` hacks). Hydra entrypoints run as `python -m fastmri_knee.<cli>`. `scripts/` is removed once
   empty.
2. **Dependency extras.** `pyproject.toml` splits deps into extras: `train` (torch, lightning, smp, torchmetrics,
   fastmri, hydra, mlflow, optuna), `export` (onnx, onnxruntime), `gateway` (fastapi, uvicorn, tritonclient[grpc],
   pydantic-settings, prometheus-client, structlog, opentelemetry), with `numpy` and `h5py` in core. Dev tools go in
   `[dependency-groups] dev`. CPU torch index stays the default for macOS/CI.
3. **Single environment.** `make setup` creates `.venv` from `uv.lock` (`--frozen`). Remove the other venvs
   (ask first; they are local dirs).
4. **numpy preprocessing.** `fastmri_knee/preprocess.py` implements k-space → normalized 320×320 magnitude image
   with numpy only (no torch import, enforced by a test that imports it with torch blocked). The training dataset
   calls it, then converts to a tensor, so training and serving share one implementation.
5. **Preprocessing correctness.** Verify the FFT-shift question (Open question 1) and record the decision here.
   Version the preprocessing (`PREPROCESS_VERSION = "1"` or `"2"`) so packaging can stamp it on each model version.
6. **Data acquisition out of the training path.** Move the download/extract/split into `python -m
   fastmri_knee.data.prepare` (`make data`); `prepare_data` only checks that the data exists and gives a clear error.
7. **Tests run.** `pytest.ini` removed in favour of `[tool.pytest.ini_options]` with markers `integration`, `e2e`,
   `training`; `testpaths = ["tests"]`. Tests move to `tests/{unit,integration,e2e,training}`.
8. **Makefile** with `help`, `setup`, `lint`, `format`, `typecheck`, `test`, `test-integration`, `test-e2e`, `data`,
   `train`, `clean`. Later modules add targets.
9. **Repo cleanup.** Remove `build/`, `preds/`, `results/`, `triton_models/`, `lightning_logs/`, `unet_model.onnx`,
   `src/*.egg-info`, `split_data.py`, untrack `grafana_latency.png` (move to `docs/img/` if the README uses it).
   `latest_version.txt` (untracked, unknown purpose): ask before deleting.
10. **Colab-ready training.** Training runs on Google Colab: the package installs with
    `pip install "fastmri-knee[train] @ git+https://github.com/mzdwedar/fastmri@<sha>"`, so `requires-python` must
    include Colab's Python version. Paths (data, checkpoints, outputs) are Hydra overrides pointing to Drive;
    `accelerator: auto`. Every run writes `run_summary.json` next to the best checkpoint (git sha,
    `preprocess_version`, hyperparameters, val metrics, checkpoint sha256) for registration (SPEC-model-packaging).
    `notebooks/train_colab.ipynb` is the supported entry point.
11. **Tooling.** ruff (`E,F,I,B,UP,SIM`) + `ruff format` replaces black; mypy strict for `preprocess.py` (and
    later `gateway/`, `packaging/`); pre-commit with ruff and gitleaks. `.env.example` documents every variable.

## Commands

```
make setup       uv sync --frozen --extra dev --extra train --extra export --extra gateway
make lint        uv run ruff check . && uv run ruff format --check .
make typecheck   uv run mypy
make test        uv run pytest -m "not integration and not e2e and not training"
make data        uv run python -m fastmri_knee.data.prepare --root data
make train       uv run python -m fastmri_knee.train
```

## Testing strategy

- `tests/unit/test_preprocess.py`: numpy preprocess vs the (current) torch transform on synthetic k-space,
  `atol=1e-6`; output shape `(1, 320, 320)`, dtype float32, range `[0, 1]`; odd/even widths; zero k-space doesn't
  produce NaN.
- `tests/unit/test_preprocess_no_torch.py`: imports `fastmri_knee.preprocess` in a subprocess with
  `sys.modules["torch"] = None` and passes.
- `tests/unit/test_fftshift.py` (after the decision): a synthetic centered phantom round-trips through
  `fft → preprocess` with its center of mass in the center of the crop.
- Existing `tests/data` and `tests/code`: tests needing real data get the `training` marker; the rest run in unit.

## Boundaries

- Always: keep behaviour identical during the move (pure refactor commits separate from behaviour changes).
- Ask first: changing preprocessing semantics; Python version bump; deleting venvs or `latest_version.txt`.
- Never: change model architecture or training hyperparameters in this module.

## Success criteria

- [ ] `rm -rf .venv && make setup && make lint typecheck test` passes on macOS and Ubuntu.
- [ ] `grep -r "pythonpath\|sys.path" src tests` returns nothing; `python -c "import fastmri_knee"` works after setup.
- [ ] `pytest --collect-only -q` collects tests from all of `tests/unit`, and `-m training` selects only data tests.
- [ ] `fastmri_knee.preprocess` imports with torch unavailable; parity test passes.
- [ ] FFT-shift decision recorded in this spec with evidence (image of one real slice: input vs target).
- [ ] `git status` on a fresh clone after `make test` is clean (no generated files outside ignored dirs).

## Open questions

1. **FFT shift (suspected correctness bug).** `ifft2c` is `torch.fft.ifft2(k, norm="ortho")` without
   `ifftshift`/`fftshift`. fastMRI stores centered k-space, so `|ifft2(k)|` equals the correct image circularly
   shifted by half in each axis; the 320×320 center crop then takes the wrong region, misaligned with
   `reconstruction_rss`. **Confirmed on a synthetic phantom (2026-11-12):** `|ifft2(k)| == fftshift(correct)`
   exactly, and 0% of a centered phantom lands inside the 320×320 crop (vs 100% with shifts). Remaining step:
   confirm on one real slice, then fix (`PREPROCESS_VERSION = "2"`) and retrain. Infra work proceeds on random-init
   models either way.
2. **Python 3.12.** Proposal: bump from 3.10 (EOL). Training on Colab adds a hard constraint: `requires-python`
   must include the Colab runtime's Python (check `!python --version` on Colab). Blocker to verify: `segmentation-models-pytorch==0.3.3` pins
   `timm==0.9.2`; may need smp ≥ 0.3.4. Checkpoint compatibility must be tested.
3. Normalization divides by the per-slice max, so absolute intensity is lost in the output. Acceptable for this
   project? (Affects the API response: we return normalized images plus the scale factor as an attribute.)
