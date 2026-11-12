# Training on Colab

`notebooks/train_colab.ipynb` trains the U-Net on a Colab GPU. You open it in VS Code with the Google Colab
extension (Command Palette > "Google Colab: Sign In", then pick a Colab GPU runtime in the kernel picker).

The kernel is remote: it cannot see local files or uncommitted changes. The notebook installs the package from a
**pushed commit** and writes all outputs to Drive.

## Steps

1. Push the commit you want to train: `git push`, then `git rev-parse HEAD`.
2. Put the data on Drive as `MyDrive/fastmri/{train,val}/*.h5` (singlecoil knee).
3. Open `notebooks/train_colab.ipynb`, pick a GPU kernel, set the parameter cell (`GIT_SHA`, `RUN_NAME`, ...), run all cells.
4. Results are in `MyDrive/fastmri/runs/<RUN_NAME>/`:
   - `checkpoints/best.ckpt` (best by PSNR), `checkpoints/last.ckpt`, `checkpoints/run_summary.json`
   - `mlruns/` (MLflow file store: curves, open it with `mlflow ui --backend-store-uri <that folder>` after downloading)
   - `train_results.json`
5. Download the run folder and register it locally (K3): `make register CKPT=<checkpoint> SUMMARY=<run_summary.json>`.

## Smoke run vs real run

The defaults are a smoke run: 2 files per split, `NUM_SAMPLES=64` slices, 1 epoch. For a real run set `NUM_FILES = 0`
(all files copied), `NUM_SAMPLES = -1` and more `EPOCHS`.

## Disconnects

Run the training cell again with the same `RUN_NAME`: it resumes from `checkpoints/last.ckpt` on Drive. Resume
is exact at epoch boundaries (the checkpoint is saved after each validation). A run stopped by `max_steps`
does not continue on rerun, so control run length with `EPOCHS`.

## Data copy

Drive's FUSE mount is slow for random HDF5 reads, so the notebook copies the files to `/content/data` first. The
full singlecoil knee train set is ~88 GB and does not fit comfortably: use a subset (`NUM_FILES`) until Q18
(a committed subset manifest) is decided.

## Experiment tracking (Q17)

Colab cannot reach the local Compose MLflow. Runs log to an MLflow file store on Drive, and registration stays a
local, gated step through `run_summary.json` (checkpoint sha256 and `preprocess_version` are checked).

## Drive fallback

Status: **not yet verified** that `google.colab.drive.mount` works under the VS Code kernel (K2 acceptance is a manual
run). If the mount cell fails, options in order: re-run after signing in again; mount from a browser Colab
session instead (same notebook); or use `google.colab.auth` with the Drive API / `rclone` to copy files. Record
what worked here.

## Troubleshooting

- `not found on GitHub`: the commit is not pushed, or the repo is private (pip needs access to it).
- pip fails on Python: Colab ships Python 3.13 but the package needs >=3.10,<3.13, so the notebook installs into a uv-managed 3.12 venv at `/content/venv` and trains with that interpreter.
- No GPU: Runtime > Change runtime type. Training still runs on CPU but is far too slow beyond the smoke run.
