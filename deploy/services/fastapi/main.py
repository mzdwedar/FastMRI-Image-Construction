"""FastAPI serving for the U-Net reconstruction model (PyTorch backend).

Endpoints
- GET  /health    liveness/readiness
- POST /infer     preprocessed slices as raw float32 bytes (N x 1 x 320 x 320), same
                  contract as the Triton model, so the two backends can be load-tested alike
- POST /predict   one FastMRI .h5 file -> .h5 with the reconstructed volume
- POST /evaluate  .h5 files with ground truth -> PSNR / SSIM / NMSE
- GET  /metrics   Prometheus

Run:  CHECKPOINT_PATH=artifacts/model.ckpt uvicorn deploy.services.fastapi.main:app --port 8000
"""

import argparse
import logging
import os
import shutil
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path

import h5py
import numpy as np
import uvicorn
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from prometheus_fastapi_instrumentator import Instrumentator
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool
from torch.utils.data import DataLoader

from deploy.services.fastapi.model_deployment import ModelDeployment
from monitoring.monitor_metrics import HardwareMonitor, InferenceMonitor
from fastmri_knee.data import FastMRICustomDataset, FastMRITransform

logger = logging.getLogger("fastmri.serve")

IMAGE_SHAPE = (1, 320, 320)
SLICE_BYTES = int(np.prod(IMAGE_SHAPE)) * 4  # float32
MAX_SLICES_PER_REQUEST = 64

inference_monitor = InferenceMonitor()
hardware_monitor = HardwareMonitor()


@asynccontextmanager
async def lifespan(app: FastAPI):
    checkpoint_path = os.environ.get("CHECKPOINT_PATH")
    if not checkpoint_path:
        raise RuntimeError("Set CHECKPOINT_PATH to a trained Lightning checkpoint (.ckpt)")
    num_threads = int(os.environ["NUM_THREADS"]) if os.environ.get("NUM_THREADS") else None
    app.state.model = ModelDeployment(checkpoint_path, num_threads=num_threads)
    hardware_monitor.start()
    logger.info("Model loaded from %s", checkpoint_path)
    yield
    hardware_monitor.stop()


app = FastAPI(title="fastmri-knee", description="U-Net MRI reconstruction", version="0.2", lifespan=lifespan)


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    # Full traceback goes to the server log, never to the client
    logger.exception("Unhandled error on %s", request.url.path)
    return JSONResponse(status_code=500, content={"error": "internal server error"})


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/infer")
async def infer(request: Request):
    body = await request.body()
    if not body or len(body) % SLICE_BYTES:
        raise HTTPException(400, f"body must be float32 slices of shape {IMAGE_SHAPE} ({SLICE_BYTES} bytes each)")
    n = len(body) // SLICE_BYTES
    if n > MAX_SLICES_PER_REQUEST:
        raise HTTPException(413, f"at most {MAX_SLICES_PER_REQUEST} slices per request")
    images = np.frombuffer(body, dtype=np.float32).reshape(n, *IMAGE_SHAPE)

    inference_monitor.requests.labels("infer").inc()
    start = time.perf_counter()
    try:
        # Off the event loop: a CPU forward pass would otherwise block every other request
        out = await run_in_threadpool(app.state.model.infer, images)
    except Exception:
        inference_monitor.failures.labels("infer").inc()
        raise
    inference_monitor.latency.labels("infer").observe(time.perf_counter() - start)
    inference_monitor.batch_size.labels("infer").observe(n)
    inference_monitor.slices.labels("infer").inc(n)
    return Response(content=out.astype(np.float32).tobytes(), media_type="application/octet-stream")


def _save_uploads(files: list[UploadFile], directory: str) -> None:
    for f in files:
        name = os.path.basename(f.filename or "")
        if not name.endswith(".h5"):
            raise HTTPException(400, f"{name or 'upload'} is not an .h5 file")
        with open(os.path.join(directory, name), "wb") as out:
            shutil.copyfileobj(f.file, out)


def _loader(path: str) -> DataLoader:
    try:
        dataset = FastMRICustomDataset(path, transform=FastMRITransform(mask_func=None))
    except (KeyError, OSError, ValueError) as exc:
        raise HTTPException(400, f"not a FastMRI single-coil file: {exc}") from exc
    return DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)


# Sync handlers: FastAPI runs them in its threadpool, so file I/O and inference don't block the loop
@app.post("/predict")
def predict(file: UploadFile = File(...)):
    inference_monitor.requests.labels("predict").inc()
    workdir = tempfile.mkdtemp(prefix="predict-")
    try:
        _save_uploads([file], workdir)
        name = os.path.basename(file.filename)
        loader = _loader(os.path.join(workdir, name))
        start = time.perf_counter()
        volumes = app.state.model.predict_volumes(loader)
        inference_monitor.latency.labels("predict").observe(time.perf_counter() - start)
        volume = next(iter(volumes.values()))
        inference_monitor.slices.labels("predict").inc(len(volume))
        out_path = Path(workdir) / name.replace(".h5", "_pred.h5")
        with h5py.File(out_path, "w") as f:
            f.create_dataset("pred", data=volume)
    except Exception as exc:
        if not isinstance(exc, HTTPException):
            inference_monitor.failures.labels("predict").inc()
        shutil.rmtree(workdir, ignore_errors=True)
        raise
    return FileResponse(
        out_path,
        media_type="application/x-hdf5",
        filename=out_path.name,
        background=BackgroundTask(shutil.rmtree, workdir, ignore_errors=True),
    )


@app.post("/evaluate")
def evaluate(files: list[UploadFile] = File(...)):
    inference_monitor.requests.labels("evaluate").inc()
    workdir = tempfile.mkdtemp(prefix="evaluate-")
    try:
        _save_uploads(files, workdir)
        start = time.perf_counter()
        metrics = app.state.model.evaluate(_loader(workdir))
        inference_monitor.latency.labels("evaluate").observe(time.perf_counter() - start)
        return {"metrics": metrics}
    except Exception as exc:
        if not isinstance(exc, HTTPException):
            inference_monitor.failures.labels("evaluate").inc()
        raise
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


Instrumentator(excluded_handlers=["/metrics", "/health"]).instrument(app).expose(app, include_in_schema=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Serve the U-Net with FastAPI")
    parser.add_argument("--checkpoint_path", default=os.environ.get("CHECKPOINT_PATH"))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if args.checkpoint_path:
        os.environ["CHECKPOINT_PATH"] = args.checkpoint_path
    uvicorn.run("deploy.services.fastapi.main:app", host=args.host, port=args.port)
