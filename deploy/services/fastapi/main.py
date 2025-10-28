import os
import json
import argparse
import asyncio
import logging
import tempfile
import traceback
from time import perf_counter
from pathlib import Path
from collections import defaultdict
from contextlib import asynccontextmanager
from http import HTTPStatus
import shutil

import h5py
import numpy as np
import uvicorn
from torch.utils.data import DataLoader
from fastapi import FastAPI
from fastapi.responses import FileResponse, Response
from prometheus_fastapi_instrumentator import Instrumentator

from deploy.services.fastapi.model_deployment import ModelDeployment
from deploy.services.fastapi.schemas.inputs import EvaluateRequest, PredictRequest
from src.data import FastMRITransform, FastMRICustomDataset
from monitoring.monitor_metrics import hardwareMonitor, inferenceMonitor


inference_monitor = inferenceMonitor()
hardware_monitor = hardwareMonitor()


@asynccontextmanager
async def lifespan(app: FastAPI):
    parser = argparse.ArgumentParser(description="Serve a model with FastAPI")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the model checkpoint file.")
    parser.add_argument("--run_id", help="run ID to use for serving.")
    args, _ = parser.parse_known_args()

    app.state.model = ModelDeployment(checkpoint_path=args.checkpoint_path)
    app.state.hardware_task = asyncio.create_task(hardware_monitor.sample())

    logging.info("Application startup complete.")
    yield

    app.state.hardware_task.cancel()
    logging.info("Cleanup done, shutting down")


app = FastAPI(
    title="model serving",
    description="Serve a model with FastAPI",
    version="0.1",
    lifespan=lifespan
)

@app.get("/health/")
def health():
    return Response(
            content=HTTPStatus.OK.phrase,
            status_code=HTTPStatus.OK,
    )

@app.post("/predict/")
async def predict(request: PredictRequest):

    file = request.file
    fname = os.path.basename(file.filename)
    if not fname.endswith('.h5'):
        return Response(
            content=json.dumps({"error": "Invalid file type"}),
            status_code=HTTPStatus.BAD_REQUEST,
            media_type="application/json"
        )


    model = app.state.model
    inference_monitor.increment_predict_count()
    start = perf_counter()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as temp_file:
        content = await file.read()
        temp_file.write(content)
        temp_file_path = temp_file.name

    try:
        dataset = FastMRICustomDataset(temp_file_path, transform=FastMRITransform(mask_func=None))
        data_loader = DataLoader(dataset, batch_size=8,
                                shuffle=False,
                                num_workers=min(4, os.cpu_count()),
                                pin_memory=True,
                                persistent_workers=True)
        
        output = await model.predict(data_loader)
        
        pred_by_file = defaultdict(list)
        for out_slice in output:
            full_path_fname = out_slice['fname'][0]
            pred_by_file[full_path_fname].append((out_slice['slice_num'].item(), out_slice['pred'].cpu().numpy()))


        for full_path_fname, slices in pred_by_file.items():
            slices.sort(key=lambda x: x[0])  # Sort by slice number
            volume = np.stack([s[1] for s in slices], axis=0)
            fname = os.path.basename(full_path_fname)
            out_fname = fname.replace('.h5', '_pred.h5')

            out_path = Path('./results') / out_fname

            with h5py.File(out_path, 'w') as f:
                f.create_dataset('pred', data=volume)
        
        original_filename = os.path.basename(file.filename)

        duration = perf_counter() - start
        inference_monitor.observe_latency(duration=duration)
        inference_monitor.observe_throughput(num_samples=len(dataset), duration=duration)

        return FileResponse(
            path=out_path,
            media_type="application/x-hdf5",
            filename=original_filename,
            headers={"X-Filename": original_filename}
        )
    
    except FileNotFoundError:
        return Response(
            content=json.dumps({"error": "File not found", "traceback": traceback.format_exc()}),
            status_code=HTTPStatus.NOT_FOUND,
            media_type="application/json"
        )
    except ValueError:
        return Response(
            content=json.dumps({"error": "Value error", "traceback": traceback.format_exc()}),
            status_code=HTTPStatus.BAD_REQUEST,
            media_type="application/json"
        )
    except RuntimeError:
        return Response(
            content=json.dumps({"error": "Runtime error", "traceback": traceback.format_exc()}),
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            media_type="application/json"
        )
    except Exception:
        
        return Response(
            content=json.dumps({"error": "Unexpected error", "traceback": traceback.format_exc()}),
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            media_type="application/json"
        )
        
    finally:
        temp_file.close()
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

@app.post("/evaluate/")
async def evaluate(request: EvaluateRequest):
    model = app.state.model
    inference_monitor.increment_evaluate_count()

    files = request.files
    temp_files = []
    
    try:
        temp_dir = tempfile.mkdtemp()
        for file in files:
            if not file.filename.endswith('.h5'):
                raise ValueError(f"File {file.filename} is not an H5 file")
            
            temp_file_path = os.path.join(temp_dir, file.filename)
            with open(temp_file_path, 'wb') as temp_file:
                content = await file.read()
                temp_file.write(content)
            temp_files.append(temp_file_path)
        
        
        dataset = FastMRICustomDataset(temp_dir, transform=FastMRITransform(mask_func=None))
        data_loader = DataLoader(dataset, batch_size=8,
                                shuffle=False,
                                num_workers=min(4, os.cpu_count()),
                                pin_memory=True,
                                persistent_workers=True
                                )
        
        result = await model.evaluate(data_loader)

        return Response(
            content=json.dumps(result),
            status_code=HTTPStatus.OK,
            media_type="application/json",
            headers={"Content-Disposition": "inline; filename=evaluation.json"}
        )
        
    except FileNotFoundError:
        return Response(
            content=json.dumps({"error": "File not found", "traceback": traceback.format_exc()}),
            status_code=HTTPStatus.NOT_FOUND,
            media_type="application/json"
        )
    except ValueError:
        return Response(
            content=json.dumps({"error": "Value error", "traceback": traceback.format_exc()}),
            status_code=HTTPStatus.BAD_REQUEST,
            media_type="application/json"
        )
    except RuntimeError:
        return Response(
            content=json.dumps({"error": "Runtime error", "traceback": traceback.format_exc()}),
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            media_type="application/json"
        )
    except Exception:
        return Response(
            content=json.dumps({"error": "Unexpected error", "traceback": traceback.format_exc()}),
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            media_type="application/json"
        )
    finally:
        
        if 'temp_dir' in locals() and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)

# Expose Prometheus /metrics endpoint and instrument default metrics
Instrumentator().instrument(app).expose(app)


if __name__ == "__main__":
    uvicorn.run("deploy.main:app", host="0.0.0.0", port=8000, reload=False)
