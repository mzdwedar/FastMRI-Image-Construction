import json
from http import HTTPStatus
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, Response
import tempfile
import traceback
from PIL import Image
from pydantic import BaseModel
import mlflow
import argparse
import uvicorn
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
import os
import io
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Gauge
import psutil
import threading
import time
from typing import List
import numpy as np
import h5py
from collections import defaultdict
from pathlib import Path
from prometheus_client import Histogram
from time import perf_counter


from scripts.predict import predict_fn, TorchPredictor, get_best_checkpoint
from scripts.evaluate import evaluate_fn
from src.data import FastMRITransform, FastMRICustomDataset


app = FastAPI(
    title="model serving",
    description="Serve a model with FastAPI",
    version="0.1",
)

# Metrics
predict_requests_total = Counter("predict_requests_total", "Total number of predict requests")
evaluate_requests_total = Counter("evaluate_requests_total", "Total number of evaluate requests")
# Hardware/process metrics
host_cpu_percent = Gauge("host_cpu_percent", "Host CPU percent")
host_mem_percent = Gauge("host_mem_percent", "Host memory percent")
process_rss_bytes = Gauge("process_rss_bytes", "App process RSS bytes")

inference_latency = Histogram(
    "predict_inference_latency_seconds",
    "Time spent in model.predict() for /predict endpoint",
    buckets=(0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10)  # 10ms → 10s
)

def sample_hw_metrics() -> None:
    process = psutil.Process()
    # Prime cpu_percent to avoid initial 0.0
    psutil.cpu_percent(interval=None)
    while True:
        try:
            host_cpu_percent.set(psutil.cpu_percent(interval=None))
            host_mem_percent.set(psutil.virtual_memory().percent)
            process_rss_bytes.set(process.memory_info().rss)
        except Exception:
            # Do not crash the sampler; just continue
            pass
        time.sleep(5)


class PredictRequest(BaseModel):
    file: UploadFile = File(...)


class EvaluateRequest(BaseModel):
    files: List[UploadFile] = File(...)



class ModelDeployment:
    def __init__(self, checkpoint_path:str):
        
        trainer_cfg = OmegaConf.load("./configs/trainer.yaml")
        self.cfg = OmegaConf.create({"trainer": trainer_cfg.trainer})
        self.trainer = Trainer(accelerator=self.cfg.trainer.accelerator,
                      devices=self.cfg.trainer.gpus
                    )
        os.makedirs('results', exist_ok=True)
        # print("MLflow tracking URI:", self.cfg.trainer.MLFLOW_TRACKING_URI)
        # mlflow.set_tracking_uri(self.cfg.trainer.MLFLOW_TRACKING_URI)

        
        # best_checkpoint = get_best_checkpoint(run_id=run_id)
        # best_checkpoint = os.path.join('./mlruns', experiment_id, run_id, 'artifacts/checkpoints')
        self.predictor = TorchPredictor.from_checkpoint(checkpoint_path)

    async def predict(self, data_loader: DataLoader):
        
        predictions = predict_fn(self.trainer, data_loader, self.predictor)

        return predictions

    async def evaluate(self, dataloader: DataLoader):
        metrics = evaluate_fn(self.trainer, self.predictor, dataloader)

        return metrics


model: ModelDeployment | None = None


@app.get("/health/")
def health():
    return Response(
            content=HTTPStatus.OK.phrase,
            status_code=HTTPStatus.OK,
    )

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    predict_requests_total.inc()

    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as temp_file:
        content = await file.read()
        temp_file.write(content)
        temp_file_path = temp_file.name

    try:
        dataset = FastMRICustomDataset(temp_file_path, transform=FastMRITransform(mask_func=None))
        data_loader = DataLoader(dataset, shuffle=False)
         
        start = perf_counter()
        output = await model.predict(data_loader)
        duration = perf_counter() - start
        inference_latency.observe(duration)

        pred_by_file = defaultdict(list)
        for out_slice in output:
            full_path_fname = out_slice['fname'][0]
            pred_by_file[full_path_fname].append((out_slice['slice_num'].item(), out_slice['pred'].cpu().numpy()))

        # print(f"DEBUG: {pred_by_file}")
        print(f"DEBUG: filename: {pred_by_file.keys()}")

        for full_path_fname, slices in pred_by_file.items():
            slices.sort(key=lambda x: x[0])  # Sort by slice number
            volume = np.stack([s[1] for s in slices], axis=0)
            fname = os.path.basename(full_path_fname)
            out_fname = fname.replace('.h5', '_pred.h5')

            out_path = Path('./results') / out_fname

            with h5py.File(out_path, 'w') as f:
                f.create_dataset('pred', data=volume)
        
        original_filename = os.path.basename(file.filename)
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
        os.unlink(temp_file_path)


@app.post("/evaluate/")
async def evaluate(request: EvaluateRequest):
    evaluate_requests_total.inc()
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
        data_loader = DataLoader(dataset, batch_size=8, shuffle=False)
        
        
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
        
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
        
        if 'temp_dir' in locals() and os.path.exists(temp_dir):
            os.rmdir(temp_dir)


# Expose Prometheus /metrics endpoint and instrument default metrics
Instrumentator().instrument(app).expose(app)

# Start background hardware metrics sampler
threading.Thread(target=sample_hw_metrics, daemon=True).start()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Serve a model with FastAPI")
    parser.add_argument("--run_id", help="run ID to use for serving.")
    parser.add_argument(
        "--checkpoint_path", type=str, required=True, help="Path to the model checkpoint file."
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host to run the server on."
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port to run the server on."
    )
    args = parser.parse_args()

    model = ModelDeployment(checkpoint_path=args.checkpoint_path)
    uvicorn.run(app, host=args.host, port=args.port)
