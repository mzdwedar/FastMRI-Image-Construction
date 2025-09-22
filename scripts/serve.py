'''
PLAN
    - init a the model from MlRUN
    - load the model
    - have an endpoint to process an endpoint
'''
import json
from http import HTTPStatus
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response
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
from typing import List
from .predict import predict_fn, TorchPredictor, get_best_checkpoint
from .evaluate import evaluate_fn
from src.data import FastMRITransform, FastMRICustomDataset


app = FastAPI(
    title="model serving",
    description="Serve a model with FastAPI",
    version="0.1",
)


class PredictRequest(BaseModel):
    file: UploadFile = File(...)


class EvaluateRequest(BaseModel):
    files: List[UploadFile] = File(...)



class mydeployment:
    def __init__(self, run_id: str):
        
        trainer_cfg = OmegaConf.load("../configs/trainer.yaml")
        self.cfg = OmegaConf.create({"trainer": trainer_cfg.trainer})
        self.trainer = Trainer(accelerator=self.cfg.trainer.accelerator,
                      devices=self.cfg.trainer.gpus
                    )
        mlflow.set_tracking_uri(self.cfg.trainer.MLFLOW_TRACKING_URI)

        
        best_checkpoint = get_best_checkpoint(run_id=run_id)
        self.predictor = TorchPredictor.from_checkpoint(best_checkpoint)

    async def predict(self, data_loader: DataLoader):
        
        predictions = predict_fn(self.trainer, data_loader, self.predictor)

        return predictions

    async def evaluate(self, dataloader: DataLoader):
        metrics = evaluate_fn(self.trainer, self.predictor, dataloader)

        return metrics


model: mydeployment | None = None


@app.get("/health/")
def health():
    return Response(
            content=HTTPStatus.OK.phrase,
            staus_code=HTTPStatus.OK,
    )

@app.post("/predict/")
async def predict(request: PredictRequest):
    file = request.file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as temp_file:
        content = await file.read()
        temp_file.write(content)
        temp_file_path = temp_file.name

    try:
        dataset = FastMRICustomDataset(temp_file_path, transform=FastMRITransform(mask_func=None))
        data_loader = DataLoader(dataset, shuffle=False)
        image_array = await model.predict(data_loader)


        image = Image.fromarray(image_array)
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        return Response(
            content=img_byte_arr.getvalue(),
            media_type="image/png",
            headers={"Content-Disposition": "inline; filename=prediction.png"}
        )
    
    except FileNotFoundError as e:
        return Response(
            content=json.dumps({"error": "File not found", "traceback": traceback.format_exc()}),
            status_code=HTTPStatus.NOT_FOUND,
            media_type="application/json"
        )
    except ValueError as e:
        return Response(
            content=json.dumps({"error": "Value error", "traceback": traceback.format_exc()}),
            status_code=HTTPStatus.BAD_REQUEST,
            media_type="application/json"
        )
    except RuntimeError as e:
        return Response(
            content=json.dumps({"error": "Runtime error", "traceback": traceback.format_exc()}),
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            media_type="application/json"
        )
    except Exception as e:
        
        return Response(
            content=json.dumps({"error": "Unexpected error", "traceback": traceback.format_exc()}),
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            media_type="application/json"
        )
        
    finally:
        os.unlink(temp_file_path)


@app.post("/evaluate/")
async def evaluate(request: EvaluateRequest):
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
        
    except FileNotFoundError as e:
        return Response(
            content=json.dumps({"error": "File not found", "traceback": traceback.format_exc()}),
            status_code=HTTPStatus.NOT_FOUND,
            media_type="application/json"
        )
    except ValueError as e:
        return Response(
            content=json.dumps({"error": "Value error", "traceback": traceback.format_exc()}),
            status_code=HTTPStatus.BAD_REQUEST,
            media_type="application/json"
        )
    except RuntimeError as e:
        return Response(
            content=json.dumps({"error": "Runtime error", "traceback": traceback.format_exc()}),
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            media_type="application/json"
        )
    except Exception as e:
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


@app.on_event("startup")
def startup_event():
    print("Starting up...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Serve a model with FastAPI")
    parser.add_argument("--run_id", help="run ID to use for serving.")
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host to run the server on."
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port to run the server on."
    )
    args = parser.parse_args()

    model = mydeployment(run_id=args.run_id)
    uvicorn.run(app, host=args.host, port=args.port)
