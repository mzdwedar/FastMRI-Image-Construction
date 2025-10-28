from typing import List
from pydantic import BaseModel
from fastapi import UploadFile, File


class PredictRequest(BaseModel):
    file: UploadFile = File(...)


class EvaluateRequest(BaseModel):
    files: List[UploadFile] = File(...)
