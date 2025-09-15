from typing import Dict, Any
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Credit Risk API")

@app.get("/health")
def health():
    return {
        "ok": True,
        "model_type": ["model_type"]
    }

@app.post("/predict")
def predict():
    return 100