from typing import Dict, Any
from fastapi import FastAPI
from pydantic import BaseModel

from .infer import load_model, predict_one

app = FastAPI(title="Credit Risk API")

_MODEL, _META, _DEVICE = load_model()

class CreditPayload(BaseModel):
    data: Dict[str, Any]

@app.get("/health")
def health():
    return {
        "ok": True,
        "model_type": _META["model_type"],
        "n_features": len(_META["feature_names"]),
        "threshold": _META["threshold"]
    }

@app.post("/predict")
def predict(p: CreditPayload): 
    return predict_one(p.data, _MODEL, _META, _DEVICE)