from typing import Dict, Any
from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
import json

from .infer import load_model, predict_one

app = FastAPI(title="Credit Risk API")

_MODEL, _META, _DEVICE = load_model()

ARTIFACTS = Path("artifacts")

class CreditPayload(BaseModel):
    data: Dict[str, Any]

@app.get("/health")
def health():
    return {
        "ok": True,
        "model_type": _META["model_type"],
        "n_features": len(_META["feature_names"]),
        "threshold": _META["threshold"],
    }

@app.get("/schema")
def schema():
    return {
        "features:":_META['feature_names'],
        "target": _META.get("target_col", "target"),
        "notes": "Send /predict with {'data': {feature: value, ...}}"
    }

@app.post("/predict")
def predict(p: CreditPayload): 
    return predict_one(p.data, _MODEL, _META, _DEVICE)

@app.get("/metrics")
def metrics():
    meta = json.loads((ARTIFACTS / "meta.json").read_text())
    mpath = ARTIFACTS / "metrics.json"
    body = {"meta": meta}
    if mpath.exists():
        body["validation"] = json.loads(mpath.read_text())
    return body