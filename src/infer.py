from typing import Any, Dict
import torch
import torch.nn as nn
import pandas as pd
import json
import numpy as np
from pathlib import Path

from .model import LinearCredit, MlpCredit

ARTIFACTS = Path("artifacts")

def load_model():
    meta = json.loads((ARTIFACTS / "meta.json").read_text())
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    ckpt = torch.load(ARTIFACTS / "model.pt", map_location=device)

    d_in = ckpt["d_in"]
    model_type = ckpt["model_type"]

    if model_type == "linear":
        model = LinearCredit(d_in)
    else:
        model = MlpCredit(d_in)

    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()

    return model, meta, device


def predict_one(payload: Dict[str, Any], model, meta, device):
    needed = set(meta["feature_names"])
    got = set(payload.keys())
    missing = sorted(needed - got)
    if missing:
        return {"error": f"Missing fields: {missing}"}

    X_df = pd.DataFrame([payload])[meta["feature_names"]]

    mean  = np.array(meta["scaler_mean"],  dtype="float32").reshape(1, -1)
    scale = np.array(meta["scaler_scale"], dtype="float32").reshape(1, -1)
    scale = np.where(scale == 0, 1e-6, scale)

    Xt = (X_df.values.astype("float32") - mean) / scale
    Xt = torch.from_numpy(Xt).to(device)

    model.eval()
    with torch.no_grad():
        logit = model(Xt)
        p = torch.sigmoid(logit).item()

    return {"prob_bad": float(p), "pred": int(p >= meta["threshold"])}