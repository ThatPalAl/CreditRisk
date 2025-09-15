import torch
import torch.nn as nn
import pandas as pd
import json
from pathlib import Path

from model import LinearCredit, MlpCredit

ARTIFACTS = Path("artifacts")

def load_model():
    meta = json.loads((ARTIFACTS / "meta.json").read_text())
    ckpt = torch.load(ARTIFACTS / "model.pt", map_location="cpu")

    d_in = ckpt["d_in"]
    model_type = ckpt["model_type"]

    if model_type == "linear":
        model = LinearCredit(d_in)
    else:
        model = MlpCredit(d_in)

    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, meta