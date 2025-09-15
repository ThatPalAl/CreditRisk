import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset

from .model import MlpCredit, LinearCredit

''' CONFIG '''
DATA_PATH     = "data/german_credit_data/credit_data_validated.csv"
TARGET_COL    = "target"
MODEL_NAME    = "mlp"
BATCH_SIZE    = 128
EPOCHS        = 100
LR            = 1e-3
SEED          = 42
TEST_SIZE     = 0.2
THRESHOLD     = 0.5
ARTIFACTS_DIR = "artifacts"

class TabularCSVDataset(Dataset):
    def __init__(self, csv_path: str, target_col: str = "target", feature_cols = None):
        df = pd.read_csv(csv_path)
        assert target_col in df.columns, f"Dependend variable missing '{target_col}'"

        y = df[target_col].astype("float32").values
        X_df = df.drop(columns=[target_col])
        if feature_cols is not None:
            missing = set(feature_cols) - set(X_df.columns)
            assert not missing, f"missing columns: {missing}"
            X_df = X_df[feature_cols]

        self.feature_names = X_df.columns.tolist()
        X = X_df.values.astype("float32")

        self.X = X
        self.y = y
        self.n = len(X)

    def __len__(self): return self.n
    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx])
        y = torch.tensor([self.y[idx]], dtype=torch.float32)
        return x, y

def evaluate(model, device, loader, threshold=0.5):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            proba = torch.sigmoid(logits).cpu().numpy().ravel()
            ps.append(proba)
            ys.append(yb.cpu().numpy().ravel())
    y_true = np.concatenate(ys)
    y_prob = np.concatenate(ps)
    auc = roc_auc_score(y_true, y_prob)
    acc = accuracy_score(y_true, (y_prob >= threshold).astype(int))
    return auc, acc

def build_model(name: str, d_in: int):
    if name == "linear": return LinearCredit(d_in)
    if name == "mlp":    return MlpCredit(d_in)
    raise ValueError("model must be 'linear' or 'mlp'")

def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # 1) loading dataset
    full_ds = TabularCSVDataset(csv_path=DATA_PATH, target_col=TARGET_COL, feature_cols=None)

    # 2) Stratified index split - creating subsets
    df = pd.read_csv(DATA_PATH)
    y_full = df[TARGET_COL].astype(int).values
    idx_all = np.arange(len(df))
    idx_tr, idx_va = train_test_split(idx_all, test_size=TEST_SIZE, stratify=y_full, random_state=SEED, shuffle=True)
    train_ds = Subset(full_ds, indices=idx_tr)
    valid_ds = Subset(full_ds, indices=idx_va)

    # 3) Data standardization
    X_full = full_ds.X
    X_tr = X_full[idx_tr]
    X_va = X_full[idx_va]

    scaler = StandardScaler()
    X_tr_std = scaler.fit_transform(X_tr).astype("float32")
    X_va_std = scaler.transform(X_va).astype("float32")

    full_ds.X[idx_tr] = X_tr_std
    full_ds.X[idx_va] = X_va_std

    # 4) DataLoaders
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False)

    # 5) Model and device (mps for mac (silicon) - as cpu returns false)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    d_in = full_ds.X.shape[1]
    model = build_model(MODEL_NAME, d_in).to(device)

    crit = nn.BCEWithLogitsLoss()
    opt  = torch.optim.Adam(model.parameters(), lr=LR)

    best_auc, best_state = -1.0, None

    # 6) Train:
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()

        auc, acc = evaluate(model, device, valid_loader, THRESHOLD)
        print(f"epoch{epoch:02d} | val_auc={auc:.3f} | val_acc={acc:.3f}")
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    # 7) sacing artifcats
    artifacts = Path(ARTIFACTS_DIR); artifacts.mkdir(parents=True, exist_ok=True)
    
    #a) model 
    torch.save({"state_dict": best_state, "d_in": d_in, "model_type": MODEL_NAME}, artifacts / "model.pt")

    #b) meta data
    meta = {
        "feature_names": full_ds.feature_names,
        "scaler_mean": scaler.mean_.ravel().tolist(),
        "scaler_scale": scaler.scale_.ravel().tolist(),
        "threshold": THRESHOLD,
        "target_col": TARGET_COL,
        "model_type": MODEL_NAME,
        "data_path": str(DATA_PATH),
    }
    (artifacts / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Saved {artifacts} (best AUC with a score: {best_auc:.3f}) using model :{MODEL_NAME}")

if __name__ == "__main__":
    main()
    print("train.py finished succesfully")