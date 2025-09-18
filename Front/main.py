import os, json
from typing import Dict, Any, List
import importlib.util
from fastapi import FastAPI, HTTPException, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import numpy as np
import pandas as pd

FRONT_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(FRONT_DIR, ".."))

# Ruta al script con guion
LOGREG_PATH = os.path.join(ROOT_DIR, "modelo_reg_logi", "logistic-regression.py")
CSV_PATH    = os.path.join(ROOT_DIR, "modelo_reg_logi", "Estudio_de_Factores_Asociados_al_Ingreso.csv")
META_PATH   = os.path.join(ROOT_DIR, "modelo_reg_logi", "artifacts_adult_income", "model_metadata.json")

app = FastAPI(title="Clasificación (Regresión Logística) — Adult Income")
templates = Jinja2Templates(directory=os.path.join(FRONT_DIR, "Templates"))

class PredictRequest(BaseModel):
    data: Dict[str, Any]

def load_module_from_path(path: str):
    if not os.path.exists(path):
        raise RuntimeError(f"No encuentro el módulo: {path}")
    spec = importlib.util.spec_from_file_location("logreg_module", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # al importar, ENTRENARÁ y dejará final_model, etc.
    return mod

# ---------- Carga del módulo de regresión logística ----------
mod = load_module_from_path(LOGREG_PATH)

# Columnas finales usadas por el modelo
COLS_KEEP: List[str] = list(mod.to_keep)

# Deriva listas num/cat restringidas a to_keep
NUM_FEATS = [c for c in getattr(mod, "num_cols", []) if c in COLS_KEEP]
CAT_FEATS = [c for c in getattr(mod, "cat_cols", []) if c in COLS_KEEP]

# Umbral óptimo y target
BEST_TH = float(getattr(mod, "best_th", 0.5))
TARGET  = getattr(mod, "TARGET_COL", "target")

# Muestras para selects del front (solo para UX)
def cat_samples(csv_path: str, cat_cols: List[str], limit=12):
    try:
        df = pd.read_csv(csv_path, encoding="utf-8", on_bad_lines="skip")
    except Exception:
        try:
            df = pd.read_csv(csv_path, encoding="cp1252", on_bad_lines="skip")
        except Exception:
            return {}
    out = {}
    for c in cat_cols:
        if c in df.columns:
            vals = df[c].dropna().astype(str).unique().tolist()[:limit]
            out[c] = vals
    return out

CAT_SAMPLES = cat_samples(CSV_PATH, CAT_FEATS)

# Métricas del JSON (si existe)
METRICS = {}
if os.path.exists(META_PATH):
    with open(META_PATH, "r", encoding="utf-8") as f:
        m = json.load(f)
        hold = m.get("holdout_metrics", {})
        METRICS = {
            "accuracy": hold.get("accuracy"),
            "precision": hold.get("precision"),
            "recall": hold.get("recall"),
            "f1": hold.get("f1"),
            "roc_auc": hold.get("roc_auc"),
            "best_threshold": m.get("best_threshold_cv_f1"),
            "csv": os.path.basename(CSV_PATH),
        }

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "title": "Clasificador — Adult Income"})

@app.get("/schema")
def schema():
    return {
        "task": "classification",
        "target": TARGET,
        "numeric_features": NUM_FEATS,
        "categorical_features": CAT_FEATS,
        "categorical_samples": CAT_SAMPLES,
        "metrics": METRICS,
        "threshold": BEST_TH,
    }

@app.get("/metrics")
def metrics():
    return {"metrics": METRICS, "threshold": BEST_TH, "target": TARGET}

@app.post("/predict")
def predict(req: PredictRequest):
    try:
        # Construye una fila con TODAS las columnas esperadas por el modelo
        row = {c: req.data.get(c, None) for c in COLS_KEEP}
        Xnew = pd.DataFrame([row])
        proba = float(mod.final_model.predict_proba(Xnew)[:, 1][0])
        yhat  = int(proba >= BEST_TH)
        return {"target": TARGET, "proba_1": proba, "threshold": BEST_TH, "pred_class": yhat}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
