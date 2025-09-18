# Front/main.py
import os
import json
import importlib.util
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel  # <-- IMPORT QUE TE FALTABA
from fastapi.staticfiles import StaticFiles


# ---------------------- Paths según tu estructura ----------------------
FRONT_DIR = os.path.dirname(__file__)
ROOT_DIR  = os.path.abspath(os.path.join(FRONT_DIR, ".."))

# ✅ Rutas correctas: salen de Front/ y entran a modelo_reg_logi/
LOGREG_PATH = os.path.join(ROOT_DIR, "modelo_reg_logi", "logistic-regression.py")
CSV_PATH    = os.path.join(ROOT_DIR, "modelo_reg_logi", "Estudio_de_Factores_Asociados_al_Ingreso.csv")
META_PATH   = os.path.join(ROOT_DIR, "modelo_reg_logi", "artifacts_adult_income", "model_metadata.json")

# ---------------------- FastAPI & Templates ----------------------
app = FastAPI(title="Clasificación — Regresión Logística (Adult Income)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

ARTIF_DIR = os.path.join(ROOT_DIR, "modelo_reg_logi", "artifacts_adult_income")

# Servir los PNG como archivos estáticos
app.mount("/artifacts", StaticFiles(directory=ARTIF_DIR), name="artifacts")

# OJO: carpeta "Templates" con T mayúscula
templates = Jinja2Templates(directory=os.path.join(FRONT_DIR, "Templates"))

# PON ESTO ARRIBA, justo después de tus imports estándar (antes de get_module/load):
import os as _os
import matplotlib
matplotlib.use("Agg")          # backend sin GUI
import matplotlib.pyplot as plt
plt.show = lambda *a, **k: None  # evita abrir ventanas

# (opcional) refuerzo vía variable de entorno
_os.environ["MPLBACKEND"] = "Agg"


# ---------------------- Carga perezosa del módulo ----------------------
_mod = None  # cache del módulo ya cargado

def load_module_from_path(path: str):
    if not os.path.exists(path):
        raise RuntimeError(f"No encuentro el módulo: {path}")
    spec = importlib.util.spec_from_file_location("logreg_module", path)
    m = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(m)  # ejecuta logistic-regression.py (entrena/arma pipeline)
    return m

def get_module():
    global _mod
    if _mod is None:
        _mod = load_module_from_path(LOGREG_PATH)
    return _mod

# ---------------------- Utilidades ----------------------
def read_csv_flex(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc, on_bad_lines="skip")
        except Exception:
            continue
    return None

def cat_samples(csv_path: str, cat_cols: List[str], limit: int = 12) -> Dict[str, List[str]]:
    df = read_csv_flex(csv_path)
    if df is None:
        return {}
    out: Dict[str, List[str]] = {}
    for c in cat_cols:
        if c in df.columns:
            vals = df[c].dropna().astype(str).unique().tolist()[:limit]
            out[c] = vals
    return out

def load_metrics_from_meta(meta_path: str) -> Dict[str, Any]:
    if not os.path.exists(meta_path):
        return {}
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            m = json.load(f)
        hold = m.get("holdout_metrics", {}) or {}
        return {
            "accuracy": hold.get("accuracy"),
            "precision": hold.get("precision"),
            "recall": hold.get("recall"),
            "f1": hold.get("f1"),
            "roc_auc": hold.get("roc_auc"),
            "best_threshold": m.get("best_threshold_cv_f1"),
            "csv": os.path.basename(CSV_PATH) if os.path.exists(CSV_PATH) else None,
        }
    except Exception:
        return {}

# ---------------------- Rutas ----------------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    # El index SIEMPRE se sirve, aunque el módulo falle al cargar luego
    return templates.TemplateResponse("index.html", {"request": request, "title": "Clasificador — Adult Income"})

@app.get("/schema")
def schema():
    m = get_module()

    # columnas finales usadas por el modelo
    to_keep: List[str] = list(getattr(m, "to_keep", []))
    num_cols: List[str] = [c for c in getattr(m, "num_cols", []) if c in to_keep]
    cat_cols: List[str] = [c for c in getattr(m, "cat_cols", []) if c in to_keep]

    # muestras categóricas para el front
    cat_samp = cat_samples(CSV_PATH, cat_cols)

    # métricas desde metadata (si existe)
    meta_metrics = load_metrics_from_meta(META_PATH)

    # umbral óptimo: prioriza el del módulo; si no, del metadata; si no, 0.5
    threshold = getattr(m, "best_th", None)
    if threshold is None:
        threshold = meta_metrics.get("best_threshold", 0.5)

    return {
        "task": "classification",
        "target": getattr(m, "TARGET_COL", "target"),
        "numeric_features": num_cols,
        "categorical_features": cat_cols,
        "categorical_samples": cat_samp,
        "metrics": {
            "accuracy": meta_metrics.get("accuracy"),
            "precision": meta_metrics.get("precision"),
            "recall": meta_metrics.get("recall"),
            "f1": meta_metrics.get("f1"),
            "roc_auc": meta_metrics.get("roc_auc"),
            "csv": meta_metrics.get("csv"),
        },
        "threshold": threshold,
    }

@app.get("/metrics")
def metrics():
    m = get_module()
    meta_metrics = load_metrics_from_meta(META_PATH)
    threshold = getattr(m, "best_th", None)
    if threshold is None:
        threshold = meta_metrics.get("best_threshold", 0.5)
    return {
        "target": getattr(m, "TARGET_COL", "target"),
        "metrics": meta_metrics,
        "threshold": threshold,
    }

class PredictRequest(BaseModel):
    data: Dict[str, Any]

@app.get("/figs")
def list_figs():
    if not os.path.exists(ARTIF_DIR):
        return {"images": []}
    exts = {".png", ".jpg", ".jpeg", ".webp", ".svg"}
    files = []
    for f in os.listdir(ARTIF_DIR):
        if os.path.splitext(f)[1].lower() in exts:
            # URL pública que podrás usar en <img src="...">
            files.append(f"/artifacts/{f}")
    # ordena para que se vean primero las más útiles
    order_hint = ["holdout_cm", "holdout_pr", "holdout_roc", "cv_auc_por_fold", "cv_f1_por_fold"]
    files.sort(key=lambda p: next((i for i, k in enumerate(order_hint) if k in p), 999))
    return {"images": files}


@app.post("/predict")
def predict(req: PredictRequest):
    """
    Recibe: { data: {columna: valor, ...} }
    Devuelve: { target, proba_1, threshold, pred_class }
    """
    try:
        m = get_module()

        # columnas esperadas por el modelo (pipeline final)
        to_keep: List[str] = list(getattr(m, "to_keep", []))
        if not to_keep:
            raise RuntimeError("No se encontraron columnas 'to_keep' en el módulo.")

        # construye DF de una fila con todas las columnas esperadas
        row = {c: req.data.get(c, None) for c in to_keep}
        Xnew = pd.DataFrame([row])

        # usa el pipeline final para predecir probas
        if not hasattr(m, "final_model"):
            raise RuntimeError("El módulo no expone 'final_model'.")
        proba = float(m.final_model.predict_proba(Xnew)[:, 1][0])

        # umbral (best_th) o fallback 0.5
        th = getattr(m, "best_th", 0.5)
        yhat = int(proba >= th)

        return {
            "target": getattr(m, "TARGET_COL", "target"),
            "proba_1": proba,
            "threshold": th,
            "pred_class": yhat
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
