# main.py
import os
from typing import Dict
from importlib.util import spec_from_file_location, module_from_spec

from fastapi import FastAPI, Request, Form
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.cors import CORSMiddleware

# Archivos esperados
CSV_PATH = "Estudio_de_Factores_Asociados_al_Ingreso.csv"
LR_FILE_HYPHEN = "logistic-regression.py"
LR_FILE_UNDERSCORE = "logistic_regression.py"  # respaldo si lo renombras

app = FastAPI(title="Ingreso > $25,000 — Mini API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# templates
templates = Jinja2Templates(directory="templates")

# Estado global simple
LR_MOD = None
PIPE = None
META = None
ERROR_MSG = None

# -------------------------
# Utilidades de carga/ML
# -------------------------
def _load_lr_module():
    """Carga logistic-regression.py aunque tenga guion medio."""
    global LR_MOD
    if os.path.exists(LR_FILE_HYPHEN):
        spec = spec_from_file_location("lr_mod", LR_FILE_HYPHEN)
        mod = module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        LR_MOD = mod
    elif os.path.exists(LR_FILE_UNDERSCORE):
        import importlib
        LR_MOD = importlib.import_module("logistic_regression")
    else:
        raise FileNotFoundError("No encontré logistic-regression.py")

def _train_safe():
    """Entrena el modelo de forma segura (sin tumbar la app)."""
    global PIPE, META, ERROR_MSG
    try:
        if LR_MOD is None:
            _load_lr_module()
        if not os.path.exists(CSV_PATH):
            raise FileNotFoundError(f"No encontré el CSV: {CSV_PATH}")
        PIPE, META = LR_MOD.train_from_csv(CSV_PATH)
        ERROR_MSG = None
        print(f"[OK] Modelo entrenado. Acc (train): {META.get('training_accuracy'):.3f}")
    except Exception as e:
        PIPE, META = None, None
        ERROR_MSG = str(e)
        print("[ERROR] Entrenamiento falló:", ERROR_MSG)

@app.on_event("startup")
def _on_startup():
    _train_safe()

# -------------------------
# Rutas
# -------------------------
@app.get("/")
def landing(request: Request):
    if ERROR_MSG:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": ERROR_MSG,
            "meta": None,
            "features": []
        })
    return templates.TemplateResponse("index.html", {
        "request": request,
        "error": None,
        "meta": META,
        "features": META["features"],
        "cat_choices": META.get("cat_choices", {})
    })

@app.post("/predict")
async def predict(request: Request):
    if PIPE is None or META is None or ERROR_MSG:
        return RedirectResponse("/", status_code=303)
    form = await request.form()
    payload: Dict[str, str] = {k: str(v) for k, v in form.items()}
    pred, proba = LR_MOD.predict_one(PIPE, META, payload)
    label = "Sí" if pred == 1 else "No"
    return templates.TemplateResponse("index.html", {
        "request": request,
        "error": None,
        "meta": META,
        "features": META["features"],
        "cat_choices": META.get("cat_choices", {}),
        "result": {"label": label, "proba": f"{proba:.3f}"}
    })

@app.get("/retrain")
def retrain():
    _train_safe()
    if ERROR_MSG:
        return {"status": "error", "message": ERROR_MSG}
    return {"status": "ok", "training_accuracy": META.get("training_accuracy")}

@app.post("/api/predict")
async def api_predict(request: Request):
    if PIPE is None or META is None or ERROR_MSG:
        return JSONResponse({"error": ERROR_MSG or "Modelo no entrenado."}, status_code=503)
    payload = await request.json()
    pred, proba = LR_MOD.predict_one(PIPE, META, payload)
    return {"pred": int(pred), "proba": proba}
