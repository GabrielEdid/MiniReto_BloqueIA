# -*- coding: utf-8 -*-
"""
Regresión Logística con columnas en español, explicada paso a paso.

Objetivo del script
-------------------
Entrenar y evaluar un modelo de regresión logística que predice si el sueldo
mensual es mayor a $25,000. El flujo queda documentado y controlado para que
sea fácil de entender y reproducir.

Decisiones clave y por qué
--------------------------
1) Se conservan nombres de columnas en español.
   Razón: el dataset fue diseñado así y sirve como documentación directa.

2) Se aplica limpieza ligera de texto (quitar acentos, recortar espacios, "?" -> NaN).
   Razón: reduce fallas del one-hot y evita categorías duplicadas por tildes.

3) Se normaliza "Ocupación" a 6 industrias amplias.
   Razón: simplifica categorías raras y aporta señal con grupos más generales.

4) Preprocesamiento con ColumnTransformer:
   - Numéricas: imputación por mediana + estandarización (media 0, var 1).
   - Categóricas: imputación por moda + One-Hot.
   Razón: la logística requiere todo numérico y sin nulos; escalar ayuda al
          optimizador y one-hot convierte texto a números.

5) Poda automática con Permutation Importance (PI) a nivel de columna original:
   - Se mide la caída del AUC al desordenar cada columna.
   - Si el aporte medio es <= 0, la columna se marca como candidata a eliminar.
   Razón: si desordenar no empeora (o mejora), probablemente no aporta.

6) Umbral de decisión óptimo por validación cruzada para F1.
   - Se busca el punto de corte en probabilidad que maximiza F1 promedio.
   Razón: F1 balancea precisión y recall, útil con pocos positivos.

7) Evaluación en holdout y en CV con el umbral fijo encontrado.
   Razón: separa la selección de umbral (CV) de la evaluación final (holdout 20%).

8) Se reportan coeficientes del modelo final.
   Razón: permite interpretar qué variables empujan la predicción.

9) Las gráficas se guardan en PNG y se muestran en pantalla durante la ejecución.
   Razón: deja evidencia visual y facilita la revisión.

Artefactos generados
--------------------
- artifacts_adult_income/logreg_adult_pipeline.joblib   (modelo final)
- artifacts_adult_income/model_metadata.json             (contexto de entrenamiento)
- artifacts_adult_income/*.png                           (gráficas)
"""

import os
import re
import json
import unicodedata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    RocCurveDisplay, PrecisionRecallDisplay, ConfusionMatrixDisplay
)
from sklearn.inspection import permutation_importance
from joblib import dump

# -----------------------------------------------------------------------------
# Configuración básica del experimento
# -----------------------------------------------------------------------------
CSV_FILE = "Estudio_de_Factores_Asociados_al_Ingreso.csv"  # archivo de datos
OUT_DIR = "artifacts_adult_income"                          # carpeta de artefactos
RANDOM_STATE = 42                                           # semilla para reproducibilidad
N_SPLITS = 5                                                # número de folds para CV

# -----------------------------------------------------------------------------
# Utilidades de impresión y limpieza
# -----------------------------------------------------------------------------
def section(title: str):
    """Imprime un bloque de sección para mejorar la legibilidad en terminal."""
    line = "=" * 80
    print(f"\n{line}\n{title}\n{line}")

def kv(k, v):
    """Imprime pares clave:valor con un formato uniforme."""
    print(f"{k:<28}: {v}")

def strip_accents(s: str) -> str:
    """Elimina acentos/diacríticos de un string para uniformar categorías."""
    if not isinstance(s, str):
        s = str(s)
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

def norm_txt(s: str) -> str:
    """Normaliza texto: sin acentos, minúsculas y espacios comprimidos."""
    s = strip_accents(str(s)).lower().strip()
    s = re.sub(r"[\n\r\t]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s

def ensure_outdir(p):
    """Crea la carpeta de artefactos si no existe."""
    os.makedirs(p, exist_ok=True)

def save_and_show(fig, path):
    """
    Guarda la figura en PNG y la muestra en pantalla.
    Todas las gráficas del flujo llaman a esta función.
    """
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.show()

# -----------------------------------------------------------------------------
# Canonización de ocupación a los 6 sectores de la encuesta
# -----------------------------------------------------------------------------
SECTORES_CANONICOS = {
    "Tecnología y servicios profesionales",
    "Industria y manufactura",
    "Comercio y logística",
    "Gobierno y sector público",
    "Salud y servicios sociales",
    "Agro y alimentos",
}

# Mapeo de formas normalizadas -> etiqueta canónica (con acentos)
NORMALIZACION_SECTORES = {
    "tecnologia y servicios profesionales": "Tecnología y servicios profesionales",
    "industria y manufactura": "Industria y manufactura",
    "comercio y logistica": "Comercio y logística",
    "gobierno y sector publico": "Gobierno y sector público",
    "salud y servicios sociales": "Salud y servicios sociales",
    "agro y alimentos": "Agro y alimentos",
}

def canoniza_sector(valor: str) -> str:
    """
    Devuelve la etiqueta canónica si ya coincide; en caso contrario, normaliza
    y busca en el diccionario. Si no hay coincidencia, devuelve 'Otros'.
    """
    if pd.isna(valor):
        return np.nan
    if valor in SECTORES_CANONICOS:
        return valor
    v_norm = norm_txt(valor)
    return NORMALIZACION_SECTORES.get(v_norm, "Otros")

# -----------------------------------------------------------------------------
# 1) Carga de datos con tolerancia de encoding
# -----------------------------------------------------------------------------
csv_path = CSV_FILE if os.path.isabs(CSV_FILE) else os.path.join(os.path.dirname(__file__), CSV_FILE)
if not os.path.exists(csv_path):
    # Si el script se ejecuta desde otra carpeta, se prueba ruta relativa simple.
    alt = CSV_FILE
    if os.path.exists(alt):
        csv_path = alt
    else:
        raise SystemExit(f"No se encontró: {CSV_FILE}")

df = None
enc_used = None
last_err = None
for enc in ("utf-8", "cp1252", "latin-1"):
    # Se prueban varios encodings comunes en CSVs en español.
    try:
        df = pd.read_csv(csv_path, encoding=enc, on_bad_lines="skip")
        enc_used = enc
        last_err = None
        break
    except Exception as e:
        last_err = e
if last_err is not None:
    raise SystemExit(f"Error leyendo CSV: {last_err}")

section("CARGA DE DATOS")
kv("Ruta CSV", csv_path)
kv("Encoding detectado", enc_used)
kv("Filas x Columnas", f"{df.shape[0]} x {df.shape[1]}")

# Limpieza básica: en columnas de texto, se quitan acentos y "?" se considera faltante.
for c in df.columns:
    if df[c].dtype == object:
        df[c] = df[c].astype(str).map(strip_accents).str.strip()
        df[c] = df[c].replace({"?": np.nan})

# -----------------------------------------------------------------------------
# 2) Detección de la columna objetivo y construcción de X, y
# -----------------------------------------------------------------------------
# Búsqueda de una columna cuyo nombre recuerde a sueldo/ingreso para no depender de un nombre exacto.
target_candidates = [c for c in df.columns if "sueldo" in norm_txt(c) or "ingreso" in norm_txt(c) or "income" in norm_txt(c)]
if not target_candidates:
    raise SystemExit("No se encontró la columna objetivo (ej: '¿Tu sueldo mensual es mayor a $25,000?').")
TARGET_COL = target_candidates[0]

def map_target(v: str) -> int:
    """
    Mapea respuestas de la encuesta a 0/1.
    Regla: si contiene 'mayor' o algo que indique '>25', devuelve 1; si no, 0.
    """
    s = norm_txt(v)
    return 1 if ("mayor" in s or ">25" in s or "si" == s or "sí" == s) else 0

y = df[TARGET_COL].map(map_target)
X_full = df.drop(columns=[TARGET_COL])

# Canonización de la columna de sector/ocupación a las 6 clases de la encuesta, si existe.
col_ocu = next((c for c in X_full.columns if "ocup" in norm_txt(c) or "sector" in norm_txt(c)), None)
if col_ocu is not None:
    X_full[col_ocu] = X_full[col_ocu].apply(canoniza_sector)

section("COLUMNAS INICIALES")
kv("Objetivo", TARGET_COL)
kv("Total columnas (X)", X_full.shape[1])

# -----------------------------------------------------------------------------
# 3) Preprocesador (numéricas y categóricas)
# -----------------------------------------------------------------------------
def make_ohe():
    """
    Retorna el OneHotEncoder compatible con distintas versiones de scikit-learn.
    - En versiones nuevas: sparse_output=False.
    - En versiones previas: sparse=False.
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

def build_preprocessor(cols, Xref):
    """
    Separa columnas numéricas y categóricas y arma el ColumnTransformer.
    Regla para numéricas: dtype numérico o nombre que sugiera número ('hora', 'edad').
    """
    num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(Xref[c]) or "hora" in norm_txt(c) or "edad" in norm_txt(c)]
    cat_cols = [c for c in cols if c not in num_cols]

    # Numéricas: imputación por mediana + escalado.
    num_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Categóricas: imputación por moda + One-Hot.
    cat_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("oh", make_ohe())
    ])

    pre = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols)
    ])
    return pre, num_cols, cat_cols

# -----------------------------------------------------------------------------
# 4) Poda automática con Permutation Importance (AUC <= 0)
# -----------------------------------------------------------------------------
section("PODA AUTOMATICA (Permutation Importance)")
cols_now = list(X_full.columns)
pre0, _, _ = build_preprocessor(cols_now, X_full)

# Clasificador base: logística con balance de clases y más iteraciones por estabilidad.
base_clf = LogisticRegression(
    penalty="l2",
    solver="lbfgs",
    class_weight="balanced",
    max_iter=1000,
    random_state=RANDOM_STATE
)
pipe0 = Pipeline([("pre", pre0), ("clf", base_clf)])

# Promedio de importancias sobre K folds para reducir la varianza del estimador.
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
imp_sum = pd.Series(0.0, index=cols_now)
cnt_sum = pd.Series(0, index=cols_now)

for tr, va in skf.split(X_full, y):
    Xtr, Xva = X_full.iloc[tr], X_full.iloc[va]
    ytr, yva = y.iloc[tr], y.iloc[va]
    pipe0.fit(Xtr, ytr)
    pi = permutation_importance(
        pipe0, Xva, yva,
        scoring="roc_auc",
        n_repeats=15,                # más repeticiones -> promedio más estable
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    imp_sum += pd.Series(pi.importances_mean, index=cols_now)
    cnt_sum += 1

# Orden de menor a mayor aporte medio al AUC
importancias = (imp_sum / cnt_sum).sort_values(ascending=True)
print("Importancias medias (AUC) por columna (menor -> mayor):")
for k, v in importancias.items():
    print(f"  - {k:<25} {v: .6f}")

# Columnas a eliminar: aporte medio <= 0.0
to_drop = importancias[importancias <= 0.0].index.tolist()
to_keep = [c for c in cols_now if c not in to_drop]
kv("Columnas eliminadas", to_drop if to_drop else "Ninguna")
kv("Columnas finales usadas", to_keep)

# X filtrada para el resto del flujo (el CSV original no se modifica).
X = X_full[to_keep]

# -----------------------------------------------------------------------------
# 5) Búsqueda de umbral por CV que maximiza F1
# -----------------------------------------------------------------------------
def best_threshold_cv(pipe_model, X, y, cv):
    ths = np.linspace(0.05, 0.95, 19)
    scores = np.zeros_like(ths, dtype=float)
    for tr, va in cv.split(X, y):
        Xtr, Xva = X.iloc[tr], X.iloc[va]
        ytr, yva = y.iloc[tr], y.iloc[va]
        pipe_model.fit(Xtr, ytr)
        p = pipe_model.predict_proba(Xva)[:, 1]
        for i, t in enumerate(ths):
            yhat = (p >= t).astype(int)
            scores[i] += f1_score(yva, yhat, zero_division=0)
    i_best = int(scores.argmax())
    return float(ths[i_best]), {float(ths[i]): float(scores[i] / cv.get_n_splits()) for i in range(len(ths))}

# -----------------------------------------------------------------------------
# 6) Entrenamiento en holdout y reporte de métricas
# -----------------------------------------------------------------------------
pre, num_cols, cat_cols = build_preprocessor(list(X.columns), X)
clf = LogisticRegression(
    penalty="l2",
    solver="lbfgs",
    class_weight="balanced",
    max_iter=1000,
    random_state=RANDOM_STATE
)
pipe = Pipeline([("pre", pre), ("clf", clf)])

# CV para seleccionar el umbral
cv_for_th = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
best_th, _ = best_threshold_cv(Pipeline([("pre", pre), ("clf", clf)]), X, y, cv_for_th)

section("UMBRAL OPTIMO POR CV")
kv("Umbral F1 promedio", round(best_th, 2))

# Separación 80/20 para evaluación final (holdout). Stratify mantiene proporción de clases.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)
pipe.fit(X_train, y_train)

# Probabilidades y predicción con el umbral elegido
y_proba = pipe.predict_proba(X_test)[:, 1]
y_pred  = (y_proba >= best_th).astype(int)

# Métricas de desempeño
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_proba)
cm = confusion_matrix(y_test, y_pred)

section("EVALUACION HOLDOUT (20% TEST)")
kv("Umbral usado", round(best_th, 2))
kv("Accuracy", f"{acc:.4f}")
kv("Precision", f"{prec:.4f}")
kv("Recall", f"{rec:.4f}")
kv("F1-score", f"{f1:.4f}")
kv("ROC-AUC", f"{roc_auc:.4f}")
print("Matriz de confusion [tn fp; fn tp]:")
print(cm)
print("\nReporte de clasificacion:")
print(classification_report(y_test, y_pred, digits=4))

# Carpeta de salida y gráficas
ensure_outdir(OUT_DIR)

# Curva ROC del holdout
fig1, ax1 = plt.subplots()
RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax1, name="LogReg")
save_and_show(fig1, os.path.join(OUT_DIR, "holdout_roc.png"))

# Curva Precision-Recall del holdout
fig2, ax2 = plt.subplots()
PrecisionRecallDisplay.from_predictions(y_test, y_proba, ax=ax2, name="LogReg")
save_and_show(fig2, os.path.join(OUT_DIR, "holdout_pr.png"))

# Matriz de confusión del holdout
fig3, ax3 = plt.subplots()
ConfusionMatrixDisplay(confusion_matrix=cm).plot(ax=ax3, colorbar=False)
ax3.set_title("Matriz de confusion (Holdout)")
save_and_show(fig3, os.path.join(OUT_DIR, "holdout_cm.png"))

# -----------------------------------------------------------------------------
# 7) Validación cruzada externa con el umbral fijo
# -----------------------------------------------------------------------------
section(f"VALIDACION CRUZADA Stratified {N_SPLITS}-Fold (umbral {round(best_th,2)})")
skf2 = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
rows = []
for i, (tr, va) in enumerate(skf2.split(X, y), start=1):
    Xtr, Xva = X.iloc[tr], X.iloc[va]
    ytr, yva = y.iloc[tr], y.iloc[va]
    m = Pipeline([("pre", pre), ("clf", clf)])
    m.fit(Xtr, ytr)
    p = m.predict_proba(Xva)[:, 1]
    yhat = (p >= best_th).astype(int)
    f_acc = accuracy_score(yva, yhat)
    f_prec = precision_score(yva, yhat, zero_division=0)
    f_rec = recall_score(yva, yhat, zero_division=0)
    f_f1 = f1_score(yva, yhat, zero_division=0)
    f_auc = roc_auc_score(yva, p)
    cmv = confusion_matrix(yva, yhat)
    rows.append({
        "fold": i,
        "accuracy": f_acc, "precision": f_prec, "recall": f_rec, "f1": f_f1,
        "roc_auc": f_auc, "tn": int(cmv[0,0]), "fp": int(cmv[0,1]),
        "fn": int(cmv[1,0]), "tp": int(cmv[1,1])
    })

cv_df = pd.DataFrame(rows).set_index("fold")
print("Resultados por fold:")
print(cv_df.round(4).to_string())
cv_mean = cv_df.mean(numeric_only=True)
print("\nPromedios CV:")
print(cv_mean.round(4).to_string())

# Barras de F1 por fold
fig4, ax4 = plt.subplots()
cv_df["f1"].plot(kind="bar", ax=ax4)
ax4.set_title("F1 por fold (CV)")
ax4.set_xlabel("Fold")
ax4.set_ylabel("F1")
save_and_show(fig4, os.path.join(OUT_DIR, "cv_f1_por_fold.png"))

# Barras de ROC-AUC por fold
fig5, ax5 = plt.subplots()
cv_df["roc_auc"].plot(kind="bar", ax=ax5)
ax5.set_title("ROC-AUC por fold (CV)")
ax5.set_xlabel("Fold")
ax5.set_ylabel("ROC-AUC")
save_and_show(fig5, os.path.join(OUT_DIR, "cv_auc_por_fold.png"))

# -----------------------------------------------------------------------------
# 8) Entrenamiento final, reporte de coeficientes y guardado
# -----------------------------------------------------------------------------
final_model = Pipeline([("pre", pre), ("clf", clf)])
final_model.fit(X, y)

# Nombres de features después del preprocesamiento y sus coeficientes.
final_clf = final_model.named_steps["clf"]
final_pre = final_model.named_steps["pre"]
names = final_pre.get_feature_names_out()
coefs = final_clf.coef_.ravel()
top = sorted(zip(names, coefs), key=lambda x: abs(x[1]), reverse=True)

section("TOP 15 COEFICIENTES ABSOLUTOS")
for n, c in top[:15]:
    print(f"{n:<45} {c:+.4f}")

# Guardado del pipeline completo (imputación, one-hot, escalado y clasificador)
ensure_outdir(OUT_DIR)
model_path = os.path.join(OUT_DIR, "logreg_adult_pipeline.joblib")
dump(final_model, model_path)

# Metadata: contexto de entrenamiento para reproducibilidad futura.
meta = {
    "random_state": RANDOM_STATE,
    "n_splits": N_SPLITS,
    "target_column": TARGET_COL,
    "columns_dropped_by_permutation": to_drop,
    "columns_kept": to_keep,
    "best_threshold_cv_f1": float(best_th),
    "holdout_metrics": {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "roc_auc": float(roc_auc)
    },
    "cv_summary_means": {k: float(v) for k, v in cv_mean.to_dict().items()},
    "artifacts": {
        "model_joblib": model_path,
        "holdout_roc_png": os.path.join(OUT_DIR, "holdout_roc.png"),
        "holdout_pr_png":  os.path.join(OUT_DIR, "holdout_pr.png"),
        "holdout_cm_png":  os.path.join(OUT_DIR, "holdout_cm.png"),
        "cv_f1_png":       os.path.join(OUT_DIR, "cv_f1_por_fold.png"),
        "cv_auc_png":      os.path.join(OUT_DIR, "cv_auc_por_fold.png"),
    }
}
with open(os.path.join(OUT_DIR, "model_metadata.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

section("ARTEFACTOS GUARDADOS")
kv("Modelo .joblib", model_path)
kv("Metadata JSON", os.path.join(OUT_DIR, "model_metadata.json"))
kv("Gráficas", OUT_DIR)
section("FIN")
