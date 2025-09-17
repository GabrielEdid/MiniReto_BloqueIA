# -*- coding: utf-8 -*-
"""
Adult Income - Regresión Logística con scikit-learn
Evaluación: Accuracy, Precision, Recall, F1, ROC-AUC, Matriz de Confusión
+ Validación cruzada Stratified K-Fold e identificación del mejor fold
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    RocCurveDisplay, PrecisionRecallDisplay, ConfusionMatrixDisplay
)
from joblib import dump

# Intentar usar kagglehub; si no, leer adult.csv local
try:
    import kagglehub
    from kagglehub import KaggleDatasetAdapter, dataset_download, dataset_load
    USE_KAGGLEHUB = True
except Exception:
    USE_KAGGLEHUB = False

HANDLE = "mosapabdelghany/adult-income-prediction-dataset"
FILE_NAME = "adult.csv"
RANDOM_STATE = 42
N_SPLITS = 5  # K de K-Fold

# 1) Carga de datos
if USE_KAGGLEHUB:
    dataset_path = dataset_download(HANDLE)
    print("Dataset descargado (o en caché) en:", dataset_path)
    print(f"Cargando archivo: {FILE_NAME}")
    try:
        df = dataset_load(KaggleDatasetAdapter.PANDAS, HANDLE, FILE_NAME)
    except Exception as e:
        raise SystemExit(f"Error cargando '{FILE_NAME}': {e}")
else:
    if not os.path.exists(FILE_NAME):
        raise SystemExit("No se pudo importar kagglehub y no existe 'adult.csv' local.")
    df = pd.read_csv(FILE_NAME)

print("Primeras filas:\n", df.head())

# 2) Normalización de nombres de columnas
def normalize_col(c):
    c = c.strip().lower()
    c = c.replace(" ", ".").replace("-", ".")
    return c

df.columns = [normalize_col(c) for c in df.columns]

rename_map = {
    "education-num": "education.num",
    "marital-status": "marital.status",
    "capital-gain": "capital.gain",
    "capital-loss": "capital.loss",
    "hours-per-week": "hours.per.week",
    "native-country": "native.country"
}
df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

# 3) Limpieza básica
for col in df.columns:
    if df[col].dtype == object:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace({"?": np.nan})

# 4) Target y features
if "income" not in df.columns:
    raise SystemExit("No se encontró la columna 'income'.")
df["income"] = df["income"].str.replace(" ", "", regex=False)
y = df["income"].map({">50K": 1, "<=50K": 0, ">50K.": 1, "<=50K.": 0})
if y.isna().any():
    raise SystemExit("Valores inesperados en 'income'.")

X = df.drop(columns=["income"])
if "education" in X.columns and "education.num" in X.columns:
    X = X.drop(columns=["education"])

# 5) Columnas numéricas y categóricas
numeric_cols = [
    c for c in X.columns
    if pd.api.types.is_numeric_dtype(X[c])
    or c in ["age", "fnlwgt", "education.num", "capital.gain", "capital.loss", "hours.per.week"]
    if c in X.columns
]
categorical_cols = [c for c in X.columns if c not in numeric_cols]

# 6) Preprocesamiento y modelo
numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols),
    ]
)

log_reg = LogisticRegression(
    penalty="l2", solver="lbfgs",
    class_weight="balanced", max_iter=1000
)
pipe = Pipeline(steps=[("pre", preprocessor), ("clf", log_reg)])

# =========================
# A) Holdout train/test
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

pipe.fit(X_train, y_train)

y_proba = pipe.predict_proba(X_test)[:, 1]
y_pred = (y_proba >= 0.5).astype(int)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_proba)

print("\n=== Evaluación Holdout (20% Test) ===")
print(f"Exactitud (Accuracy):   {acc:.4f}")
print(f"Precisión (Precision):  {prec:.4f}")
print(f"Recuerdo (Recall):      {rec:.4f}")
print(f"F1-score:               {f1:.4f}")
print(f"ROC-AUC:                {roc_auc:.4f}")
print("\nMatriz de confusión:\n", confusion_matrix(y_test, y_pred))
print("\nReporte de clasificación:\n", classification_report(y_test, y_pred, digits=4))

disp = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred), display_labels=["<=50K", ">50K"])
disp.plot(cmap="Blues")
plt.title("Matriz de Confusión - Holdout")
plt.tight_layout()
plt.show()

RocCurveDisplay.from_predictions(y_test, y_proba)
plt.title("Curva ROC - Holdout")
plt.tight_layout()
plt.show()

PrecisionRecallDisplay.from_predictions(y_test, y_proba)
plt.title("Curva Precision-Recall - Holdout")
plt.tight_layout()
plt.show()

# ==========================================
# B) Stratified K-Fold en todo el dataset
# ==========================================
print(f"\n=== Stratified {N_SPLITS}-Fold CV (umbral 0.5) ===")
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

fold_rows = []
fold_idx = 0
best_by_f1 = {"fold": None, "f1": -1}
best_by_auc = {"fold": None, "auc": -1}

for train_idx, val_idx in skf.split(X, y):
    fold_idx += 1
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # Nuevo pipeline por fold para evitar fugas de estado
    fold_pipe = Pipeline(steps=[("pre", preprocessor), ("clf", log_reg)])
    fold_pipe.fit(X_tr, y_tr)

    val_proba = fold_pipe.predict_proba(X_val)[:, 1]
    val_pred = (val_proba >= 0.5).astype(int)

    f_acc = accuracy_score(y_val, val_pred)
    f_prec = precision_score(y_val, val_pred, zero_division=0)
    f_rec = recall_score(y_val, val_pred, zero_division=0)
    f_f1 = f1_score(y_val, val_pred, zero_division=0)
    f_auc = roc_auc_score(y_val, val_proba)
    f_cm = confusion_matrix(y_val, val_pred)

    fold_rows.append({
        "fold": fold_idx,
        "accuracy": f_acc,
        "precision": f_prec,
        "recall": f_rec,
        "f1": f_f1,
        "roc_auc": f_auc,
        "tn": int(f_cm[0,0]), "fp": int(f_cm[0,1]),
        "fn": int(f_cm[1,0]), "tp": int(f_cm[1,1])
    })

    if f_f1 > best_by_f1["f1"]:
        best_by_f1 = {"fold": fold_idx, "f1": f_f1, "cm": f_cm}
        best_by_f1["metrics"] = (f_acc, f_prec, f_rec, f_f1, f_auc)

    if f_auc > best_by_auc["auc"]:
        best_by_auc = {"fold": fold_idx, "auc": f_auc, "cm": f_cm}
        best_by_auc["metrics"] = (f_acc, f_prec, f_rec, f_f1, f_auc)

cv_df = pd.DataFrame(fold_rows).set_index("fold")
print("\nResultados por fold:\n", cv_df.round(4))
print("\nPromedios CV:\n", cv_df.mean().round(4))

print(f"\nMejor fold por F1: fold {best_by_f1['fold']}  "
      f"(F1={best_by_f1['f1']:.4f})")
acc_bf, prec_bf, rec_bf, f1_bf, auc_bf = best_by_f1["metrics"]
print(f"  Acc={acc_bf:.4f}  Prec={prec_bf:.4f}  Rec={rec_bf:.4f}  ROC-AUC={auc_bf:.4f}")
print("  CM (tn, fp, fn, tp):", best_by_f1["cm"].ravel().tolist())

print(f"\nMejor fold por ROC-AUC: fold {best_by_auc['fold']}  "
      f"(ROC-AUC={best_by_auc['auc']:.4f})")
acc_ba, prec_ba, rec_ba, f1_ba, auc_ba = best_by_auc["metrics"]
print(f"  Acc={acc_ba:.4f}  Prec={prec_ba:.4f}  Rec={rec_ba:.4f}  F1={f1_ba:.4f}")
print("  CM (tn, fp, fn, tp):", best_by_auc["cm"].ravel().tolist())

# Gráfica de matriz de confusión del mejor fold por F1
best_cm = best_by_f1["cm"]
disp_cv = ConfusionMatrixDisplay(confusion_matrix=best_cm, display_labels=["<=50K", ">50K"])
disp_cv.plot(cmap="Blues")
plt.title(f"Matriz de Confusión - Mejor Fold por F1 (Fold {best_by_f1['fold']})")
plt.tight_layout()
plt.show()

# =========================
# C) Guardado de artefactos
# =========================
OUT_DIR = "artifacts_adult_income"
os.makedirs(OUT_DIR, exist_ok=True)

# Reentrenar en todo el dataset para exportar modelo final
final_pipe = Pipeline(steps=[("pre", preprocessor), ("clf", log_reg)])
final_pipe.fit(X, y)
model_path = os.path.join(OUT_DIR, "logreg_adult_pipeline.joblib")
dump(final_pipe, model_path)
print(f"\nPipeline final guardado en: {model_path}")

meta = {
    "random_state": RANDOM_STATE,
    "numeric_cols": numeric_cols,
    "categorical_cols": categorical_cols,
    "holdout_metrics": {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "roc_auc": float(roc_auc)
    },
    "cv_summary": cv_df.mean().to_dict(),
    "best_fold_by_f1": {"fold": int(best_by_f1["fold"]), "f1": float(best_by_f1["f1"])},
    "best_fold_by_auc": {"fold": int(best_by_auc["fold"]), "roc_auc": float(best_by_auc["auc"])}
}
with open(os.path.join(OUT_DIR, "metadata.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)
print(f"Metadatos guardados en: {os.path.join(OUT_DIR, 'metadata.json')}")
