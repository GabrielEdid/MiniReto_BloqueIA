# logistic-regression.py
# Regresión logística robusta con tuning y umbral óptimo para predicción más certera

from typing import Dict, Tuple, List
import os, re, json
import numpy as np
import pandas as pd
from unidecode import unidecode

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve
)

# -------- Parámetros generales --------
RANDOM_STATE = 42
TEST_SIZE = 0.20
N_SPLITS = 5
SAVE_ARTIFACTS = False
ARTIFACTS_DIR = "artifacts_ingreso"

# ======== Utilidades de lectura/normalización ========
def _read_csv_any(csv_path: str) -> pd.DataFrame:
    encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(csv_path, encoding=enc, sep=None, engine="python")
        except Exception as e:
            last_err = e
            for sep in [";", ",", "\t"]:
                try:
                    return pd.read_csv(csv_path, encoding=enc, sep=sep, engine="python")
                except Exception as e2:
                    last_err = e2
                    continue
    raise last_err if last_err else ValueError("No pude leer el CSV con encodings/separadores comunes.")

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = []
    for c in df.columns:
        c2 = unidecode(str(c)).lower()
        c2 = re.sub(r'[^a-z0-9]+', '_', c2).strip('_')
        cols.append(c2)
    out = df.copy()
    out.columns = cols
    return out

def _detect_target(df_norm: pd.DataFrame) -> str:
    # Busca sueldo/ingreso/salario + 25,000 (ej. "…mayor_a_25_000…")
    for c in df_norm.columns:
        if re.search(r'(sueldo|ingreso|salario).*(25_?0{3}|25000)', c):
            return c
    raise ValueError(f"No encontré columna objetivo (~25,000). Revisa columnas: {list(df_norm.columns)}")

def _pick_features(df_norm: pd.DataFrame, target_col: str) -> List[str]:
    candidates = [
        'edad',
        'tipo_de_empleo',
        'nivel_educativo_mas_alto',
        'estado_civil',
        'ocupacion',
        'relacion_en_el_hogar',
        'raza',
        'genero',
        'pais_de_origen',
        'horas_trabajadas_por_semana',
        'tuviste_ganancias_de_capital_este_ano',
        'tuviste_perdidas_de_capital_este_ano',
    ]
    feats = [c for c in candidates if c in df_norm.columns and c != target_col]
    if not feats:
        feats = [c for c in df_norm.columns if c != target_col][:12]
    return feats

def _to_binary(y: pd.Series) -> pd.Series:
    y = y.astype(str).str.strip().str.lower()
    y = y.replace({'si':1,'sí':1,'yes':1,'true':1,'1':1,'no':0,'false':0,'0':0})
    try:
        return y.astype(float).astype(int)
    except Exception:
        classes = list(pd.Series(y).unique())
        if len(classes) == 2:
            pos_keys = ['si','sí','yes','1','mayor','>']
            pos = classes[1]
            for c in classes:
                if any(k in str(c) for k in pos_keys):
                    pos = c; break
            return (y == pos).astype(int)
        raise ValueError("La columna objetivo no es binaria (2 clases).")

# ======== Preprocesamiento / columnas / opciones UI ========
def _split_num_cat(X: pd.DataFrame, feature_names: List[str]):
    num_cols = [c for c in feature_names if c in X.columns and (
        pd.api.types.is_numeric_dtype(X[c]) or c in ['edad','horas_trabajadas_por_semana']
    )]
    cat_cols = [c for c in feature_names if c not in num_cols]
    return num_cols, cat_cols

def _cat_choices_from_data(X: pd.DataFrame, cat_cols: List[str]):
    choices = {}
    for c in cat_cols:
        vals = (X[c].astype(str).str.strip().str.lower().replace({'nan':''})
                .value_counts(dropna=True).head(25).index.tolist())
        choices[c] = [v for v in vals if v]
    return choices

def _build_preprocessor(numeric_cols, categorical_cols) -> ColumnTransformer:
    num = Pipeline([("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())])
    cat = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", num, numeric_cols), ("cat", cat, categorical_cols)])

# ======== Tuning y selección de umbral ========
def _best_threshold_f1(y_true, proba) -> float:
    # Calcula umbral que maximiza F1 usando la curva Precision-Recall
    prec, rec, thr = precision_recall_curve(y_true, proba)
    # sklearn devuelve thr de tamaño n-1; ajustamos
    f1 = (2 * prec * rec) / (prec + rec + 1e-9)
    # Ignorar el primer punto (umbral implícito) para evitar nan
    idx = int(np.nanargmax(f1[1:])) + 1
    # Si no hay thresholds (caso raro), fallback a 0.5
    return float(thr[idx-1]) if len(thr) > 0 else 0.5

def _cv_threshold(pipe_template: Pipeline, X: pd.DataFrame, y: pd.Series,
                  n_splits: int = N_SPLITS) -> float:
    # Obtiene umbral robusto por K-Fold (mediana de thresholds por fold)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    thresholds = []
    for tr_idx, va_idx in skf.split(X, y):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
        model = pipe_template
        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_va)[:, 1]
        thresholds.append(_best_threshold_f1(y_va, proba))
    return float(np.median(thresholds)) if thresholds else 0.5

# ======== Entrenamiento principal ========
def train_from_csv(csv_path: str) -> Tuple[Pipeline, dict]:
    # 1) Carga / normaliza
    df = _read_csv_any(csv_path)
    df = _normalize_columns(df)

    # 2) Target / features
    target = _detect_target(df)
    features = _pick_features(df, target)
    X = df[features].copy()
    y = _to_binary(df[target])

    # 3) Columnas y preprocesador
    num_cols, cat_cols = _split_num_cat(X, features)
    pre = _build_preprocessor(num_cols, cat_cols)

    # 4) Modelo base y grid de hiperparámetros
    base_lr = LogisticRegression(penalty="l2", solver="lbfgs",
                                 max_iter=1000, random_state=RANDOM_STATE)
    pipe = Pipeline([("pre", pre), ("clf", base_lr)])
    param_grid = {
        "clf__C": [0.01, 0.1, 1.0, 3.0, 10.0],
        "clf__class_weight": [None, "balanced"],
    }
    # Seleccionamos el mejor por ROC-AUC (umbral-independiente)
    gs = GridSearchCV(pipe, param_grid, scoring="roc_auc",
                      cv=StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE),
                      n_jobs=-1, refit=True)
    gs.fit(X, y)
    best_pipe = gs.best_estimator_

    # 5) Holdout para métricas con UMBRAL ÓPTIMO
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    # Reajustamos en train del holdout
    best_pipe.fit(X_tr, y_tr)
    proba_te = best_pipe.predict_proba(X_te)[:, 1]

    # Umbral óptimo (por F1) estimado con CV usando TODO el dataset y el mejor pipeline
    # (robusto; luego se usa para servir predicciones)
    decision_threshold = _cv_threshold(best_pipe, X, y, n_splits=N_SPLITS)

    y_pred_te = (proba_te >= decision_threshold).astype(int)

    hold_acc = accuracy_score(y_te, y_pred_te)
    hold_prec = precision_score(y_te, y_pred_te, zero_division=0)
    hold_rec  = recall_score(y_te, y_pred_te, zero_division=0)
    hold_f1   = f1_score(y_te, y_pred_te, zero_division=0)
    try:
        hold_auc = roc_auc_score(y_te, proba_te)
    except Exception:
        hold_auc = float("nan")
    hold_cm = confusion_matrix(y_te, y_pred_te)

    # 6) CV de métricas (con el MISMO threshold)
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    cv_rows = []
    for i, (tri, vai) in enumerate(skf.split(X, y), 1):
        X_tr, X_va = X.iloc[tri], X.iloc[vai]
        y_tr, y_va = y.iloc[tri], y.iloc[vai]
        m = gs.best_estimator_
        m.fit(X_tr, y_tr)
        p = m.predict_proba(X_va)[:, 1]
        yp = (p >= decision_threshold).astype(int)
        f_acc = accuracy_score(y_va, yp)
        f_prec = precision_score(y_va, yp, zero_division=0)
        f_rec = recall_score(y_va, yp, zero_division=0)
        f_f1 = f1_score(y_va, yp, zero_division=0)
        try:
            f_auc = roc_auc_score(y_va, p)
        except Exception:
            f_auc = float("nan")
        f_cm = confusion_matrix(y_va, yp)
        cv_rows.append({
            "fold": i,
            "accuracy": f_acc,
            "precision": f_prec,
            "recall": f_rec,
            "f1": f_f1,
            "roc_auc": f_auc,
            "tn": int(f_cm[0,0]), "fp": int(f_cm[0,1]),
            "fn": int(f_cm[1,0]), "tp": int(f_cm[1,1]),
        })
    cv_df = pd.DataFrame(cv_rows).set_index("fold")
    cv_mean = cv_df.mean(numeric_only=True).to_dict()

    # 7) Reentrenar en TODO para servir (production pipeline)
    final_pipe = gs.best_estimator_
    final_pipe.fit(X, y)

    # 8) Opciones para UI
    cat_choices = _cat_choices_from_data(X, cat_cols)

    # 9) Meta con TODO
    meta = {
        "features": features,
        "target": target,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "cat_choices": cat_choices,
        "decision_threshold": float(decision_threshold),
        "gridsearch": {
            "best_params": gs.best_params_,
            "best_score_roc_auc": float(gs.best_score_)
        },
        "holdout": {
            "test_size": TEST_SIZE,
            "accuracy": float(hold_acc),
            "precision": float(hold_prec),
            "recall": float(hold_rec),
            "f1": float(hold_f1),
            "roc_auc": float(hold_auc),
            "confusion_matrix": {
                "tn": int(hold_cm[0,0]), "fp": int(hold_cm[0,1]),
                "fn": int(hold_cm[1,0]), "tp": int(hold_cm[1,1]),
            },
            "classification_report": classification_report(y_te, y_pred_te, output_dict=True, zero_division=0)
        },
        "cv": {
            "n_splits": N_SPLITS,
            "folds": cv_df.round(6).reset_index().to_dict(orient="records"),
            "mean": {k: (float(v) if pd.notna(v) else None) for k,v in cv_mean.items()}
        },
        # Para compatibilidad con tu UI anterior
        "training_accuracy": float(final_pipe.score(X, y)),
    }

    if SAVE_ARTIFACTS:
        os.makedirs(ARTIFACTS_DIR, exist_ok=True)
        with open(os.path.join(ARTIFACTS_DIR, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    return final_pipe, meta

# ======== Predicción (usa el umbral óptimo aprendido) ========
def predict_one(pipe: Pipeline, meta: dict, payload: Dict[str, str]):
    row = {}
    for f in meta["features"]:
        v = payload.get(f, None)
        if v in [None, ""]:
            row[f] = np.nan
        else:
            if f in meta["num_cols"]:
                try:
                    row[f] = float(v)
                except Exception:
                    row[f] = np.nan
            else:
                row[f] = str(v).strip().lower()
    X = pd.DataFrame([row])
    proba = float(pipe.predict_proba(X)[0, 1])
    thr = float(meta.get("decision_threshold", 0.5))
    pred = int(proba >= thr)
    return pred, proba
