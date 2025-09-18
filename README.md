# MiniReto_BloqueIA

Proyecto con dos partes:

- Entrenamiento/ML: [modelo_reg_logi/logistic-regression.py](modelo_reg_logi/logistic-regression.py), que entrena una Regresión Logística sobre el dataset en español y genera artefactos (modelo y gráficas).
- API/Front: [Front/main.py](Front/main.py), un backend FastAPI que carga el modelo, expone endpoints y sirve una UI simple en HTML ([Front/Templates/index.html](Front/Templates/index.html)).

## Estructura principal

- `modelo_reg_logi/`
  - Código y datos del modelo (CSV: [modelo_reg_logi/Estudio_de_Factores_Asociados_al_Ingreso.csv](modelo_reg_logi/Estudio_de_Factores_Asociados_al_Ingreso.csv))
  - Script de entrenamiento: [modelo_reg_logi/logistic-regression.py](modelo_reg_logi/logistic-regression.py)
  - Artefactos de entrenamiento: `modelo_reg_logi/artifacts_adult_income/` (PNG, `.joblib`, `model_metadata.json`)
- `Front/`
  - API y front en FastAPI: [Front/main.py](Front/main.py)
  - Plantilla HTML: [Front/Templates/index.html](Front/Templates/index.html)
  - Nota: La API sirve las imágenes desde `modelo_reg_logi/artifacts_adult_income` en la ruta `/artifacts`.
- Requerimientos globales: [requirements.txt](requirements.txt)

## Requisitos

- Python 3.10+ y `pip`
- macOS/Linux/Windows

## Ambiente virtual de Python

Crear y activar:

```bash
python3 -m venv env
source env/bin/activate
```

Desactivar:

```bash
deactivate
```

## Instalación de dependencias

Instalar todas las dependencias (ML + API/Front) desde el archivo unificado:

```bash
pip install -r requirements.txt
```

## Ejecutar solo el entrenamiento (ML)

Esto leerá el CSV, entrenará el modelo, generará métricas y guardará artefactos en `modelo_reg_logi/artifacts_adult_income/`.

```bash
cd modelo_reg_logi
python logistic-regression.py
```

Artefactos esperados:

- Modelo: `logreg_adult_pipeline.joblib`
- Metadata: `model_metadata.json`
- Gráficas: `holdout_roc.png`, `holdout_pr.png`, `holdout_cm.png`, `cv_f1_por_fold.png`, `cv_auc_por_fold.png`

## Ejecutar la API y el Front

Levanta FastAPI y sirve la UI. La API importa perezosamente el script del modelo, por lo que entrenará/cargará el pipeline al primer uso.

```bash
cd Front
uvicorn main:app --reload --port 8000
```

Abrir en el navegador:

- UI: http://127.0.0.1:8000/
- Imágenes generadas: se listan en la UI y también están disponibles bajo `/artifacts/...`

### Endpoints útiles

- `GET /` — UI HTML.
- `GET /schema` — Esquema del modelo (columnas, métricas, umbral).
- `GET /metrics` — Métricas y umbral desde metadata.
- `GET /figs` — Lista de imágenes de `artifacts_adult_income`.
- `POST /predict` — Predicción para un registro.

Ejemplo de `POST /predict` (campos de ejemplo del dataset):

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "data": {
      "Edad": 35,
      "Tipo de empleo": "Sector privado",
      "Nivel educativo más alto": "Universidad/Licenciatura",
      "Estado civil": "Soltero(a)",
      "Ocupación": "Tecnología y servicios profesionales",
      "Relación en el hogar": "Hijo(a)",
      "Raza": "Mestizo(a)",
      "Género": "Masculino",
      "País de origen": "Mexico",
      "Horas trabajadas por semana": 40
    }
  }'
```

La respuesta incluye `proba_1`, `pred_class` y el `threshold` usado.

## Notas

- La API sirve las imágenes desde `modelo_reg_logi/artifacts_adult_income` bajo la ruta `/artifacts`.
- El script del modelo maneja distintos encodings del CSV y realiza imputación/One-Hot internamente.
- Si cambias el CSV o el script, vuelve a ejecutar el entrenamiento o reinicia la API para regenerar artefactos.
  La respuesta incluye `proba_1`, `pred_class` y el `threshold` usado.

## Notas

- La API sirve las imágenes desde `modelo_reg_logi/artifacts_adult_income` bajo la ruta `/artifacts`.
- El script del modelo maneja distintos encodings del CSV y realiza imputación/One-Hot internamente.
- Si cambias el CSV o el script, vuelve a ejecutar el entrenamiento o reinicia la API para regenerar artefactos.
