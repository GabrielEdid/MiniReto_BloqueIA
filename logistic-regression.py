import kagglehub
from kagglehub import KaggleDatasetAdapter, dataset_download, dataset_load

HANDLE = "mosapabdelghany/adult-income-prediction-dataset"
FILE_NAME = "adult.csv"  # Nombre fijo

# Descarga (o usa caché) del dataset
dataset_path = dataset_download(HANDLE)
print("Dataset descargado (o en caché) en:", dataset_path)
print(f"Cargando archivo fijo: {FILE_NAME}")

try:
    df = dataset_load(
        KaggleDatasetAdapter.PANDAS,
        HANDLE,
        FILE_NAME,
    )
except Exception as e:
    raise SystemExit(f"Error cargando '{FILE_NAME}': {e}\nVerifica el nombre exacto dentro del dataset.")

print("Primeras filas:")
print(df.head())