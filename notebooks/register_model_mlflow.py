"""
register_model_mlflow.py

Script para registrar el modelo TrashNet v5 en MLflow.

Este script carga el modelo TrashNet v5 preentrenado (descargado de
Roboflow Universe), registra sus parametros y metricas oficiales en
MLflow y almacena el archivo del modelo como artefacto en el
Model Registry.

Se ejecuta una sola vez durante la fase de configuracion del proyecto.
No requiere entrenamiento propio ya que utiliza los pesos oficiales
publicados por Polygence Project en Roboflow Universe.

Modelo: TrashNet v5
Fuente: https://universe.roboflow.com/polygence-project/
        trashnet-a-set-of-annotated-images-of-trash-that-can-be-used-for-object-detection
Licencia: CC BY 4.0

Metricas registradas (fuente: Roboflow Universe - TrashNet v5):
    - mAP50:     0.661
    - precision: 0.802
    - recall:    0.601

Clases detectadas (6):
    glass, paper, cardboard, plastic, metal, trash

Uso:
    Asegurate de tener el entorno virtual activado y ejecuta:
    python notebooks/register_model_mlflow.py

    Para ver los resultados en la interfaz de MLflow:
    mlflow ui
    Luego abre: http://localhost:5000
"""

import mlflow
from ultralytics import YOLO

# ── Configuracion MLflow ──────────────────────────────────────────────────────
EXPERIMENT_NAME = "recycling-trashnet-registration"
MODEL_PATH = "trashnet.pt"

# ── Metricas oficiales TrashNet v5 (Roboflow Universe) ────────────────────────
# Fuente: https://universe.roboflow.com/polygence-project/
#         trashnet-a-set-of-annotated-images-of-trash-that-can-be-used-for-object-detection/model/5
TRASHNET_METRICS = {
    "mAP50":     0.661,
    "precision": 0.802,
    "recall":    0.601,
}

# ── Parametros del modelo ─────────────────────────────────────────────────────
MODEL_PARAMS = {
    "model_name":      "TrashNet v5",
    "model_type":      "YOLOv8 Object Detection",
    "source":          "Roboflow Universe - Polygence Project",
    "num_classes":     6,
    "classes":         "glass,paper,cardboard,plastic,metal,trash",
    "dataset_images":  2524,
    "input_size":      640,
    "framework":       "Ultralytics",
    "training_mode":   "pretrained",
    "model_card":      "https://universe.roboflow.com/polygence-project/trashnet-a-set-of-annotated-images-of-trash-that-can-be-used-for-object-detection/model/5",
    "license":         "CC BY 4.0",
    "python_version":  "3.11",
}


def register_trashnet_model() -> None:
    """
    Carga TrashNet v5 y lo registra en MLflow.

    El proceso completo incluye:
        1. Crear o seleccionar el experimento en MLflow.
        2. Iniciar un nuevo run con nombre descriptivo.
        3. Cargar el modelo TrashNet desde trashnet.pt.
        4. Registrar los parametros del modelo.
        5. Registrar las metricas oficiales de Roboflow.
        6. Guardar el archivo trashnet.pt como artefacto del run.
        7. Imprimir el Run ID para referencia futura.

    Note:
        El archivo trashnet.pt debe existir en la raiz del proyecto.
        Se descarga previamente con notebooks/download_trashnet.py.

        El archivo trashnet.pt esta en el .gitignore y se gestiona
        exclusivamente como artefacto de MLflow, no como archivo
        del repositorio.

    Returns:
        None

    Raises:
        FileNotFoundError: si trashnet.pt no existe en la raiz del proyecto.
        Exception: si MLflow no puede conectarse al servidor de tracking.
    """
    # Crear o seleccionar experimento
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="trashnet-v5-roboflow") as run:

        # 1. Cargar modelo TrashNet
        print("Cargando modelo TrashNet v5...")
        model = YOLO(MODEL_PATH)
        print(f"Clases del modelo: {list(model.names.values())}")

        # 2. Registrar parametros del modelo
        print("Registrando parametros...")
        mlflow.log_params(MODEL_PARAMS)

        # 3. Registrar metricas oficiales de Roboflow
        print("Registrando metricas...")
        mlflow.log_metrics(TRASHNET_METRICS)

        # 4. Guardar modelo como artefacto
        print("Registrando artefacto...")
        mlflow.log_artifact(MODEL_PATH, artifact_path="model")

        # 5. Imprimir Run ID para referencia
        print("=" * 50)
        print("Modelo registrado exitosamente.")
        print(f"Experimento : {EXPERIMENT_NAME}")
        print(f"Run ID      : {run.info.run_id}")
        print(f"Artefacto   : {MODEL_PATH}")
        print(f"Clases      : {MODEL_PARAMS['classes']}")
        print("=" * 50)
        print("Abre MLflow UI con: mlflow ui")
        print("Luego visita: http://localhost:5000")


if __name__ == "__main__":
    register_trashnet_model()