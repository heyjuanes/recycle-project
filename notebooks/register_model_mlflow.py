"""
register_model_mlflow.py

Script para registrar el modelo YOLOv8n preentrenado en MLflow.

Este script descarga el modelo YOLOv8n preentrenado en COCO, registra
sus parametros y metricas oficiales en MLflow y almacena el archivo
del modelo como artefacto en el Model Registry.

Se ejecuta una sola vez durante la fase de configuracion del proyecto.
No requiere entrenamiento propio ya que utiliza los pesos oficiales
publicados por Ultralytics.

Metricas registradas (fuente: Ultralytics YOLOv8 official benchmarks):
    - mAP50:     0.525
    - mAP50_95:  0.372
    - precision: 0.680
    - recall:    0.532

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
EXPERIMENT_NAME = "recycling-yolov8-registration"
MODEL_PATH = "yolov8n.pt"

# ── Metricas oficiales YOLOv8n en COCO ───────────────────────────────────────
# Fuente: https://docs.ultralytics.com/models/yolov8/
COCO_METRICS = {
    "mAP50": 0.525,
    "mAP50_95": 0.372,
    "precision": 0.680,
    "recall": 0.532,
}

# ── Parametros del modelo ─────────────────────────────────────────────────────
MODEL_PARAMS = {
    "model_type": "YOLOv8n",
    "weights": "COCO pretrained",
    "num_classes": 80,
    "input_size": 640,
    "framework": "Ultralytics",
    "training_mode": "pretrained",
    "model_card": "https://docs.ultralytics.com/models/yolov8/",
    "license": "AGPL-3.0",
    "python_version": "3.11",
}


def register_pretrained_model() -> None:
    """
    Descarga YOLOv8n preentrenado y lo registra en MLflow.

    El proceso completo incluye:
        1. Crear o seleccionar el experimento en MLflow.
        2. Iniciar un nuevo run con nombre descriptivo.
        3. Descargar y cargar el modelo YOLOv8n (automatico si no existe).
        4. Registrar los parametros del modelo.
        5. Registrar las metricas oficiales de COCO.
        6. Guardar el archivo yolov8n.pt como artefacto del run.
        7. Imprimir el Run ID para referencia futura.

    Note:
        El archivo yolov8n.pt se descarga automaticamente desde los
        servidores de Ultralytics (~6MB) si no existe en el directorio
        de trabajo actual.

        El archivo yolov8n.pt esta en el .gitignore y se gestiona
        exclusivamente como artefacto de MLflow, no como archivo
        del repositorio.

    Returns:
        None

    Raises:
        Exception: si MLflow no puede conectarse al servidor de tracking
            o si la descarga del modelo falla por problemas de red.
    """
    # Crear o seleccionar experimento
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="yolov8n-coco-pretrained") as run:

        # 1. Cargar modelo (descarga automatica si no existe)
        print("Cargando modelo YOLOv8n...")
        YOLO(MODEL_PATH)

        # 2. Registrar parametros del modelo
        print("Registrando parametros...")
        mlflow.log_params(MODEL_PARAMS)

        # 3. Registrar metricas oficiales de COCO
        print("Registrando metricas...")
        mlflow.log_metrics(COCO_METRICS)

        # 4. Guardar modelo como artefacto
        print("Registrando artefacto...")
        mlflow.log_artifact(MODEL_PATH, artifact_path="model")

        # 5. Imprimir Run ID para referencia
        print("=" * 50)
        print(f"Modelo registrado exitosamente.")
        print(f"Experimento : {EXPERIMENT_NAME}")
        print(f"Run ID      : {run.info.run_id}")
        print(f"Artefacto   : {MODEL_PATH}")
        print("=" * 50)
        print("Abre MLflow UI con: mlflow ui")
        print("Luego visita: http://localhost:5000")


if __name__ == "__main__":
    register_pretrained_model()