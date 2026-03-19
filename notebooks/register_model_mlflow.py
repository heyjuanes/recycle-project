"""
register_model_mlflow.py

Script para registrar el modelo YOLO Waste Detection en MLflow.

Este script carga el modelo preentrenado (descargado de GitHub),
registra sus parametros en MLflow y almacena el archivo del modelo
como artefacto en el Model Registry.

Se ejecuta una sola vez durante la fase de configuracion del proyecto.
No requiere entrenamiento propio ya que utiliza los pesos publicados
por gianlucasposito en GitHub bajo licencia MIT.

Modelo: YOLO Waste Detection
Fuente: https://github.com/gianlucasposito/YOLO-Waste-Detection
Licencia: MIT

Clases detectadas (5):
    Glass, Metal, Paper, Plastic, Waste

Uso:
    Asegurate de tener el entorno virtual activado y ejecuta:
    python notebooks/register_model_mlflow.py

    Para ver los resultados en la interfaz de MLflow:
    mlflow ui
    Luego abre: http://localhost:5000
"""

import os
import mlflow
from ultralytics import YOLO

# ── Configuracion MLflow ──────────────────────────────────────────────────────
EXPERIMENT_NAME = "recycling-waste-detection"
MODEL_PATH = "trashnet.pt"

# ── Parametros del modelo ─────────────────────────────────────────────────────
MODEL_PARAMS = {
    "model_name":      "YOLO Waste Detection",
    "model_type":      "YOLOv8 Object Detection (fine-tuned)",
    "source":          "github.com/gianlucasposito/YOLO-Waste-Detection",
    "num_classes":     5,
    "classes":         "Glass,Metal,Paper,Plastic,Waste",
    "dataset_images":  4127,
    "dataset_train":   3502,
    "dataset_val":     580,
    "dataset_test":    45,
    "input_size":      640,
    "framework":       "Ultralytics YOLOv8",
    "base_model":      "YOLOv8n (COCO pretrained)",
    "training_mode":   "fine-tuned",
    "license":         "MIT",
    "python_version":  "3.11",
}


def register_model() -> None:
    """
    Carga el modelo YOLO Waste Detection y lo registra en MLflow.

    El proceso completo incluye:
        1. Crear o seleccionar el experimento en MLflow.
        2. Iniciar un nuevo run con nombre descriptivo.
        3. Cargar el modelo desde trashnet.pt.
        4. Registrar los parametros del modelo.
        5. Guardar el archivo trashnet.pt como artefacto del run.
        6. Imprimir el Run ID para referencia futura.

    Note:
        El archivo trashnet.pt debe existir en la raiz del proyecto.
        Descargalo con:
            Invoke-WebRequest -Uri "https://github.com/gianlucasposito/
            YOLO-Waste-Detection/raw/main/best_model.pt" -OutFile "trashnet.pt"

        El archivo trashnet.pt esta en el .gitignore y se gestiona
        exclusivamente como artefacto de MLflow, no como archivo
        del repositorio.

    Returns:
        None

    Raises:
        FileNotFoundError: si trashnet.pt no existe en la raiz del proyecto.
        Exception: si MLflow no puede conectarse al servidor de tracking.
    """
    # Resolver ruta absoluta al modelo desde la raiz del proyecto
    root = os.path.join(os.path.dirname(__file__), "..")
    model_full_path = os.path.join(root, MODEL_PATH)

    # Crear o seleccionar experimento
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="yolov8-waste-detection-gianlucasposito") as run:

        # 1. Cargar modelo
        print("Cargando modelo YOLO Waste Detection...")
        model = YOLO(model_full_path)
        print(f"Clases del modelo: {list(model.names.values())}")

        # 2. Registrar parametros del modelo
        print("Registrando parametros...")
        mlflow.log_params(MODEL_PARAMS)

        # 3. Guardar modelo como artefacto
        print("Registrando artefacto...")
        mlflow.log_artifact(model_full_path, artifact_path="model")

        # 4. Imprimir Run ID para referencia
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
    register_model()