import mlflow
from ultralytics import YOLO

EXPERIMENT_NAME = "recycling-yolov8-registration"
MODEL_PATH = "yolov8n.pt"


def register_pretrained_model():
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name="yolov8n-coco-pretrained") as run:
        print("Cargando modelo YOLOv8n...")
        YOLO(MODEL_PATH)
        mlflow.log_params({
            "model_type": "YOLOv8n",
            "weights": "COCO pretrained",
            "num_classes": 80,
            "input_size": 640,
            "framework": "Ultralytics",
            "training_mode": "pretrained",
        })
        mlflow.log_metrics({
            "mAP50": 0.525,
            "mAP50_95": 0.372,
            "precision": 0.680,
            "recall": 0.532,
        })
        print("Registrando artefacto...")
        mlflow.log_artifact(MODEL_PATH, artifact_path="model")
        print("Listo! Run ID:", run.info.run_id)
        print("Abre MLflow con: mlflow ui")


if __name__ == "__main__":
    register_pretrained_model()