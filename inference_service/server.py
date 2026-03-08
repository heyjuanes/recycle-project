import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "proto"))

import io
import grpc
from concurrent import futures
from PIL import Image
from ultralytics import YOLO

import recycling_pb2 # type: ignore
import recycling_pb2_grpc # type: ignore
from material_classifier import get_material

MODEL_PATH = "yolov8n.pt"
PORT = "50051"


class RecyclingInferenceServicer(recycling_pb2_grpc.RecyclingInferenceServicer):
    """Servidor gRPC que ejecuta inferencia YOLOv8 sobre imagenes."""

    def __init__(self):
        """Inicializa el servicer cargando el modelo YOLOv8."""
        print("Cargando modelo YOLOv8n...")
        self.model = YOLO(MODEL_PATH)
        print("Modelo listo.")

    def DetectObjects(self, request, context):
        """
        Recibe una imagen en bytes, ejecuta YOLOv8 y retorna detecciones.

        Args:
            request: DetectionRequest con imagen en bytes.
            context: contexto gRPC.

        Returns:
            DetectionResponse con lista de objetos detectados.
        """
        try:
            image = Image.open(io.BytesIO(request.image_data))
            results = self.model(image, verbose=False)
            detected_objects = []

            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    class_name = self.model.names[class_id]
                    confidence = float(box.conf[0])
                    x1, y1, x2, y2 = [float(c) for c in box.xyxy[0]]
                    material = get_material(class_name)

                    detected_objects.append(
                        recycling_pb2.DetectedObject(
                            class_name=class_name,
                            confidence=confidence,
                            x1=x1,
                            y1=y1,
                            x2=x2,
                            y2=y2,
                            material=material,
                        )
                    )

            return recycling_pb2.DetectionResponse(
                objects=detected_objects,
                success=True,
                message=f"Se detectaron {len(detected_objects)} objetos.",
            )

        except Exception as e:
            return recycling_pb2.DetectionResponse(
                objects=[],
                success=False,
                message=f"Error: {str(e)}",
            )


def serve():
    """Inicia el servidor gRPC en el puerto 50051."""
    options = [
        ("grpc.max_receive_message_length", 20 * 1024 * 1024),
        ("grpc.max_send_message_length", 20 * 1024 * 1024),
    ]
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4), options=options)
    recycling_pb2_grpc.add_RecyclingInferenceServicer_to_server(
        RecyclingInferenceServicer(), server
    )
    server.add_insecure_port(f"[::]:{PORT}")
    server.start()
    print(f"Servidor gRPC corriendo en puerto {PORT}...")
    server.wait_for_termination()


if __name__ == "__main__":
    serve()