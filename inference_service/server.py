"""
server.py

Servidor gRPC que expone el servicio de inferencia TrashNet para deteccion
de objetos reciclables. Actua como el microservicio de inferencia del sistema,
recibiendo imagenes en bytes y retornando las detecciones con su clasificacion
de material reciclable.

Modelo: TrashNet v5 (Roboflow Universe - Polygence Project)
Clases: glass, paper, cardboard, plastic, metal, trash
Licencia del modelo: CC BY 4.0
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "proto"))

import io
import grpc
from concurrent import futures
from PIL import Image
from ultralytics import YOLO

import recycling_pb2  # type: ignore
import recycling_pb2_grpc  # type: ignore
from material_classifier import get_material

# ── Configuracion del servidor ────────────────────────────────────────────────
MODEL_PATH = "trashnet.pt"
PORT = "50051"
MAX_WORKERS = 4
MAX_MESSAGE_SIZE = 20 * 1024 * 1024  # 20MB


class RecyclingInferenceServicer(recycling_pb2_grpc.RecyclingInferenceServicer):
    """
    Implementacion del servicio gRPC RecyclingInference.

    Carga el modelo TrashNet al inicializar y expone el metodo DetectObjects
    para ejecutar inferencia sobre imagenes recibidas en bytes.

    El modelo TrashNet detecta 6 clases de residuos:
        glass, paper, cardboard, plastic, metal, trash

    Attributes:
        model (YOLO): instancia del modelo TrashNet cargado en memoria.
    """

    def __init__(self):
        """
        Inicializa el servicer cargando el modelo TrashNet desde disco.

        El modelo se carga una sola vez en memoria al iniciar el servidor
        para optimizar el tiempo de respuesta en cada inferencia posterior.

        Raises:
            FileNotFoundError: si el archivo trashnet.pt no existe en
                la ruta esperada.
        """
        print("Cargando modelo TrashNet v5...")
        self.model = YOLO(MODEL_PATH)
        print("Modelo TrashNet listo. Clases:", list(self.model.names.values()))

    def DetectObjects(self, request, context):
        """
        Ejecuta inferencia TrashNet sobre una imagen recibida via gRPC.

        Decodifica la imagen desde bytes, ejecuta el modelo TrashNet y
        construye la respuesta con los objetos detectados, su confianza,
        coordenadas del bounding box y categoria de material reciclable.

        Args:
            request (DetectionRequest): mensaje gRPC con el campo
                image_data (bytes) conteniendo la imagen codificada.
            context (grpc.ServicerContext): contexto de la llamada gRPC.
                Usado internamente por el framework para manejo de errores.

        Returns:
            DetectionResponse: mensaje gRPC con los campos:
                - objects (list[DetectedObject]): lista de objetos detectados.
                  Cada objeto contiene class_name, confidence, x1, y1, x2, y2
                  y material.
                - success (bool): True si la inferencia fue exitosa.
                - message (str): descripcion del resultado o del error.

        Raises:
            Exception: cualquier error durante la decodificacion o inferencia
                es capturado y retornado como DetectionResponse con
                success=False y el mensaje del error.
        """
        try:
            # Decodificar bytes a imagen PIL
            image = Image.open(io.BytesIO(request.image_data))

            # Ejecutar inferencia TrashNet
            results = self.model(image, verbose=False)
            detected_objects = []

            # Iterar sobre cada deteccion
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
                message=f"Error durante la inferencia: {str(e)}",
            )


def serve():
    """
    Inicializa y arranca el servidor gRPC en el puerto 50051.

    Configura el servidor con un limite de mensaje de 20MB para soportar
    imagenes de alta resolucion, y un pool de 4 threads para manejar
    solicitudes concurrentes.

    El servidor bloquea la ejecucion hasta recibir una senal de terminacion
    (Ctrl+C o signal del sistema operativo).

    Note:
        El servidor escucha en todas las interfaces de red ([::]) para
        ser accesible tanto desde localhost como desde Docker.
    """
    options = [
        ("grpc.max_receive_message_length", MAX_MESSAGE_SIZE),
        ("grpc.max_send_message_length", MAX_MESSAGE_SIZE),
    ]
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=MAX_WORKERS),
        options=options,
    )
    recycling_pb2_grpc.add_RecyclingInferenceServicer_to_server(
        RecyclingInferenceServicer(), server
    )
    server.add_insecure_port(f"[::]:{PORT}")
    server.start()
    print(f"Servidor gRPC corriendo en puerto {PORT}...")
    server.wait_for_termination()


if __name__ == "__main__":
    serve()