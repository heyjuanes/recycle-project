import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "proto"))

import grpc
import recycling_pb2
import recycling_pb2_grpc


def test_connection():
    """Prueba la conexion con el servidor gRPC enviando una imagen de prueba."""
    channel = grpc.insecure_channel("localhost:50051")
    stub = recycling_pb2_grpc.RecyclingInferenceStub(channel)

    with open("yolov8n.pt", "rb") as f:
        fake_image = f.read(100)

    request = recycling_pb2.DetectionRequest(image_data=fake_image)
    response = stub.DetectObjects(request)

    print("Conexion exitosa:", response.success)
    print("Mensaje:", response.message)


if __name__ == "__main__":
    test_connection()