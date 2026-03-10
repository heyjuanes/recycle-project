"""
test_client.py

Cliente de prueba para verificar que la conexion gRPC con el
inference_service esta activa y respondiendo correctamente.

Este script NO es una prueba unitaria de pytest. Es una herramienta
de diagnostico manual para confirmar que el servidor gRPC esta corriendo
antes de levantar la interfaz Streamlit.

Uso:
    1. Asegurate de que inference_service/server.py esta corriendo.
    2. Ejecuta: python inference_service/test_client.py

Resultado esperado:
    Conexion exitosa: False
    Mensaje: Error durante la inferencia: cannot identify image file...

    El success=False es esperado porque se envian bytes invalidos como imagen.
    Lo importante es que el servidor RESPONDIO, confirmando que la conexion
    gRPC esta activa.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "proto"))

import grpc
import recycling_pb2  # type: ignore
import recycling_pb2_grpc  # type: ignore

# ── Configuracion ─────────────────────────────────────────────────────────────
GRPC_HOST = "localhost:50051"


def test_connection() -> None:
    """
    Prueba la conexion con el servidor gRPC enviando una peticion minima.

    Crea un canal gRPC hacia el inference_service, construye un
    DetectionRequest con bytes invalidos (no es una imagen real) y
    verifica que el servidor responde con un DetectionResponse valido.

    Un response con success=False pero con un mensaje de error es suficiente
    para confirmar que la conexion esta activa y el servidor esta procesando
    peticiones correctamente.

    Returns:
        None

    Raises:
        grpc.RpcError: si el servidor no esta corriendo o no es accesible
            en localhost:50051.
    """
    print(f"Conectando a {GRPC_HOST}...")
    channel = grpc.insecure_channel(GRPC_HOST)
    stub = recycling_pb2_grpc.RecyclingInferenceStub(channel)

    # Enviamos bytes invalidos intencionalmente para probar solo la conexion
    request = recycling_pb2.DetectionRequest(image_data=b"bytes_de_prueba")
    response = stub.DetectObjects(request)

    print("Conexion exitosa:", response.success)
    print("Mensaje:", response.message)
    print("Objetos detectados:", len(response.objects))


if __name__ == "__main__":
    test_connection()