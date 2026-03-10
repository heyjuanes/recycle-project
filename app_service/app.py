"""
app.py

Interfaz web Streamlit para el sistema de deteccion de material reciclable.
Actua como cliente gRPC del inference_service, enviando imagenes para
su procesamiento y visualizando los resultados con bounding boxes
y clasificacion de material.

Uso:
    Asegurate de que inference_service/server.py esta corriendo, luego:
    streamlit run app_service/app.py

    La interfaz estara disponible en: http://localhost:8501

Autor: Marco Antonio Acosta, Juan Esteban Espitia, Geraldine Filigrana
Asignatura: Desarrollo de Proyectos de IA
Universidad Autonoma de Occidente
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "proto"))

import io
import grpc
import streamlit as st
from PIL import Image, ImageDraw

import recycling_pb2  # type: ignore
import recycling_pb2_grpc  # type: ignore

# ── Configuracion ─────────────────────────────────────────────────────────────
GRPC_HOST = "localhost:50051"

# Colores por categoria de material para los bounding boxes
MATERIAL_COLORS = {
    "plastico": "#3498DB",       # Azul
    "metal": "#E74C3C",          # Rojo
    "vidrio": "#2ECC71",         # Verde
    "carton": "#F39C12",         # Naranja
    "no clasificado": "#95A5A6", # Gris
}


def get_stub() -> recycling_pb2_grpc.RecyclingInferenceStub:
    """
    Crea y retorna el stub gRPC conectado al inference_service.

    Establece un canal gRPC inseguro (sin TLS) hacia el servidor
    de inferencia en localhost:50051 y retorna el stub del servicio
    RecyclingInference listo para realizar llamadas RPC.

    Returns:
        RecyclingInferenceStub: stub gRPC para llamar a DetectObjects.
    """
    channel = grpc.insecure_channel(GRPC_HOST)
    return recycling_pb2_grpc.RecyclingInferenceStub(channel)


def draw_boxes(image: Image.Image, objects: list) -> Image.Image:
    """
    Dibuja bounding boxes sobre la imagen con etiquetas de clase y material.

    Para cada objeto detectado dibuja un rectangulo de color segun su
    categoria de material, y una etiqueta con el nombre de la clase,
    el material y el porcentaje de confianza.

    Args:
        image (Image.Image): imagen PIL original sobre la que se dibujaran
            los bounding boxes.
        objects (list): lista de DetectedObject retornados por el
            inference_service via gRPC. Cada objeto tiene los atributos:
            class_name, material, confidence, x1, y1, x2, y2.

    Returns:
        Image.Image: copia de la imagen original con los bounding boxes
            y etiquetas dibujados.
    """
    draw = ImageDraw.Draw(image)

    for obj in objects:
        color = MATERIAL_COLORS.get(obj.material, "#95A5A6")

        # Dibujar rectangulo del bounding box
        draw.rectangle(
            [obj.x1, obj.y1, obj.x2, obj.y2],
            outline=color,
            width=3,
        )

        # Construir etiqueta con clase, material y confianza
        label = f"{obj.class_name} ({obj.material}) {obj.confidence:.0%}"

        # Dibujar fondo de la etiqueta
        draw.rectangle(
            [obj.x1, obj.y1 - 20, obj.x1 + len(label) * 7, obj.y1],
            fill=color,
        )

        # Dibujar texto de la etiqueta
        draw.text((obj.x1 + 2, obj.y1 - 18), label, fill="white")

    return image


def run_detection(image_bytes: bytes):
    """
    Comprime la imagen y la envia al inference_service via gRPC.

    Antes de enviar la imagen aplica dos optimizaciones:
        1. Redimensiona a maximo 1280px en cualquier dimension con thumbnail
           para reducir el tamano sin distorsionar la proporcion.
        2. Recodifica a JPEG con calidad 85 para reducir el peso en bytes.

    Esto garantiza que la imagen este por debajo del limite de 4MB de gRPC
    incluso para fotografias de alta resolucion.

    Args:
        image_bytes (bytes): bytes crudos de la imagen subida por el usuario
            directamente desde el file_uploader de Streamlit.

    Returns:
        DetectionResponse: respuesta gRPC del inference_service con los
            campos success, message y objects (lista de DetectedObject).
    """
    # Abrir y convertir a RGB (maneja PNG con canal alfa RGBA)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Redimensionar manteniendo proporcion
    image.thumbnail((1280, 1280))

    # Recodificar a JPEG para reducir tamano
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    compressed = buffer.getvalue()

    # Enviar al inference_service via gRPC
    stub = get_stub()
    request = recycling_pb2.DetectionRequest(image_data=compressed)
    return stub.DetectObjects(request)


def main() -> None:
    """
    Funcion principal de la interfaz Streamlit.

    Configura la pagina, muestra el cargador de imagenes y orquesta
    el flujo completo:
        1. Recibir imagen del usuario.
        2. Mostrar imagen original.
        3. Llamar a run_detection para enviar al inference_service.
        4. Mostrar imagen anotada con bounding boxes.
        5. Mostrar tarjetas con los resultados por objeto detectado.

    Maneja tres estados posibles:
        - Detecciones exitosas: muestra imagen anotada y tarjetas.
        - Sin detecciones: muestra mensaje informativo.
        - Error: muestra mensaje de error con detalle.
    """
    st.set_page_config(
        page_title="Detector de Reciclaje",
        page_icon="♻️",
        layout="wide",
    )

    st.title("♻️ Sistema de Deteccion de Material Reciclable")
    st.markdown("Sube una imagen para detectar objetos y clasificar su material.")

    uploaded_file = st.file_uploader(
        "Selecciona una imagen",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded_file is not None:
        image_bytes = uploaded_file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Imagen original")
            st.image(image, use_container_width=True)

        with st.spinner("Detectando objetos..."):
            response = run_detection(image_bytes)

        if response.success and response.objects:
            annotated = draw_boxes(image.copy(), response.objects)

            with col2:
                st.subheader("Detecciones")
                st.image(annotated, use_container_width=True)

            st.subheader(f"Se encontraron {len(response.objects)} objetos")

            for obj in response.objects:
                color = MATERIAL_COLORS.get(obj.material, "#95A5A6")
                st.markdown(
                    f"<div style='background-color:{color}20; "
                    f"border-left: 4px solid {color}; padding: 8px; "
                    f"margin: 4px 0; border-radius: 4px;'>"
                    f"<b>{obj.class_name}</b> → {obj.material.upper()} "
                    f"({obj.confidence:.0%} confianza)</div>",
                    unsafe_allow_html=True,
                )

        elif response.success and not response.objects:
            with col2:
                st.subheader("Detecciones")
                st.info("No se detectaron objetos reciclables en la imagen.")

        else:
            st.error(f"Error: {response.message}")


if __name__ == "__main__":
    main()