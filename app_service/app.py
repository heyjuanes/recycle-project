import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "proto"))

import io
import grpc
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

import recycling_pb2
import recycling_pb2_grpc

GRPC_HOST = "localhost:50051"

MATERIAL_COLORS = {
    "plastico": "#3498DB",
    "metal": "#E74C3C",
    "vidrio": "#2ECC71",
    "carton": "#F39C12",
    "no clasificado": "#95A5A6",
}


def get_stub():
    """Crea y retorna el stub gRPC conectado al inference_service."""
    channel = grpc.insecure_channel(GRPC_HOST)
    return recycling_pb2_grpc.RecyclingInferenceStub(channel)


def draw_boxes(image: Image.Image, objects: list) -> Image.Image:
    """
    Dibuja bounding boxes sobre la imagen con clase y material.

    Args:
        image: imagen PIL original.
        objects: lista de DetectedObject retornados por gRPC.

    Returns:
        Imagen PIL con bounding boxes dibujados.
    """
    draw = ImageDraw.Draw(image)
    for obj in objects:
        color = MATERIAL_COLORS.get(obj.material, "#95A5A6")
        draw.rectangle([obj.x1, obj.y1, obj.x2, obj.y2], outline=color, width=3)
        label = f"{obj.class_name} ({obj.material}) {obj.confidence:.0%}"
        draw.rectangle(
            [obj.x1, obj.y1 - 20, obj.x1 + len(label) * 7, obj.y1],
            fill=color,
        )
        draw.text((obj.x1 + 2, obj.y1 - 18), label, fill="white")
    return image


def run_detection(image_bytes: bytes):
    """
    Envia imagen al inference_service via gRPC y retorna la respuesta.

    Args:
        image_bytes: imagen codificada en bytes.

    Returns:
        DetectionResponse con los objetos detectados.
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image.thumbnail((1280, 1280))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    compressed = buffer.getvalue()

    stub = get_stub()
    request = recycling_pb2.DetectionRequest(image_data=compressed)
    return stub.DetectObjects(request)


def main():
    """Funcion principal de la interfaz Streamlit."""
    st.set_page_config(
        page_title="Detector de Reciclaje",
        page_icon="♻️",
        layout="wide",
    )

    st.title("♻️ Sistema de Detección de Material Reciclable")
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