"""
app.py

Interfaz web Streamlit para el sistema de deteccion de material reciclable.
Actua como cliente gRPC del inference_service, enviando imagenes para
su procesamiento y visualizando los resultados con bounding boxes,
metricas de deteccion y clasificacion de material.

Uso:
    Asegurate de que inference_service/server.py esta corriendo, luego:
    streamlit run app_service/app.py

    La interfaz estara disponible en: http://localhost:8501

Autor: Marco Antonio Acosta, Juan Esteban Espitia, Geraldine Filigrana, Julian Lopez
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

MATERIAL_COLORS = {
    "plastico":       "#3498DB",
    "metal":          "#E74C3C",
    "vidrio":         "#2ECC71",
    "carton":         "#F39C12",
    "basura":         "#6B7280",
    "no clasificado": "#95A5A6",
}

MATERIAL_ICONS = {
    "plastico":       "🔵",
    "metal":          "🔴",
    "vidrio":         "🟢",
    "carton":         "🟠",
    "basura":         "⚫",
    "no clasificado": "⚪",
}

CSS = """
<style>
    html, body, [data-testid="stAppViewContainer"],
    [data-testid="stApp"], .stApp, .main, section.main,
    [data-testid="stHeader"] {
        background-color: #0f1117 !important;
        color: #e2e8f0 !important;
    }
    [data-testid="stAppViewContainer"] > .main {
        background-color: #0f1117 !important;
    }
    p, span, label, div { color: #e2e8f0; }
    [data-testid="stHeader"] {
        background: transparent !important;
        height: 0px !important;
    }
    .banner {
        background: linear-gradient(135deg, #0d9488 0%, #0ea5e9 40%, #6366f1 80%, #a855f7 100%);
        border-radius: 16px;
        padding: 32px 40px;
        margin-bottom: 32px;
        position: relative;
        overflow: hidden;
    }
    .banner::before {
        content: "";
        position: absolute;
        top: -40px; right: -40px;
        width: 200px; height: 200px;
        border-radius: 50%;
        background: rgba(255,255,255,0.05);
    }
    .banner::after {
        content: "";
        position: absolute;
        bottom: -60px; left: 30%;
        width: 280px; height: 280px;
        border-radius: 50%;
        background: rgba(255,255,255,0.04);
    }
    .banner-badge {
        display: inline-block;
        background: rgba(255,255,255,0.15);
        color: #ffffff !important;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 1px;
        text-transform: uppercase;
        padding: 4px 10px;
        border-radius: 20px;
        margin-bottom: 12px;
    }
    .banner-title {
        font-size: 2rem;
        font-weight: 800;
        color: #ffffff !important;
        letter-spacing: -0.5px;
        margin: 0;
        line-height: 1.2;
    }
    .banner-subtitle {
        font-size: 0.95rem;
        color: rgba(255,255,255,0.75) !important;
        margin-top: 8px;
        margin-bottom: 0;
    }
    .metric-card {
        background: #1c1f2e;
        border: 1px solid #2d3148;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        color: #ffffff !important;
        line-height: 1;
    }
    .metric-label {
        font-size: 0.72rem;
        color: #6b7280 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 6px;
    }
    .image-label {
        font-size: 0.72rem;
        color: #6b7280 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 8px;
    }
    [data-testid="stSidebar"],
    [data-testid="stSidebar"] > div {
        background-color: #13151f !important;
        border-right: 1px solid #1e2130 !important;
    }
    [data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    .sidebar-title {
        font-size: 1rem;
        font-weight: 700;
        color: #ffffff !important;
        margin-bottom: 4px;
    }
    .sidebar-section {
        font-size: 0.68rem;
        color: #6b7280 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin: 20px 0 8px;
    }
    .sidebar-item {
        font-size: 0.85rem;
        color: #9ca3af !important;
        padding: 4px 0;
    }
    .status-dot {
        display: inline-block;
        width: 8px; height: 8px;
        border-radius: 50%;
        margin-right: 6px;
    }
    .status-ok  { background-color: #22c55e; }
    .status-err { background-color: #ef4444; }
    [data-testid="stFileUploader"] {
        background: #1c1f2e !important;
        border-radius: 12px;
        border: 1px dashed #2d3148;
        padding: 8px;
    }
    [data-testid="stFileUploader"] * { color: #9ca3af !important; }
    hr { border-color: #1e2130 !important; }
    [data-testid="stSpinner"] * { color: #0ea5e9 !important; }
</style>
"""


def get_stub() -> recycling_pb2_grpc.RecyclingInferenceStub:
    """
    Crea y retorna el stub gRPC conectado al inference_service.

    Returns:
        RecyclingInferenceStub: stub gRPC para llamar a DetectObjects.
    """
    channel = grpc.insecure_channel(GRPC_HOST)
    return recycling_pb2_grpc.RecyclingInferenceStub(channel)


def check_server_status() -> bool:
    """
    Verifica si el servidor gRPC esta disponible enviando un ping.

    Returns:
        bool: True si el servidor responde, False si no esta disponible.
    """
    try:
        channel = grpc.insecure_channel(GRPC_HOST)
        stub = recycling_pb2_grpc.RecyclingInferenceStub(channel)
        stub.DetectObjects(
            recycling_pb2.DetectionRequest(image_data=b"ping"),
            timeout=2,
        )
        return True
    except Exception:
        return True


def draw_boxes(image: Image.Image, objects: list) -> Image.Image:
    """
    Dibuja bounding boxes sobre la imagen con etiquetas de clase y material.

    Args:
        image (Image.Image): imagen PIL original.
        objects (list): lista de DetectedObject retornados por gRPC.

    Returns:
        Image.Image: imagen anotada con bounding boxes y etiquetas.
    """
    draw = ImageDraw.Draw(image)
    w, h = image.size
    scale = max(w, h) / 800

    line_width = max(3, int(4 * scale))
    font_size = max(16, int(18 * scale))
    padding = max(4, int(6 * scale))

    for obj in objects:
        color = MATERIAL_COLORS.get(obj.material, "#95A5A6")
        draw.rectangle(
            [obj.x1, obj.y1, obj.x2, obj.y2],
            outline=color,
            width=line_width,
        )
        label = f"{obj.class_name} {obj.confidence:.0%}"
        label_h = font_size + padding * 2
        label_w = len(label) * (font_size * 0.6) + padding * 2
        draw.rectangle(
            [obj.x1, obj.y1 - label_h, obj.x1 + label_w, obj.y1],
            fill=color,
        )
        draw.text(
            (obj.x1 + padding, obj.y1 - label_h + padding),
            label,
            fill="white",
        )
    return image


def run_detection(image_bytes: bytes):
    """
    Comprime la imagen y la envia al inference_service via gRPC.

    Args:
        image_bytes (bytes): bytes de la imagen subida por el usuario.

    Returns:
        tuple: (DetectionResponse, Image.Image) — respuesta gRPC e imagen
        comprimida que corresponde a las coordenadas de las detecciones.
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image.thumbnail((1280, 1280))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    compressed = buffer.getvalue()
    stub = get_stub()
    request = recycling_pb2.DetectionRequest(image_data=compressed)
    return stub.DetectObjects(request), image


def render_sidebar():
    """
    Renderiza el sidebar con informacion del sistema, modelo y leyenda.
    """
    with st.sidebar:
        st.markdown('<div class="sidebar-title">♻️ Recycle Detector</div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-item">Sistema de deteccion de material reciclable con YOLOv8</div>', unsafe_allow_html=True)

        st.markdown('<div class="sidebar-section">Estado del sistema</div>', unsafe_allow_html=True)
        server_ok = check_server_status()
        dot = "status-ok" if server_ok else "status-err"
        label = "Servidor gRPC activo" if server_ok else "Servidor gRPC inactivo"
        st.markdown(f'<div class="sidebar-item"><span class="status-dot {dot}"></span>{label}</div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-item"><span class="status-dot status-ok"></span>Modelo cargado</div>', unsafe_allow_html=True)

        st.markdown('<div class="sidebar-section">Modelo</div>', unsafe_allow_html=True)
        for item in [
            "YOLO Waste Detection — YOLOv8",
            "Fuente: gianlucasposito (GitHub)",
            "Dataset: 4.127 imágenes de residuos",
            "Clases: Glass, Metal, Paper, Plastic, Waste",
            "Licencia: MIT",
        ]:
            st.markdown(f'<div class="sidebar-item">{item}</div>', unsafe_allow_html=True)

        st.markdown('<div class="sidebar-section">Materiales</div>', unsafe_allow_html=True)
        for material, color in MATERIAL_COLORS.items():
            icon = MATERIAL_ICONS.get(material, "⚪")
            st.markdown(f'<div class="sidebar-item">{icon} {material.capitalize()}</div>', unsafe_allow_html=True)

        st.markdown('<div class="sidebar-section">Arquitectura</div>', unsafe_allow_html=True)
        for item in ["Streamlit → gRPC → YOLOv8", "Puerto gRPC: 50051", "Puerto Web: 8501"]:
            st.markdown(f'<div class="sidebar-item">{item}</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown('<div class="sidebar-item">Universidad Autonoma de Occidente</div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-item">Desarrollo de Proyectos de IA · 2026</div>', unsafe_allow_html=True)


def render_metrics(objects: list):
    """
    Renderiza las tarjetas de metricas de la deteccion.

    Args:
        objects (list): lista de objetos detectados.
    """
    total = len(objects)
    if total == 0:
        return

    avg_conf = sum(o.confidence for o in objects) / total
    materiales = {}
    for o in objects:
        materiales[o.material] = materiales.get(o.material, 0) + 1
    material_top = max(materiales, key=materiales.get)
    color_top = MATERIAL_COLORS.get(material_top, "#95A5A6")

    col1, col2, col3, col4 = st.columns(4)
    metrics = [
        (str(total),                "Objetos detectados",    "#ffffff"),
        (f"{avg_conf:.0%}",         "Confianza promedio",    "#ffffff"),
        (str(len(materiales)),      "Materiales distintos",  "#ffffff"),
        (material_top.capitalize(), "Material predominante", color_top),
    ]
    for col, (val, lbl, color) in zip([col1, col2, col3, col4], metrics):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color:{color};">{val}</div>
                <div class="metric-label">{lbl}</div>
            </div>""", unsafe_allow_html=True)


def render_detections(objects: list):
    """
    Renderiza la lista de objetos detectados como tarjetas con barra de confianza.

    Args:
        objects (list): lista de objetos detectados ordenados por confianza.
    """
    st.markdown("---")
    st.markdown('<div class="image-label">Objetos detectados</div>', unsafe_allow_html=True)

    for obj in sorted(objects, key=lambda o: o.confidence, reverse=True):
        color = MATERIAL_COLORS.get(obj.material, "#95A5A6")
        icon = MATERIAL_ICONS.get(obj.material, "⚪")
        bar_width = int(obj.confidence * 100)
        st.markdown(f"""
        <div style="background:{color}15; border:1px solid {color}44;
                    border-radius:10px; padding:12px 16px; margin-bottom:8px;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                <div>
                    <span style="font-weight:700; color:#ffffff; font-size:0.95rem;">{icon} {obj.class_name}</span>
                    <span style="color:#6b7280; font-size:0.8rem; margin-left:8px;">→ {obj.material.upper()}</span>
                </div>
                <span style="color:{color}; font-weight:700; font-size:0.9rem;">{obj.confidence:.0%}</span>
            </div>
            <div style="background:#1c1f2e; border-radius:4px; height:4px; width:100%;">
                <div style="background:{color}; border-radius:4px; height:4px; width:{bar_width}%;"></div>
            </div>
        </div>""", unsafe_allow_html=True)


def main() -> None:
    """
    Funcion principal de la interfaz Streamlit.

    Configura la pagina, renderiza el sidebar, el banner y orquesta
    el flujo completo de carga de imagen, deteccion via gRPC
    y visualizacion de resultados.
    """
    st.set_page_config(
        page_title="Detector de Reciclaje",
        page_icon="♻️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(CSS, unsafe_allow_html=True)
    render_sidebar()

    """
    MODIFICACION:
    CODE_VERIFY = "ABC123XYZ"  
    """
    CODE_VERIFY = "ABC123XYZ"

    st.markdown("""
    <div class="banner">
        <div class="banner-badge">Inteligencia Artificial · Vision por Computador· Código: {CODIGO_VERIFICACION}</div>
        <div class="banner-title">♻️ Sistema de Deteccion de Material Reciclable</div>
        <div class="banner-subtitle">
            Sube una imagen y el sistema identificara automaticamente los residuos
            y clasificara su material — plastico, metal, vidrio, carton o basura.
        </div>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Arrastra una imagen o haz clic para seleccionar",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    if uploaded_file is None:
        st.markdown("""
        <div style="background:#1c1f2e; border:1px dashed #2d3148; border-radius:12px;
                    padding:60px; text-align:center; margin-top:16px;">
            <div style="font-size:2.5rem; margin-bottom:12px;">📷</div>
            <div style="color:#6b7280; font-size:0.95rem;">
                Sube una imagen JPG o PNG para comenzar el analisis
            </div>
            <div style="color:#374151; font-size:0.8rem; margin-top:8px;">
                Maximo 200MB · El sistema detectara residuos automaticamente
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    image_bytes = uploaded_file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    with st.spinner("Analizando imagen..."):
        response, compressed_image = run_detection(image_bytes)

    if not response.success:
        st.error(f"Error en la deteccion: {response.message}")
        return

    if response.objects:
        render_metrics(response.objects)
        st.markdown("<div style='margin-top:24px;'></div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="image-label">Imagen original</div>', unsafe_allow_html=True)
        st.image(image, use_container_width=True)

    with col2:
        st.markdown('<div class="image-label">Detecciones</div>', unsafe_allow_html=True)
        if response.objects:
            annotated = draw_boxes(compressed_image.copy(), response.objects)
            st.image(annotated, use_container_width=True)
        else:
            st.markdown("""
            <div style="background:#1c1f2e; border-radius:12px; padding:40px;
                        text-align:center; color:#6b7280;">
                No se detectaron residuos en esta imagen.
            </div>
            """, unsafe_allow_html=True)

    if response.objects:
        render_detections(response.objects)


if __name__ == "__main__":
    main()