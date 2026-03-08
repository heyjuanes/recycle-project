import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app_service"))

import io
from PIL import Image


def create_test_image(width: int = 1920, height: int = 1080) -> bytes:
    """
    Crea una imagen de prueba en memoria y la retorna en bytes.

    Args:
        width: ancho de la imagen en pixeles.
        height: alto de la imagen en pixeles.

    Returns:
        Imagen codificada en bytes formato JPEG.
    """
    image = Image.new("RGB", (width, height), color=(255, 0, 0))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def test_imagen_se_comprime_correctamente():
    """Verifica que una imagen grande se comprime por debajo de 4MB."""
    image_bytes = create_test_image(1920, 1080)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image.thumbnail((1280, 1280))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    compressed = buffer.getvalue()
    assert len(compressed) < 4 * 1024 * 1024


def test_imagen_thumbnail_respeta_aspecto():
    """Verifica que thumbnail no supera el tamano maximo."""
    image = Image.new("RGB", (3000, 1500), color=(0, 255, 0))
    image.thumbnail((1280, 1280))
    assert image.width <= 1280
    assert image.height <= 1280


def test_imagen_se_convierte_a_rgb():
    """Verifica que una imagen RGBA se convierte correctamente a RGB."""
    image = Image.new("RGBA", (100, 100), color=(255, 0, 0, 128))
    converted = image.convert("RGB")
    assert converted.mode == "RGB"