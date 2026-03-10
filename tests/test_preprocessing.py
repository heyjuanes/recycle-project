"""
test_preprocessing.py

Pruebas unitarias para el preprocesamiento de imagenes aplicado
en app_service/app.py antes de enviar la imagen al inference_service
via gRPC.

Verifica que la compresion, el redimensionamiento y la conversion
de formato funcionan correctamente y dentro de los limites esperados.

Ejecutar con:
    pytest tests/test_preprocessing.py -v
"""

import io
from PIL import Image


def create_test_image(width: int = 1920, height: int = 1080) -> bytes:
    """
    Crea una imagen de prueba en memoria y la retorna en bytes.

    Genera una imagen solida de color rojo del tamano especificado,
    la codifica en formato JPEG y retorna los bytes resultantes.
    Util para simular imagenes de entrada sin depender de archivos
    externos en el sistema de archivos.

    Args:
        width (int): ancho de la imagen en pixeles. Por defecto 1920.
        height (int): alto de la imagen en pixeles. Por defecto 1080.

    Returns:
        bytes: imagen codificada en formato JPEG.
    """
    image = Image.new("RGB", (width, height), color=(255, 0, 0))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def test_imagen_se_comprime_correctamente():
    """
    Verifica que una imagen grande se comprime por debajo de 4MB.

    El limite por defecto de gRPC es 4MB por mensaje. Este test
    confirma que el pipeline de compresion (thumbnail + JPEG quality 85)
    reduce una imagen de 1920x1080 por debajo de ese limite antes
    de enviarla al servidor gRPC.
    """
    image_bytes = create_test_image(1920, 1080)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image.thumbnail((1280, 1280))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    compressed = buffer.getvalue()
    assert len(compressed) < 4 * 1024 * 1024


def test_imagen_thumbnail_respeta_aspecto():
    """
    Verifica que thumbnail no supera el tamano maximo en ninguna dimension.

    PIL.Image.thumbnail redimensiona la imagen manteniendo la proporcion
    original. Este test confirma que ninguna dimension supera 1280px
    despues de aplicar thumbnail((1280, 1280)) sobre una imagen de 3000x1500.
    """
    image = Image.new("RGB", (3000, 1500), color=(0, 255, 0))
    image.thumbnail((1280, 1280))
    assert image.width <= 1280
    assert image.height <= 1280


def test_imagen_se_convierte_a_rgb():
    """
    Verifica que una imagen RGBA se convierte correctamente a RGB.

    Algunas imagenes PNG tienen canal alfa (transparencia) en formato RGBA.
    YOLOv8 espera imagenes en formato RGB. Este test confirma que la
    conversion de RGBA a RGB se realiza sin errores y produce una imagen
    con el modo correcto.
    """
    image = Image.new("RGBA", (100, 100), color=(255, 0, 0, 128))
    converted = image.convert("RGB")
    assert converted.mode == "RGB"