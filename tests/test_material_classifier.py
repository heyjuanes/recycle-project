"""
test_material_classifier.py

Pruebas unitarias para el modulo material_classifier.py.

Verifica que el mapeo de clases COCO a categorias de material reciclable
funciona correctamente para las clases principales, asi como para clases
desconocidas y variaciones de mayusculas.

El modelo YOLOv8n fue entrenado en COCO (80 clases). El clasificador
actua como capa de reglas de negocio que traduce esas clases a materiales
reciclables del dominio de la gestion de residuos.

Ejecutar con:
    pytest tests/test_material_classifier.py -v
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "inference_service"))

from material_classifier import get_material, get_material_info  # type: ignore


# ── Pruebas de get_material ───────────────────────────────────────────────────

def test_bottle_es_plastico():
    """Verifica que bottle se clasifica como plastico."""
    assert get_material("bottle") == "plastico"


def test_wine_glass_es_vidrio():
    """Verifica que wine glass se clasifica como vidrio."""
    assert get_material("wine glass") == "vidrio"


def test_book_es_carton():
    """Verifica que book se clasifica como carton."""
    assert get_material("book") == "carton"


def test_scissors_es_metal():
    """Verifica que scissors se clasifica como metal."""
    assert get_material("scissors") == "metal"


def test_banana_es_organico():
    """Verifica que banana se clasifica como organico."""
    assert get_material("banana") == "organico"


def test_waste_es_basura():
    """Verifica que waste se clasifica como basura."""
    assert get_material("waste") == "basura"


def test_clase_desconocida():
    """Verifica que una clase no mapeada retorna no clasificado."""
    assert get_material("dinosaur") == "no clasificado"


def test_case_insensitive():
    """Verifica que el clasificador no distingue mayusculas de minusculas."""
    assert get_material("BOTTLE") == "plastico"
    assert get_material("Banana") == "organico"
    assert get_material("SCISSORS") == "metal"


# ── Pruebas de get_material_info ──────────────────────────────────────────────

def test_bottle_contenedor_amarillo():
    """Verifica que bottle tiene contenedor Amarillo."""
    info = get_material_info("bottle")
    assert info["contenedor"] == "Amarillo"


def test_wine_glass_es_reciclable():
    """Verifica que wine glass es reciclable."""
    info = get_material_info("wine glass")
    assert info["reciclable"] is True


def test_suitcase_no_es_reciclable():
    """Verifica que suitcase no es reciclable."""
    info = get_material_info("suitcase")
    assert info["reciclable"] is False


def test_clase_desconocida_no_es_reciclable():
    """Verifica que una clase desconocida retorna reciclable False."""
    info = get_material_info("dinosaur")
    assert info["reciclable"] is False
    assert info["contenedor"] == "Desconocido"