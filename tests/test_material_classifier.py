"""
test_material_classifier.py

Pruebas unitarias para el modulo material_classifier.py.

Verifica que el mapeo de clases TrashNet a categorias de material reciclable
funciona correctamente para las 6 clases del modelo, asi como para clases
desconocidas y variaciones de mayusculas.

Clases del modelo TrashNet v5:
    glass, paper, cardboard, plastic, metal, trash

Ejecutar con:
    pytest tests/test_material_classifier.py -v
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "inference_service"))

from material_classifier import get_material, get_material_info  # type: ignore


# ── Pruebas de get_material ───────────────────────────────────────────────────

def test_glass_es_vidrio():
    """Verifica que glass se clasifica como vidrio."""
    assert get_material("glass") == "vidrio"


def test_paper_es_papel():
    """Verifica que paper se clasifica como papel."""
    assert get_material("paper") == "papel"


def test_cardboard_es_carton():
    """Verifica que cardboard se clasifica como carton."""
    assert get_material("cardboard") == "carton"


def test_plastic_es_plastico():
    """Verifica que plastic se clasifica como plastico."""
    assert get_material("plastic") == "plastico"


def test_metal_es_metal():
    """Verifica que metal se clasifica como metal."""
    assert get_material("metal") == "metal"


def test_trash_es_basura():
    """Verifica que trash se clasifica como basura."""
    assert get_material("trash") == "basura"


def test_clase_desconocida():
    """Verifica que una clase no mapeada retorna no clasificado."""
    assert get_material("dinosaur") == "no clasificado"


def test_case_insensitive():
    """Verifica que el clasificador no distingue mayusculas de minusculas."""
    assert get_material("GLASS") == "vidrio"
    assert get_material("Plastic") == "plastico"
    assert get_material("METAL") == "metal"


# ── Pruebas de get_material_info ──────────────────────────────────────────────

def test_plastic_contenedor_amarillo():
    """Verifica que plastic tiene contenedor Amarillo."""
    info = get_material_info("plastic")
    assert info["contenedor"] == "Amarillo"


def test_glass_es_reciclable():
    """Verifica que glass es reciclable."""
    info = get_material_info("glass")
    assert info["reciclable"] is True


def test_trash_no_es_reciclable():
    """Verifica que trash no es reciclable."""
    info = get_material_info("trash")
    assert info["reciclable"] is False


def test_clase_desconocida_no_es_reciclable():
    """Verifica que una clase desconocida retorna reciclable False."""
    info = get_material_info("dinosaur")
    assert info["reciclable"] is False
    assert info["contenedor"] == "Desconocido"