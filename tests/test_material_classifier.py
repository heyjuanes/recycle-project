import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "inference_service"))

from material_classifier import get_material # type: ignore


def test_bottle_es_plastico():
    """Verifica que bottle se clasifica como plastico."""
    assert get_material("bottle") == "plastico"


def test_can_es_metal():
    """Verifica que can se clasifica como metal."""
    assert get_material("can") == "metal"


def test_wine_glass_es_vidrio():
    """Verifica que wine glass se clasifica como vidrio."""
    assert get_material("wine glass") == "vidrio"


def test_book_es_carton():
    """Verifica que book se clasifica como carton."""
    assert get_material("book") == "carton"


def test_clase_desconocida():
    """Verifica que una clase desconocida retorna no clasificado."""
    assert get_material("dinosaur") == "no clasificado"


def test_case_insensitive():
    """Verifica que el clasificador no distingue mayusculas."""
    assert get_material("BOTTLE") == "plastico"
    assert get_material("Bottle") == "plastico"