"""
material_classifier.py

Modulo de reglas de negocio para clasificacion de material reciclable.
Mapea clases de objetos detectados por YOLOv8 a categorias de material.
"""

# Mapa de clases YOLO a categorias de material reciclable
MATERIAL_MAP = {
    # Plastico
    "bottle": "plastico",
    "cup": "plastico",
    "bowl": "plastico",
    "chair": "plastico",
    "keyboard": "plastico",
    "remote": "plastico",
    "cell phone": "plastico",
    # Metal
    "can": "metal",
    "fork": "metal",
    "knife": "metal",
    "spoon": "metal",
    "scissors": "metal",
    "clock": "metal",
    # Carton
    "book": "carton",
    "laptop": "carton",
    "mouse": "carton",
    # Vidrio
    "wine glass": "vidrio",
    "vase": "vidrio",
}


def get_material(class_name: str) -> str:
    """
    Retorna la categoria de material reciclable para una clase YOLO.

    Convierte el nombre de la clase a minusculas antes de buscar en el mapa
    para garantizar que la busqueda sea insensible a mayusculas.

    Args:
        class_name (str): nombre de la clase detectada por YOLOv8.
            Ejemplos: 'bottle', 'can', 'wine glass'.

    Returns:
        str: categoria del material reciclable. Puede ser:
            - 'plastico'
            - 'metal'
            - 'vidrio'
            - 'carton'
            - 'no clasificado' si la clase no esta en el mapa.

    Examples:
        >>> get_material('bottle')
        'plastico'
        >>> get_material('BOTTLE')
        'plastico'
        >>> get_material('dinosaur')
        'no clasificado'
    """
    return MATERIAL_MAP.get(class_name.lower(), "no clasificado")