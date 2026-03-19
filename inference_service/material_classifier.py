"""
material_classifier.py

Modulo de reglas de negocio para clasificacion de material reciclable.
Mapea clases detectadas por YOLOv8 entrenado en TrashNet a categorias
de material reciclable con informacion de contenedor y reciclabilidad.
"""

# Mapa de clases del modelo TrashNet a informacion de reciclaje
MATERIAL_MAP = {
    "glass":     "vidrio",
    "paper":     "papel",
    "cardboard": "carton",
    "plastic":   "plastico",
    "metal":     "metal",
    "trash":     "basura",
}

# Informacion extendida por categoria de material
MATERIAL_INFO = {
    "vidrio":   {"contenedor": "Verde",    "reciclable": True},
    "papel":    {"contenedor": "Azul",     "reciclable": True},
    "carton":   {"contenedor": "Azul",     "reciclable": True},
    "plastico": {"contenedor": "Amarillo", "reciclable": True},
    "metal":    {"contenedor": "Gris",     "reciclable": True},
    "basura":   {"contenedor": "Negro",    "reciclable": False},
}


def get_material(class_name: str) -> str:
    """
    Retorna la categoria de material reciclable para una clase del modelo.

    Convierte el nombre de la clase a minusculas antes de buscar en el mapa
    para garantizar que la busqueda sea insensible a mayusculas.

    Args:
        class_name (str): nombre de la clase detectada por YOLOv8.
            Ejemplos: 'glass', 'plastic', 'cardboard'.

    Returns:
        str: categoria del material reciclable. Puede ser:
            - 'vidrio'
            - 'papel'
            - 'carton'
            - 'plastico'
            - 'metal'
            - 'basura'
            - 'no clasificado' si la clase no esta en el mapa.

    Examples:
        >>> get_material('plastic')
        'plastico'
        >>> get_material('GLASS')
        'vidrio'
        >>> get_material('dinosaur')
        'no clasificado'
    """
    return MATERIAL_MAP.get(class_name.lower(), "no clasificado")


def get_material_info(class_name: str) -> dict:
    """
    Retorna informacion extendida de reciclaje para una clase del modelo.

    Args:
        class_name (str): nombre de la clase detectada por YOLOv8.
            Ejemplos: 'metal', 'trash', 'paper'.

    Returns:
        dict: diccionario con las claves:
            - 'contenedor' (str): color del contenedor de reciclaje.
            - 'reciclable' (bool): indica si el material es reciclable.
            Retorna valores por defecto si la clase no esta en el mapa.

    Examples:
        >>> get_material_info('plastic')
        {'contenedor': 'Amarillo', 'reciclable': True}
        >>> get_material_info('trash')
        {'contenedor': 'Negro', 'reciclable': False}
        >>> get_material_info('dinosaur')
        {'contenedor': 'Desconocido', 'reciclable': False}
    """
    material = get_material(class_name)
    return MATERIAL_INFO.get(material, {"contenedor": "Desconocido", "reciclable": False})