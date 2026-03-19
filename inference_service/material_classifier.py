"""
material_classifier.py

Modulo de reglas de negocio para clasificacion de material reciclable.
Mapea clases detectadas por YOLOv8n (COCO) a categorias de material
reciclable con informacion de contenedor y reciclabilidad.

El modelo YOLOv8n fue entrenado en COCO (80 clases). Este modulo actua
como capa de reglas de negocio que traduce esas clases a materiales
reciclables especificos del dominio de la gestion de residuos.
"""

# ── Mapa de clases COCO → material reciclable ─────────────────────────────────
MATERIAL_MAP = {
    # Plástico
    "bottle":        "plastico",
    "cup":           "plastico",
    "bowl":          "plastico",
    "cell phone":    "plastico",
    "remote":        "plastico",
    "frisbee":       "plastico",
    "toothbrush":    "plastico",

    # Metal
    "scissors":      "metal",
    "knife":         "metal",
    "fork":          "metal",
    "spoon":         "metal",
    "can":           "metal",

    # Vidrio
    "wine glass":    "vidrio",
    "vase":          "vidrio",

    # Cartón / Papel
    "book":          "carton",
    "laptop":        "carton",
    "keyboard":      "carton",
    "mouse":         "carton",
    "clock":         "carton",

    # Orgánico
    "banana":        "organico",
    "apple":         "organico",
    "orange":        "organico",
    "broccoli":      "organico",
    "carrot":        "organico",
    "sandwich":      "organico",
    "pizza":         "organico",
    "cake":          "organico",
    "hot dog":       "organico",
    "donut":         "organico",

    # Basura general
    "suitcase":      "basura",
    "handbag":       "basura",
    "backpack":      "basura",
    "umbrella":      "basura",
    "tie":           "basura",
    "toaster":       "basura",
    "hair drier":    "basura",
    "trash":         "basura",
    "waste":         "basura",

    # Aliases del modelo gianlucasposito (clases en ingles capitalizadas)
    "glass":         "vidrio",
    "paper":         "carton",
    "cardboard":     "carton",
    "plastic":       "plastico",
    "metal":         "metal",
}

# ── Información extendida por categoría de material ───────────────────────────
MATERIAL_INFO = {
    "vidrio":    {"contenedor": "Verde",     "reciclable": True},
    "papel":     {"contenedor": "Azul",      "reciclable": True},
    "carton":    {"contenedor": "Azul",      "reciclable": True},
    "plastico":  {"contenedor": "Amarillo",  "reciclable": True},
    "metal":     {"contenedor": "Gris",      "reciclable": True},
    "organico":  {"contenedor": "Cafe",      "reciclable": True},
    "basura":    {"contenedor": "Negro",     "reciclable": False},
}


def get_material(class_name: str) -> str:
    """
    Retorna la categoria de material reciclable para una clase del modelo.

    Convierte el nombre de la clase a minusculas antes de buscar en el mapa
    para garantizar que la busqueda sea insensible a mayusculas.

    Args:
        class_name (str): nombre de la clase detectada por el modelo.
            Ejemplos: 'bottle', 'wine glass', 'banana', 'Glass', 'Plastic'.

    Returns:
        str: categoria del material reciclable. Puede ser:
            - 'vidrio'
            - 'papel'
            - 'carton'
            - 'plastico'
            - 'metal'
            - 'organico'
            - 'basura'
            - 'no clasificado' si la clase no esta en el mapa.

    Examples:
        >>> get_material('bottle')
        'plastico'
        >>> get_material('wine glass')
        'vidrio'
        >>> get_material('banana')
        'organico'
        >>> get_material('waste')
        'basura'
        >>> get_material('dinosaur')
        'no clasificado'
    """
    return MATERIAL_MAP.get(class_name.lower(), "no clasificado")


def get_material_info(class_name: str) -> dict:
    """
    Retorna informacion extendida de reciclaje para una clase del modelo.

    Args:
        class_name (str): nombre de la clase detectada por el modelo.
            Ejemplos: 'bottle', 'banana', 'wine glass', 'waste'.

    Returns:
        dict: diccionario con las claves:
            - 'contenedor' (str): color del contenedor de reciclaje.
            - 'reciclable' (bool): indica si el material es reciclable.
            Retorna valores por defecto si la clase no esta en el mapa.

    Examples:
        >>> get_material_info('bottle')
        {'contenedor': 'Amarillo', 'reciclable': True}
        >>> get_material_info('banana')
        {'contenedor': 'Cafe', 'reciclable': True}
        >>> get_material_info('waste')
        {'contenedor': 'Negro', 'reciclable': False}
        >>> get_material_info('dinosaur')
        {'contenedor': 'Desconocido', 'reciclable': False}
    """
    material = get_material(class_name)
    return MATERIAL_INFO.get(material, {"contenedor": "Desconocido", "reciclable": False})