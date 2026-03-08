MATERIAL_MAP = {
    "bottle": "plastico",
    "cup": "plastico",
    "bowl": "plastico",
    "chair": "plastico",
    "keyboard": "plastico",
    "remote": "plastico",
    "cell phone": "plastico",
    "can": "metal",
    "fork": "metal",
    "knife": "metal",
    "spoon": "metal",
    "scissors": "metal",
    "book": "carton",
    "laptop": "carton",
    "mouse": "carton",
    "clock": "carton",
    "wine glass": "vidrio",
    "vase": "vidrio",
}


def get_material(class_name: str) -> str:
    """
    Retorna la categoria de material reciclable para una clase YOLO.

    Args:
        class_name: nombre de la clase detectada por YOLOv8.

    Returns:
        Categoria del material: plastico, metal, vidrio, carton
        o 'no clasificado' si no esta en el mapa.
    """
    return MATERIAL_MAP.get(class_name.lower(), "no clasificado")