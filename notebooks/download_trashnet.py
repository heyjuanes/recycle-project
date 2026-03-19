"""
download_trashnet.py

Descarga el modelo TrashNet v5 desde Roboflow Universe
y lo coloca en la raíz del proyecto como trashnet.pt

Uso:
    python notebooks/download_trashnet.py
"""

import shutil
from pathlib import Path
from roboflow import Roboflow


def download_model() -> None:
    """Descarga TrashNet v5 de Roboflow y lo mueve a la raíz."""
    rf = Roboflow(api_key="k521NcBshZ9kLY9QhUyA")

    project = rf.workspace("polygence-project").project(
        "trashnet-a-set-of-annotated-images-of-trash-that-can-be-used-for-object-detection"
    )

    version = project.version(5)

    # Forzar nombre de carpeta sin caracteres inválidos en Windows
    version.download("yolov8", location="trashnet-v5")

    # Buscar best.pt dentro de la carpeta descargada
    candidates = list(Path("trashnet-v5").rglob("best.pt"))

    if candidates:
        shutil.copy(candidates[0], "trashnet.pt")
        print(f"✅ Modelo guardado como trashnet.pt")
        print(f"   Copiado desde: {candidates[0]}")
    else:
        print("❌ No se encontró best.pt. Contenido de la carpeta:")
        for f in Path("trashnet-v5").rglob("*"):
            print(f"   {f}")


if __name__ == "__main__":
    download_model()