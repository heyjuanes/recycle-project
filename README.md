# ♻️ Recycle Project — Sistema Inteligente de Detección de Material Reciclable

![MIT License](https://img.shields.io/badge/License-MIT-green.svg)
![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Waste%20Detection-purple.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30-red.svg)
![gRPC](https://img.shields.io/badge/gRPC-1.60-orange.svg)
![MLflow](https://img.shields.io/badge/MLflow-2.10-blue.svg)
![uv](https://img.shields.io/badge/uv-package%20manager-black.svg)

**Asignatura:** Desarrollo de Proyectos de IA  
**Docente:** Jan Polanco Velasco  
**Universidad:** Universidad Autónoma de Occidente — Facultad de Ingeniería  
**Equipo:** Marco Antonio Acosta · Juan Esteban Espitia · Geraldine Filigrana Sánchez · Julian David Lopez  
**Periodo:** Marzo 2026

---

## 📋 Tabla de Contenido

- [Descripción del Proyecto](#descripción-del-proyecto)
- [Arquitectura del Sistema](#arquitectura-del-sistema)
- [Estructura del Repositorio](#estructura-del-repositorio)
- [Tecnologías y Dependencias](#tecnologías-y-dependencias)
- [Requisitos Previos](#requisitos-previos)
- [Instalación y Ejecución Local](#instalación-y-ejecución-local)
- [Ejecución con Docker](#ejecución-con-docker)
- [Documentación de Módulos y Funciones](#documentación-de-módulos-y-funciones)
- [El Modelo — YOLO Waste Detection](#el-modelo--yolo-waste-detection)
- [Gestión del Modelo con MLflow](#gestión-del-modelo-con-mlflow)
- [Comunicación gRPC](#comunicación-grpc)
- [Pruebas Unitarias](#pruebas-unitarias)
- [Gestión del Proyecto — Tablero Kanban](#gestión-del-proyecto--tablero-kanban)
- [Decisiones de Diseño](#decisiones-de-diseño)
- [Limitaciones Conocidas](#limitaciones-conocidas)
- [Licencia](#licencia)

---

## Descripción del Proyecto

Este proyecto implementa una aplicación web de visión por computador que permite al usuario cargar una imagen, detectar automáticamente objetos reciclables presentes en ella y clasificar cada objeto según su tipo de material (vidrio, papel, plástico, metal o basura general).

El sistema utiliza **YOLO Waste Detection**, un modelo YOLOv8 entrenado específicamente en imágenes de residuos reales, publicado por [gianlucasposito](https://github.com/gianlucasposito/YOLO-Waste-Detection) en GitHub bajo licencia MIT. A diferencia de modelos de propósito general como YOLOv8n-COCO (entrenado en 80 clases genéricas), este modelo fue fine-tuneado exclusivamente sobre un dataset de 4.127 imágenes de residuos anotadas con bounding boxes, lo que le da mayor precisión en el dominio específico del reciclaje.

La aplicación tiene fines **educativos y demostrativos**, orientada a ilustrar cómo integrar un modelo de Machine Learning especializado en una arquitectura de microservicios real, siguiendo buenas prácticas de ingeniería de software.

### Problema que resuelve

En muchos entornos urbanos los usuarios no saben en qué contenedor depositar un objeto. Este sistema permite fotografiar el objeto y obtener de inmediato su categoría de material y el contenedor correspondiente, facilitando la separación correcta de residuos.

---

## Arquitectura del Sistema

El sistema está dividido en **dos microservicios independientes** que se comunican mediante **gRPC**:

```
┌─────────────────────────────────────────────────────┐
│                   USUARIO                           │
│         (navegador web en localhost:8501)           │
└──────────────────────┬──────────────────────────────┘
                       │ HTTP
┌──────────────────────▼──────────────────────────────┐
│              app_service (Streamlit)                │
│  - Recibe imagen del usuario                        │
│  - Comprime y codifica la imagen en bytes           │
│  - Actúa como CLIENTE gRPC                          │
│  - Visualiza resultados con bounding boxes          │
└──────────────────────┬──────────────────────────────┘
                       │ gRPC (localhost:50051)
┌──────────────────────▼──────────────────────────────┐
│           inference_service (servidor gRPC)         │
│  - Carga el modelo YOLO Waste Detection al iniciar  │
│  - Recibe imagen en bytes                           │
│  - Ejecuta inferencia con YOLOv8                    │
│  - Clasifica cada detección por material            │
│  - Retorna objetos detectados con coordenadas       │
└─────────────────────────────────────────────────────┘
```

### Flujo de datos paso a paso

1. El usuario sube una imagen JPG/PNG desde la interfaz Streamlit.
2. `app_service` lee los bytes de la imagen y la comprime a máximo 1280px para optimizar la transferencia.
3. `app_service` envía los bytes al `inference_service` mediante una llamada gRPC (`DetectObjects`).
4. `inference_service` decodifica los bytes, crea una imagen PIL y la pasa al modelo.
5. El modelo retorna las detecciones: clase, confianza y coordenadas del bounding box.
6. `inference_service` consulta `material_classifier` para obtener el material de cada clase.
7. `inference_service` empaqueta los resultados en un `DetectionResponse` y lo retorna vía gRPC.
8. `app_service` recibe la respuesta, dibuja los bounding boxes sobre la imagen y muestra los resultados.

---

## Estructura del Repositorio

```
recycle-project/
├── app_service/
│   ├── __init__.py
│   └── app.py                  # Interfaz Streamlit + cliente gRPC
├── inference_service/
│   ├── __init__.py
│   ├── server.py               # Servidor gRPC + lógica de inferencia YOLOv8
│   ├── material_classifier.py  # Módulo de reglas clase → material reciclable
│   └── test_client.py          # Cliente de prueba para verificar la conexión gRPC
├── proto/
│   ├── recycling.proto         # Definición del contrato gRPC
│   ├── recycling_pb2.py        # Código Python generado desde el .proto (mensajes)
│   └── recycling_pb2_grpc.py   # Código Python generado desde el .proto (servicios)
├── notebooks/
│   ├── download_trashnet.py      # Script de descarga del modelo
│   └── register_model_mlflow.py  # Registra el modelo en MLflow
├── data/
│   ├── raw/                    # Dataset original sin procesar
│   └── processed/              # Dataset preprocesado
├── tests/
│   ├── __init__.py
│   ├── sample_images/          # Imágenes de prueba por clase (glass, plastic, metal, paper, waste)
│   ├── test_material_classifier.py  # Pruebas del módulo de clasificación
│   └── test_preprocessing.py        # Pruebas del preprocesamiento de imágenes
├── reports/                    # Figuras y métricas del proyecto
├── src/
│   └── __init__.py
├── .vscode/
│   └── settings.json           # Configuración de rutas para VS Code
├── .gitignore
├── docker-compose.yml
├── Dockerfile
├── LICENSE
├── requirements.txt
└── README.md
```

> ⚠️ El archivo `trashnet.pt` no está en el repositorio (excluido por `.gitignore`). Descárgalo ejecutando:
> ```powershell
> Invoke-WebRequest -Uri "https://github.com/gianlucasposito/YOLO-Waste-Detection/raw/main/best_model.pt" -OutFile "trashnet.pt"
> ```

---

## Tecnologías y Dependencias

| Tecnología | Versión | Uso |
|---|---|---|
| Python | 3.11 | Lenguaje base |
| YOLOv8 (Ultralytics) | ≥ 8.0.0 | Detección de objetos reciclables |
| PyTorch | ≥ 2.0.0 | Framework de deep learning |
| Streamlit | ≥ 1.30.0 | Interfaz web |
| gRPC | ≥ 1.60.0 | Comunicación entre microservicios |
| grpcio-tools | ≥ 1.60.0 | Generación de código desde .proto |
| MLflow | ≥ 2.10.0 | Gestión del ciclo de vida del modelo |
| OpenCV | ≥ 4.9.0 | Procesamiento de imágenes |
| Pillow | ≥ 10.0.0 | Manipulación de imágenes |
| NumPy | ≥ 1.26.0 | Operaciones numéricas |
| Pandas | ≥ 2.0.0 | Manejo de datos tabulares |
| pytest | ≥ 8.0.0 | Pruebas unitarias |
| Docker | ≥ 24.0 | Contenerización |
| uv | latest | Gestor de paquetes y entornos virtuales |

---

## Requisitos Previos

- Python 3.11 o superior instalado y en el PATH
- [uv](https://docs.astral.sh/uv/) instalado (`pip install uv`)
- Git instalado
- Docker Desktop instalado (solo para ejecución con Docker)
- Al menos 2GB de espacio en disco (para el modelo y dependencias)

---

## Instalación y Ejecución Local

### 1. Clonar el repositorio

```bash
git clone https://github.com/heyjuanes/recycle-project.git
cd recycle-project
```

### 2. Crear y activar el entorno virtual con uv

**Windows (PowerShell):**
```powershell
uv venv --python 3.11
.venv\Scripts\activate
```

**macOS / Linux:**
```bash
uv venv --python 3.11
source .venv/bin/activate
```

### 3. Instalar dependencias

```powershell
uv pip install -r requirements.txt
```

### 4. Descargar el modelo

El archivo `trashnet.pt` no está en el repositorio. Descárgalo ejecutando:

```powershell
Invoke-WebRequest -Uri "https://github.com/gianlucasposito/YOLO-Waste-Detection/raw/main/best_model.pt" -OutFile "trashnet.pt"
```

### 5. Iniciar el servidor gRPC (inference_service)

Abre una terminal, activa el entorno virtual y ejecuta:

```powershell
python inference_service/server.py
```

Deberías ver:
```
Cargando modelo TrashNet v5...
Modelo TrashNet listo. Clases: ['Glass', 'Metal', 'Paper', 'Plastic', 'Waste']
Servidor gRPC corriendo en puerto 50051...
```

### 6. Iniciar la interfaz Streamlit (app_service)

Abre una **segunda terminal**, activa el entorno virtual y ejecuta:

```powershell
streamlit run app_service/app.py
```

La aplicación se abrirá automáticamente en: **http://localhost:8501**

### 7. Usar la aplicación

1. Abre http://localhost:8501 en tu navegador.
2. Haz clic en **"Browse files"** o arrastra una imagen JPG/PNG.
3. El sistema detectará automáticamente los residuos y clasificará su material.
4. Verás la imagen anotada con bounding boxes y una tabla de resultados.

Puedes usar las imágenes de prueba incluidas en `tests/sample_images/` para verificar el funcionamiento del sistema en cada una de las 5 clases detectables.

---

## Ejecución con Docker

### Construir la imagen

```bash
docker-compose build
```

### Ejecutar la aplicación

```bash
docker-compose up
```

La aplicación estará disponible en: **http://localhost:8501**

### Detener la aplicación

```bash
docker-compose down
```

### Notas sobre Docker

- El `Dockerfile` usa la imagen base `python:3.11-slim` para minimizar el tamaño.
- El servidor gRPC y la interfaz Streamlit se inician juntos dentro del mismo contenedor.
- El puerto `8501` expone la interfaz web y el puerto `50051` expone el servidor gRPC.
- Los experimentos de MLflow se persisten en el volumen `./mlruns`.
- El archivo `trashnet.pt` debe existir en la raíz **antes** de construir la imagen Docker.

---

## Documentación de Módulos y Funciones

### `inference_service/material_classifier.py`

Módulo de reglas de negocio que mapea las clases detectadas por el modelo a categorías de material reciclable con información de contenedor y reciclabilidad.

| Clase del modelo | Material | Contenedor | Reciclable |
|---|---|---|---|
| `glass` | Vidrio | Verde ♻️ | Sí |
| `paper` | Papel | Azul ♻️ | Sí |
| `cardboard` | Cartón | Azul ♻️ | Sí |
| `plastic` | Plástico | Amarillo ♻️ | Sí |
| `metal` | Metal | Gris ♻️ | Sí |
| `trash` / `waste` | Basura general | Negro 🗑️ | No |

**`get_material(class_name: str) -> str`** — retorna el material en español. Insensible a mayúsculas. Retorna `"no clasificado"` si la clase no está en el mapa.

**`get_material_info(class_name: str) -> dict`** — retorna diccionario con `contenedor` y `reciclable`.

---

### `inference_service/server.py`

Servidor gRPC que expone el servicio de inferencia YOLOv8.

- **`__init__`**: carga `trashnet.pt` una sola vez al arrancar e imprime las clases disponibles.
- **`DetectObjects`**: decodifica imagen desde bytes, ejecuta inferencia, consulta `material_classifier` y retorna `DetectionResponse` con bounding boxes, clases y materiales.
- **`serve`**: arranca el servidor en puerto `50051` con límite de 20MB y 4 workers.

---

### `app_service/app.py`

Interfaz web Streamlit que actúa como cliente gRPC.

- **`get_stub`**: crea el canal gRPC hacia `localhost:50051`.
- **`draw_boxes`**: dibuja bounding boxes con color por material y etiqueta de clase + confianza.
- **`run_detection`**: comprime imagen a JPEG 85% / 1280px y la envía por gRPC.
- **`main`**: orquesta la interfaz completa — upload, detección, visualización y tarjetas de resultados.

---

### `proto/recycling.proto`

Define el contrato gRPC entre `app_service` e `inference_service`.

- **`DetectionRequest`**: `image_data (bytes)`.
- **`DetectedObject`**: `class_name`, `confidence`, `x1`, `y1`, `x2`, `y2`, `material`.
- **`DetectionResponse`**: `objects`, `success`, `message`.

---

## El Modelo — YOLO Waste Detection

### ¿Qué es y de dónde viene?

El modelo utilizado es **YOLO Waste Detection**, desarrollado por [gianlucasposito](https://github.com/gianlucasposito/YOLO-Waste-Detection) y publicado en GitHub bajo licencia MIT. Está construido sobre la arquitectura **YOLOv8 nano** de Ultralytics y fue entrenado mediante **fine-tuning** específico en imágenes de residuos reales.

El proceso fue el siguiente: se tomó YOLOv8n — el modelo base preentrenado por Ultralytics en el dataset COCO — y se re-entrenó sobre un dataset de 4.127 imágenes de basura anotadas con bounding boxes. Este proceso de fine-tuning aprovecha el conocimiento visual general de COCO (formas, texturas, bordes) y lo especializa para reconocer exclusivamente residuos reciclables. El resultado final se guardó como `best_model.pt`, que corresponde a los mejores pesos obtenidos durante ese entrenamiento.

### ¿Por qué fine-tuning y no entrenamiento desde cero?

Entrenar un modelo de visión desde cero requiere millones de imágenes y días de cómputo en GPU. El fine-tuning aprovecha el conocimiento previo del modelo base y solo ajusta los parámetros necesarios para el nuevo dominio, con órdenes de magnitud menos datos y tiempo. Es la práctica estándar de la industria para adaptar modelos de visión a dominios específicos.

### Dataset de entrenamiento

El modelo fue entrenado con el dataset público [waste-detection-using-yoloV5](https://github.com/utpalpaul108/waste-detection-using-yoloV5):

| Atributo | Detalle |
|---|---|
| Total de imágenes | 4.127 |
| Entrenamiento | 3.502 imágenes |
| Validación | 580 imágenes |
| Test | 45 imágenes |
| Número de clases | 5 |

### Clases detectadas

| Clase | Material | Contenedor |
|---|---|---|
| `Glass` | Vidrio | ♻️ Verde |
| `Paper` | Papel | ♻️ Azul |
| `Plastic` | Plástico | ♻️ Amarillo |
| `Metal` | Metal | ♻️ Gris |
| `Waste` | Basura general | 🗑️ Negro |

### ¿Cómo funciona internamente?

```
Imagen de entrada
      │
      ▼
┌─────────────┐
│  Backbone   │  ← Extrae características visuales: bordes, formas, texturas
└──────┬──────┘
       ▼
┌─────────────┐
│    Neck     │  ← Combina información a diferentes escalas de la imagen
└──────┬──────┘
       ▼
┌─────────────┐
│    Head     │  ← Predice: ¿qué residuo? ¿dónde? ¿qué tan seguro?
└──────┬──────┘
       ▼
  Detecciones (clase + confianza + coordenadas)
```

La variante **nano** fue elegida porque corre eficientemente en CPU sin necesidad de GPU, priorizando velocidad de respuesta en entornos de desarrollo local.

### Fuente y licencia

| Atributo | Detalle |
|---|---|
| Repositorio | [gianlucasposito/YOLO-Waste-Detection](https://github.com/gianlucasposito/YOLO-Waste-Detection) |
| Licencia | MIT |
| Framework | Ultralytics YOLOv8 |
| Dataset base | [utpalpaul108/waste-detection-using-yoloV5](https://github.com/utpalpaul108/waste-detection-using-yoloV5) |

---

## Gestión del Modelo con MLflow

```bash
mlflow ui  # Abre http://localhost:5000
```

- **Experimento:** `recycling-trashnet-registration`
- **Run:** `trashnet-v5-roboflow`
- **Artefacto:** `trashnet.pt` bajo `model/`

```powershell
python notebooks/register_model_mlflow.py
```

---

## Comunicación gRPC

### Regenerar código desde el .proto

```bash
python -m grpc_tools.protoc -I proto --python_out=proto --grpc_python_out=proto proto/recycling.proto
```

### Probar la conexión manualmente

```bash
python inference_service/test_client.py
```

Resultado esperado: `Conexion exitosa: False` — el servidor respondió aunque los bytes enviados no sean una imagen válida, lo que confirma que gRPC está activo.

---

## Pruebas Unitarias

```bash
pytest tests/ -v
```

**13 pruebas** cubren el clasificador de materiales (clases, contenedores, reciclabilidad, case-insensitivity) y el preprocesamiento de imágenes (compresión, aspect ratio, conversión RGB).

---

## Gestión del Proyecto — Tablero Kanban

El proyecto se gestiona con **Jira** usando Gitflow: ramas `main` y `develop`, con ramas `feature/` por cada funcionalidad.

🔗 **Tablero Kanban:** https://recycle-project.atlassian.net/jira/software/projects/KAN/boards/1?atlOrigin=eyJpIjoiNWJlMGExZmIxMGNhNDc2OWI2NmFiMDk1ZTc5NGFiNjgiLCJwIjoiaiJ9

---

## Decisiones de Diseño

**¿Por qué YOLO Waste Detection en lugar de YOLOv8n-COCO?** El modelo elegido fue fine-tuneado sobre 4.127 imágenes de residuos reales para las 5 categorías exactas del dominio del reciclaje, mejorando la precisión en el dominio específico y eliminando la necesidad de remapeo entre clases genéricas y materiales.

**¿Por qué separar inference_service de app_service?** Bajo acoplamiento: cambiar el modelo solo requiere modificar `inference_service` sin tocar la interfaz. Permite escalar cada servicio de forma independiente.

**¿Por qué comprimir la imagen antes de enviarla por gRPC?** gRPC tiene un límite de 4MB por mensaje. La compresión JPEG 85% con máximo 1280px reduce imágenes de 12MB+ sin pérdida visual significativa.

**¿Por qué uv en lugar de pip?** `uv` es un instalador escrito en Rust, significativamente más rápido que pip — reduce la instalación de dependencias de minutos a segundos, especialmente útil en Docker.

**¿Por qué no subir trashnet.pt al repositorio?** Los modelos son artefactos pesados que no deben versionarse con Git. Se gestionan como artefactos de MLflow y se descargan bajo demanda, siguiendo buenas prácticas de MLOps.

---

## Limitaciones Conocidas

- El modelo detecta 5 clases sin incluir `Cardboard` como clase separada — el cartón es clasificado como `Paper` o `Waste` según el contexto visual.
- La clase `Waste` actúa como categoría residual para objetos ambiguos.
- El conjunto de test del dataset original es pequeño (45 imágenes) — el rendimiento en imágenes muy distintas al dominio de entrenamiento puede variar.
- No se reconocen residuos orgánicos ni materiales compuestos.
- El sistema está diseñado para ejecución local. El despliegue en nube corresponde al Módulo 4 del curso.

---

## Licencia

Este proyecto está distribuido bajo la licencia **MIT**. Consulta el archivo [LICENSE](LICENSE) para más información.

El modelo YOLO Waste Detection está distribuido bajo licencia **MIT** por gianlucasposito. Fuente: [github.com/gianlucasposito/YOLO-Waste-Detection](https://github.com/gianlucasposito/YOLO-Waste-Detection).