# ♻️ Recycle Project — Sistema Inteligente de Detección de Material Reciclable

![MIT License](https://img.shields.io/badge/License-MIT-green.svg)
![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![TrashNet](https://img.shields.io/badge/TrashNet-Roboflow-purple.svg)
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
- [El Modelo — TrashNet v5](#el-modelo--trashnet-v5)
- [Gestión del Modelo con MLflow](#gestión-del-modelo-con-mlflow)
- [Comunicación gRPC](#comunicación-grpc)
- [Pruebas Unitarias](#pruebas-unitarias)
- [Gestión del Proyecto — Tablero Kanban](#gestión-del-proyecto--tablero-kanban)
- [Decisiones de Diseño](#decisiones-de-diseño)
- [Limitaciones Conocidas](#limitaciones-conocidas)
- [Licencia](#licencia)

---

## Descripción del Proyecto

Este proyecto implementa una aplicación web de visión por computador que permite al usuario cargar una imagen, detectar automáticamente objetos reciclables presentes en ella y clasificar cada objeto según su tipo de material (vidrio, papel, cartón, plástico, metal o basura general).

El sistema utiliza **TrashNet v5**, un modelo YOLOv8 preentrenado específicamente en residuos reciclables, publicado por Polygence Project en Roboflow Universe bajo licencia CC BY 4.0. A diferencia de modelos de propósito general, TrashNet fue entrenado con 2.524 imágenes de residuos reales anotadas con bounding boxes, lo que le permite detectar directamente las 6 categorías de reciclaje sin necesidad de un módulo de remapeo adicional.

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
│  - Carga el modelo TrashNet v5 al iniciar           │
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
4. `inference_service` decodifica los bytes, crea una imagen PIL y la pasa a TrashNet.
5. TrashNet retorna las detecciones: clase, confianza y coordenadas del bounding box.
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
│   ├── server.py               # Servidor gRPC + lógica de inferencia TrashNet
│   ├── material_classifier.py  # Módulo de reglas clase → material reciclable
│   └── test_client.py          # Cliente de prueba para verificar la conexión gRPC
├── proto/
│   ├── recycling.proto         # Definición del contrato gRPC
│   ├── recycling_pb2.py        # Código Python generado desde el .proto (mensajes)
│   └── recycling_pb2_grpc.py   # Código Python generado desde el .proto (servicios)
├── notebooks/
│   ├── download_trashnet.py      # Descarga trashnet.pt desde Roboflow
│   └── register_model_mlflow.py  # Registra el modelo en MLflow
├── data/
│   ├── raw/                    # Dataset original sin procesar
│   └── processed/              # Dataset preprocesado
├── tests/
│   ├── __init__.py
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

> ⚠️ El archivo `trashnet.pt` no está en el repositorio (excluido por `.gitignore`). Descárgalo con el script `notebooks/download_trashnet.py` antes de ejecutar el sistema.

---

## Tecnologías y Dependencias

| Tecnología | Versión | Uso |
|---|---|---|
| Python | 3.11 | Lenguaje base |
| TrashNet v5 (YOLOv8) | Roboflow | Detección de objetos reciclables |
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
| roboflow | latest | Descarga del modelo TrashNet |
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

### 4. Descargar el modelo TrashNet

El archivo `trashnet.pt` no está en el repositorio. Descárgalo ejecutando:

```powershell
python notebooks/download_trashnet.py
```

Esto descarga el modelo TrashNet v5 desde Roboflow Universe y lo guarda como `trashnet.pt` en la raíz del proyecto.

### 5. Iniciar el servidor gRPC (inference_service)

Abre una terminal, activa el entorno virtual y ejecuta:

```powershell
python inference_service/server.py
```

Deberías ver:
```
Cargando modelo TrashNet v5...
Modelo TrashNet listo. Clases: ['glass', 'paper', 'cardboard', 'plastic', 'metal', 'trash']
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

Módulo de reglas de negocio que mapea las clases detectadas por TrashNet a categorías de material reciclable con información de contenedor y reciclabilidad.

**`MATERIAL_MAP`** — diccionario que define la relación entre clase del modelo y material en español:

| Clase TrashNet | Material | Contenedor | Reciclable |
|---|---|---|---|
| `glass` | Vidrio | Verde ♻️ | Sí |
| `paper` | Papel | Azul ♻️ | Sí |
| `cardboard` | Cartón | Azul ♻️ | Sí |
| `plastic` | Plástico | Amarillo ♻️ | Sí |
| `metal` | Metal | Gris ♻️ | Sí |
| `trash` | Basura general | Negro 🗑️ | No |

**`get_material(class_name: str) -> str`**
- Recibe el nombre de la clase detectada por TrashNet.
- Convierte el nombre a minúsculas para evitar problemas de capitalización.
- Retorna la categoría del material en español.
- Si la clase no está en el mapa, retorna `"no clasificado"`.

**`get_material_info(class_name: str) -> dict`**
- Retorna un diccionario con `contenedor` (str) y `reciclable` (bool).
- Útil para la interfaz cuando necesita el color del contenedor y el estado de reciclabilidad por separado.

---

### `inference_service/server.py`

Servidor gRPC que expone el servicio de inferencia TrashNet.

**`RecyclingInferenceServicer`** — clase que implementa el contrato gRPC definido en `recycling.proto`.

- **`__init__(self)`**: carga el modelo TrashNet v5 desde `trashnet.pt` al inicializar el servidor. El modelo se carga una sola vez en memoria para optimizar el rendimiento e imprime las clases disponibles al arrancar.

- **`DetectObjects(self, request, context) -> DetectionResponse`**: método principal del servicio.
  - Recibe un `DetectionRequest` con la imagen codificada en bytes.
  - Decodifica los bytes a imagen PIL usando `io.BytesIO`.
  - Ejecuta inferencia TrashNet con `verbose=False` para suprimir logs internos.
  - Itera sobre cada detección extrayendo: `class_id`, `class_name`, `confidence` y coordenadas `x1, y1, x2, y2`.
  - Consulta `get_material()` para cada clase detectada.
  - Construye y retorna un `DetectionResponse` con la lista de `DetectedObject`.
  - En caso de error captura la excepción y retorna `success=False` con el mensaje del error.

**`serve()`**: función que inicializa y arranca el servidor gRPC.
  - Configura límites de mensaje de 20MB para soportar imágenes grandes.
  - Usa un `ThreadPoolExecutor` con 4 workers para manejar solicitudes concurrentes.
  - Escucha en el puerto `50051` en todas las interfaces (`[::]`).

---

### `app_service/app.py`

Interfaz web Streamlit que actúa como cliente gRPC.

**`get_stub() -> RecyclingInferenceStub`**
- Crea un canal gRPC inseguro hacia `localhost:50051`.
- Retorna un stub del servicio `RecyclingInference` listo para hacer llamadas RPC.

**`draw_boxes(image: Image.Image, objects: list) -> Image.Image`**
- Recibe la imagen PIL original y la lista de objetos detectados.
- Para cada objeto dibuja un rectángulo de color según el material usando `ImageDraw`.
- Cada material tiene un color asignado: plástico (azul), metal (rojo), vidrio (verde), cartón (naranja), basura (gris).
- Dibuja una etiqueta con la clase, el material y el porcentaje de confianza sobre cada bounding box.
- Retorna la imagen anotada.

**`run_detection(image_bytes: bytes) -> DetectionResponse`**
- Recibe los bytes crudos de la imagen subida por el usuario.
- Abre la imagen con PIL y la convierte a RGB.
- Aplica `thumbnail((1280, 1280))` para redimensionar imágenes grandes sin distorsionar la proporción.
- Recodifica la imagen a JPEG con calidad 85 para reducir el tamaño antes de enviarla por gRPC.
- Crea el stub, construye el `DetectionRequest` y llama a `DetectObjects`.
- Retorna el `DetectionResponse` con los resultados.

**`main()`**
- Configura la página Streamlit con título, ícono y layout en dos columnas.
- Muestra el `file_uploader` para aceptar imágenes JPG y PNG.
- Al recibir una imagen llama a `run_detection` con un spinner de carga.
- Si hay detecciones exitosas muestra la imagen original en la columna izquierda y la imagen anotada en la columna derecha.
- Muestra una tarjeta por cada objeto detectado con el color del material correspondiente.
- Maneja los casos de imagen sin detecciones y errores de conexión.

---

### `notebooks/download_trashnet.py`

Script para descargar el modelo TrashNet v5 desde Roboflow Universe.

**`download_model()`**
- Se conecta a Roboflow Universe usando la API key del proyecto.
- Descarga el modelo TrashNet v5 en formato YOLOv8 hacia la carpeta `trashnet-v5/`.
- Busca el archivo `best.pt` dentro de la carpeta descargada usando `rglob`.
- Lo copia a la raíz del proyecto como `trashnet.pt`.
- Imprime la ruta de destino al finalizar.

---

### `notebooks/register_model_mlflow.py`

Script de registro del modelo TrashNet v5 en MLflow.

**`register_trashnet_model()`**
- Crea o selecciona el experimento `recycling-trashnet-registration` en MLflow.
- Inicia un run llamado `trashnet-v5-roboflow`.
- Carga el modelo TrashNet con `YOLO("trashnet.pt")` e imprime las clases disponibles.
- Registra parámetros del modelo: nombre, tipo, fuente, número de clases, lista de clases, imágenes de entrenamiento, tamaño de entrada, framework, modo y licencia.
- Registra las métricas oficiales de TrashNet v5: mAP50=0.661, precision=0.802, recall=0.601.
- Guarda el archivo `trashnet.pt` como artefacto bajo la ruta `model/` del run.
- Imprime el Run ID al finalizar.

---

### `inference_service/test_client.py`

Cliente de prueba para verificar que la conexión gRPC está funcionando.

**`test_connection()`**
- Crea un canal gRPC hacia `localhost:50051`.
- Envía una petición mínima con bytes no válidos como imagen.
- Verifica que el servidor responde (aunque sea con `success=False`).
- Imprime el estado de la conexión y el mensaje retornado.
- Útil para diagnosticar problemas de red o de configuración del servidor antes de usar la interfaz completa.

---

### `proto/recycling.proto`

Define el contrato de comunicación entre `app_service` e `inference_service`.

**Servicio `RecyclingInference`:**
- Método `DetectObjects`: recibe un `DetectionRequest` y retorna un `DetectionResponse`.

**Mensaje `DetectionRequest`:**
- `image_data (bytes)`: imagen codificada en bytes para ser procesada.

**Mensaje `DetectedObject`:**
- `class_name (string)`: nombre de la clase detectada por TrashNet.
- `confidence (float)`: nivel de confianza de la detección (0.0 a 1.0).
- `x1, y1, x2, y2 (float)`: coordenadas del bounding box en píxeles.
- `material (string)`: categoría de material reciclable asignada.

**Mensaje `DetectionResponse`:**
- `objects (repeated DetectedObject)`: lista de todos los objetos detectados.
- `success (bool)`: indica si la inferencia fue exitosa.
- `message (string)`: mensaje descriptivo del resultado o del error.

---

## El Modelo — TrashNet v5

### ¿Qué es TrashNet?

**TrashNet** es un modelo de detección de objetos basado en la arquitectura **YOLOv8**, entrenado específicamente en imágenes de residuos reciclables y publicado por **Polygence Project** en [Roboflow Universe](https://universe.roboflow.com/polygence-project/trashnet-a-set-of-annotated-images-of-trash-that-can-be-used-for-object-detection) bajo licencia **CC BY 4.0**.

A diferencia de modelos de propósito general como YOLOv8n-COCO, TrashNet fue entrenado exclusivamente con imágenes de residuos, lo que lo hace más preciso para el dominio específico del reciclaje.

### Clases detectadas

| Clase | Material | Contenedor |
|---|---|---|
| `glass` | Vidrio | ♻️ Verde |
| `paper` | Papel | ♻️ Azul |
| `cardboard` | Cartón | ♻️ Azul |
| `plastic` | Plástico | ♻️ Amarillo |
| `metal` | Metal | ♻️ Gris |
| `trash` | Basura general | 🗑️ Negro |

### ¿Cómo funciona internamente?

Cuando recibe una imagen, TrashNet (YOLOv8) la procesa en tres etapas:

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

### Métricas oficiales (TrashNet v5 — Roboflow Universe)

| Métrica | Valor |
|---|---|
| mAP@0.5 | 66.1% |
| Precision | 80.2% |
| Recall | 60.1% |
| Imágenes de entrenamiento | 2.524 |

> Fuente: [Roboflow Universe — TrashNet v5](https://universe.roboflow.com/polygence-project/trashnet-a-set-of-annotated-images-of-trash-that-can-be-used-for-object-detection/model/5)  
> Licencia: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

---

## Gestión del Modelo con MLflow

MLflow se usa para llevar el ciclo de vida del modelo TrashNet v5.

### Iniciar la interfaz MLflow

```bash
mlflow ui
```

Abre: **http://localhost:5000**

### Qué encontrarás en MLflow

- **Experimento:** `recycling-trashnet-registration`
- **Run:** `trashnet-v5-roboflow`
- **Parámetros registrados:** nombre del modelo, fuente, número de clases, lista de clases, imágenes de entrenamiento, tamaño de entrada, framework, licencia.
- **Métricas registradas:** mAP50, precision, recall (métricas oficiales de TrashNet v5 en Roboflow).
- **Artefacto:** archivo `trashnet.pt` almacenado bajo la ruta `model/`.

### Registrar el modelo en MLflow

```powershell
python notebooks/register_model_mlflow.py
```

---

## Comunicación gRPC

### ¿Por qué gRPC?

gRPC es el estándar de la industria para comunicación entre microservicios de ML. Ofrece tipificación estricta del contrato mediante Protocol Buffers, alta eficiencia con datos binarios (imágenes) y facilidad para escalar hacia múltiples servicios.

### Regenerar el código desde el .proto

Si modificas `proto/recycling.proto`, regenera el código Python con:

```bash
python -m grpc_tools.protoc -I proto --python_out=proto --grpc_python_out=proto proto/recycling.proto
```

Esto regenera `recycling_pb2.py` y `recycling_pb2_grpc.py` dentro de la carpeta `proto/`.

### Probar la conexión gRPC manualmente

Con el servidor corriendo en una terminal:

```bash
python inference_service/test_client.py
```

Resultado esperado:
```
Conexion exitosa: False
Mensaje: Error: cannot identify image file ...
```

> El `False` es esperado porque el cliente de prueba envía bytes inválidos. Lo importante es que el servidor **respondió**, lo que confirma que la conexión gRPC está activa.

---

## Pruebas Unitarias

Las pruebas se ejecutan con pytest y cubren los módulos críticos del sistema.

### Ejecutar todas las pruebas

```bash
pytest tests/ -v
```

### Resultado esperado

```
tests/test_material_classifier.py::test_glass_es_vidrio             PASSED
tests/test_material_classifier.py::test_plastic_es_plastico         PASSED
tests/test_material_classifier.py::test_metal_es_metal              PASSED
tests/test_material_classifier.py::test_cardboard_es_carton         PASSED
tests/test_material_classifier.py::test_clase_desconocida           PASSED
tests/test_material_classifier.py::test_case_insensitive            PASSED
tests/test_preprocessing.py::test_imagen_se_comprime_correctamente  PASSED
tests/test_preprocessing.py::test_imagen_thumbnail_respeta_aspecto  PASSED
tests/test_preprocessing.py::test_imagen_se_convierte_a_rgb        PASSED

9 passed in 0.32s
```

### Descripción de las pruebas

**`tests/test_material_classifier.py`** — prueba el módulo `material_classifier`:
- `test_glass_es_vidrio`: verifica que `glass` se clasifica como `vidrio`.
- `test_plastic_es_plastico`: verifica que `plastic` se clasifica como `plastico`.
- `test_metal_es_metal`: verifica que `metal` se clasifica como `metal`.
- `test_cardboard_es_carton`: verifica que `cardboard` se clasifica como `carton`.
- `test_clase_desconocida`: verifica que una clase no mapeada retorna `no clasificado`.
- `test_case_insensitive`: verifica que el clasificador no distingue mayúsculas de minúsculas.

**`tests/test_preprocessing.py`** — prueba el preprocesamiento de imágenes:
- `test_imagen_se_comprime_correctamente`: crea una imagen de 1920x1080, la comprime y verifica que el resultado pesa menos de 4MB (límite de gRPC por defecto).
- `test_imagen_thumbnail_respeta_aspecto`: verifica que `thumbnail(1280, 1280)` no supera el tamaño máximo en ninguna dimensión.
- `test_imagen_se_convierte_a_rgb`: verifica que una imagen RGBA se convierte correctamente a RGB sin errores.

> ⚠️ Los tests de `test_material_classifier.py` verifican las clases de TrashNet. Si actualizaste el clasificador, asegúrate de actualizar también los tests para que reflejen las nuevas clases.

---

## Gestión del Proyecto — Tablero Kanban

El proyecto se gestiona con **Jira** usando las columnas: To Do → In Progress → Review → Done.

Cada funcionalidad del sistema se registra como un issue independiente con su rama `feature/` correspondiente siguiendo la metodología **Gitflow**.

🔗 **Tablero Kanban:** https://recycle-project.atlassian.net/jira/software/projects/KAN/boards/1?atlOrigin=eyJpIjoiNWJlMGExZmIxMGNhNDc2OWI2NmFiMDk1ZTc5NGFiNjgiLCJwIjoiaiJ9

---

## Decisiones de Diseño

**¿Por qué TrashNet en lugar de YOLOv8n-COCO?**
YOLOv8n-COCO fue entrenado en 80 clases de objetos cotidianos, no específicamente en residuos. TrashNet fue entrenado con 2.524 imágenes de residuos reales anotadas con bounding boxes para las 6 categorías exactas del dominio del reciclaje. Esto elimina la necesidad de un módulo de remapeo entre clases COCO y materiales, simplifica el código y mejora la precisión en el dominio específico.

**¿Por qué separar inference_service de app_service?**
Para aplicar el principio de bajo acoplamiento: si en el futuro se cambia el modelo, solo se modifica `inference_service` sin tocar la interfaz. Además permite escalar el servidor de inferencia de forma independiente del frontend.

**¿Por qué comprimir la imagen antes de enviarla por gRPC?**
gRPC tiene un límite de 4MB por mensaje por defecto. Imágenes de alta resolución (12MB+) superan este límite. La compresión a JPEG calidad 85 con máximo 1280px reduce el tamaño sin pérdida visual significativa para los propósitos de detección.

**¿Por qué uv en lugar de pip?**
`uv` es un instalador de paquetes Python escrito en Rust, significativamente más rápido que pip. Reduce el tiempo de instalación de dependencias de minutos a segundos, lo que es especialmente útil al construir la imagen Docker o configurar el entorno en una máquina nueva.

**¿Por qué no subir trashnet.pt al repositorio?**
Los archivos de modelos son artefactos pesados que no deben versionarse con Git. Se gestionan exclusivamente como artefactos de MLflow y se descargan bajo demanda mediante `notebooks/download_trashnet.py`, siguiendo las buenas prácticas de MLOps.

---

## Limitaciones Conocidas

- El modelo TrashNet v5 tiene un Recall de 60.1%, lo que significa que puede no detectar todos los objetos presentes en una imagen, especialmente en condiciones de baja iluminación o ángulos inusuales.
- La clase `trash` actúa como categoría residual para objetos que no son claramente reciclables, por lo que puede capturar objetos de categorías ambiguas.
- Objetos grandes o con formas complejas pueden aparecer con detecciones duplicadas de diferente confianza.
- El sistema está diseñado para ejecución local. El despliegue en nube corresponde al Módulo 4 del curso.
- No se reconocen residuos orgánicos ni materiales compuestos.
- La clasificación de material se basa en las clases directas del modelo, no en reglas adicionales de negocio.

---

## Licencia

Este proyecto está distribuido bajo la licencia **MIT**. Consulta el archivo [LICENSE](LICENSE) para más información.

El modelo TrashNet v5 está distribuido bajo licencia **CC BY 4.0** por Polygence Project. Fuente: [Roboflow Universe](https://universe.roboflow.com/polygence-project/trashnet-a-set-of-annotated-images-of-trash-that-can-be-used-for-object-detection).