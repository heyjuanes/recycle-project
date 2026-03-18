# ♻️ Recycle Project — Sistema Inteligente de Detección de Material Reciclable

![MIT License](https://img.shields.io/badge/License-MIT-green.svg)
![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30-red.svg)
![gRPC](https://img.shields.io/badge/gRPC-1.60-orange.svg)
![MLflow](https://img.shields.io/badge/MLflow-2.10-blue.svg)

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
- [El Modelo — YOLOv8n](#el-modelo--yolov8n)
- [Gestión del Modelo con MLflow](#gestión-del-modelo-con-mlflow)
- [Comunicación gRPC](#comunicación-grpc)
- [Pruebas Unitarias](#pruebas-unitarias)
- [Gestión del Proyecto — Tablero Kanban](#gestión-del-proyecto--tablero-kanban)
- [Decisiones de Diseño](#decisiones-de-diseño)
- [Limitaciones Conocidas](#limitaciones-conocidas)
- [Licencia](#licencia)

---

## Descripción del Proyecto

Este proyecto implementa una aplicación web de visión por computador que permite al usuario cargar una imagen, detectar automáticamente objetos reciclables presentes en ella y clasificar cada objeto según su tipo de material (plástico, metal, vidrio o cartón).

El sistema utiliza **YOLOv8n** (You Only Look Once, versión 8 nano) como modelo de detección de objetos, un modelo preentrenado en el dataset COCO capaz de identificar 80 clases de objetos cotidianos. Sobre las detecciones de YOLO se aplica un módulo de reglas de negocio que mapea cada clase detectada a una categoría de material reciclable.

La aplicación tiene fines **educativos y demostrativos**, orientada a ilustrar cómo integrar un modelo de Machine Learning en una arquitectura de microservicios real, siguiendo buenas prácticas de ingeniería de software.

### Problema que resuelve

En muchos entornos urbanos los usuarios no saben en qué contenedor depositar un objeto. Este sistema permite fotografiar el objeto y obtener de inmediato su categoría de material, facilitando la separación correcta de residuos.

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
│  - Carga el modelo YOLOv8n al iniciar               │
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
4. `inference_service` decodifica los bytes, crea una imagen PIL y la pasa a YOLOv8.
5. YOLOv8 retorna las detecciones: clase, confianza y coordenadas del bounding box.
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
│   ├── material_classifier.py  # Módulo de reglas objeto → material
│   └── test_client.py          # Cliente de prueba para verificar la conexión gRPC
├── proto/
│   ├── recycling.proto         # Definición del contrato gRPC
│   ├── recycling_pb2.py        # Código Python generado desde el .proto (mensajes)
│   └── recycling_pb2_grpc.py   # Código Python generado desde el .proto (servicios)
├── notebooks/
│   └── register_model_mlflow.py  # Script de registro del modelo en MLflow
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

---

## Tecnologías y Dependencias

| Tecnología | Versión | Uso |
|---|---|---|
| Python | 3.11 | Lenguaje base |
| YOLOv8n (Ultralytics) | ≥ 8.0.0 | Detección de objetos |
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
| uv | latest | Instalación rápida de dependencias |

---

## Requisitos Previos

- Python 3.11 o superior instalado y en el PATH
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

### 2. Crear y activar el entorno virtual

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

### 4. Iniciar el servidor gRPC (inference_service)

Abre una terminal, activa el entorno virtual y ejecuta:

```bash
python inference_service/server.py
```

Deberías ver:
```
Cargando modelo YOLOv8n...
Modelo listo.
Servidor gRPC corriendo en puerto 50051...
```

> ⚠️ El modelo `yolov8n.pt` se descarga automáticamente (~6MB) la primera vez que se ejecuta el servidor.

### 5. Iniciar la interfaz Streamlit (app_service)

Abre una **segunda terminal**, activa el entorno virtual y ejecuta:

```bash
streamlit run app_service/app.py
```

La aplicación se abrirá automáticamente en: **http://localhost:8501**

### 6. Usar la aplicación

1. Abre http://localhost:8501 en tu navegador.
2. Haz clic en **"Browse files"** o arrastra una imagen JPG/PNG.
3. El sistema detectará automáticamente los objetos y clasificará su material.
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

---

## Documentación de Módulos y Funciones

### `inference_service/material_classifier.py`

Módulo de reglas de negocio que mapea clases de objetos detectados por YOLOv8 a categorías de material reciclable.

**`MATERIAL_MAP`** — diccionario que define la relación entre clase YOLO y material:

| Clase YOLO | Material |
|---|---|
| bottle, cup, bowl, chair, keyboard, remote, cell phone | plástico |
| can, fork, knife, spoon, scissors | metal |
| book, laptop, mouse, clock | cartón |
| wine glass, vase | vidrio |

**`get_material(class_name: str) -> str`**
- Recibe el nombre de la clase detectada por YOLOv8.
- Convierte el nombre a minúsculas para evitar problemas de capitalización.
- Retorna la categoría de material correspondiente.
- Si la clase no está en el mapa, retorna `"no clasificado"`.

---

### `inference_service/server.py`

Servidor gRPC que expone el servicio de inferencia YOLOv8.

**`RecyclingInferenceServicer`** — clase que implementa el contrato gRPC definido en `recycling.proto`.

- **`__init__(self)`**: carga el modelo YOLOv8n desde el archivo `yolov8n.pt` al inicializar el servidor. El modelo se carga una sola vez en memoria para optimizar el rendimiento.

- **`DetectObjects(self, request, context) -> DetectionResponse`**: método principal del servicio.
  - Recibe un `DetectionRequest` con la imagen codificada en bytes.
  - Decodifica los bytes a imagen PIL usando `io.BytesIO`.
  - Ejecuta inferencia YOLOv8 con `verbose=False` para suprimir logs internos.
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
- Cada material tiene un color asignado: plástico (azul), metal (rojo), vidrio (verde), cartón (naranja), no clasificado (gris).
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

### `notebooks/register_model_mlflow.py`

Script de registro del modelo YOLOv8n preentrenado en MLflow.

**`register_pretrained_model()`**
- Crea o selecciona el experimento `recycling-yolov8-registration` en MLflow.
- Inicia un run llamado `yolov8n-coco-pretrained`.
- Descarga y carga el modelo YOLOv8n con `YOLO("yolov8n.pt")` — la descarga es automática si el archivo no existe.
- Registra parámetros del modelo: tipo, pesos, número de clases, tamaño de entrada, framework y modo de entrenamiento.
- Registra las métricas oficiales de YOLOv8n en COCO: mAP50=0.525, mAP50-95=0.372, precision=0.680, recall=0.532.
- Guarda el archivo `yolov8n.pt` como artefacto bajo la ruta `model/` del run.
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
- `class_name (string)`: nombre de la clase detectada por YOLOv8.
- `confidence (float)`: nivel de confianza de la detección (0.0 a 1.0).
- `x1, y1, x2, y2 (float)`: coordenadas del bounding box en píxeles.
- `material (string)`: categoría de material reciclable asignada.

**Mensaje `DetectionResponse`:**
- `objects (repeated DetectedObject)`: lista de todos los objetos detectados.
- `success (bool)`: indica si la inferencia fue exitosa.
- `message (string)`: mensaje descriptivo del resultado o del error.

---

## El Modelo — YOLOv8n

### ¿Qué es YOLO?

**YOLO** (You Only Look Once) es una familia de modelos de detección de objetos que existe desde 2015. El nombre describe su principio fundamental: a diferencia de sistemas más antiguos que analizaban la imagen varias veces en diferentes regiones, YOLO analiza **toda la imagen en una sola pasada**, lo que lo hace extremadamente rápido.

**YOLOv8** es la versión más reciente, desarrollada por **Ultralytics** en 2023. La variante **n** (nano) es la más liviana de la familia, diseñada específicamente para correr en CPU sin necesidad de GPU.

### ¿Por qué no fue necesario entrenarlo?

Ultralytics entrenó YOLOv8n con el dataset **COCO** (Common Objects in Context), uno de los datasets más grandes del mundo de la visión por computador:

- **330,000 imágenes** de escenas cotidianas
- **1.5 millones** de objetos anotados manualmente
- **80 clases** de objetos: personas, vehículos, animales, utensilios, muebles, etc.

El resultado de ese entrenamiento está comprimido en el archivo `yolov8n.pt` (~6MB). Al ejecutar `YOLO("yolov8n.pt")` se carga toda esa inteligencia directamente en memoria, lista para detectar objetos en cualquier imagen.

### ¿Cómo funciona internamente?

Cuando recibe una imagen, YOLOv8 la procesa en tres etapas:

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
│    Head     │  ← Predice: ¿qué objeto? ¿dónde? ¿qué tan seguro?
└──────┬──────┘
       ▼
  Detecciones (clase + confianza + coordenadas)
```
Se eligió **YOLOv8n** porque el proyecto corre en CPU local, priorizando velocidad de respuesta sobre precisión máxima.

### Métricas oficiales (COCO benchmark)

| Métrica | Valor |
|---|---|
| mAP@0.5 | 52.5% |
| mAP@0.5:0.95 | 37.2% |
| Precision | 68.0% |
| Recall | 53.2% |

> Fuente:  [Ultralytics YOLOv8 Model Card](https://docs.ultralytics.com/models/yolov8/)

---

## Gestión del Modelo con MLflow

MLflow se usa para llevar el ciclo de vida del modelo YOLOv8n.

### Iniciar la interfaz MLflow

```bash
mlflow ui
```

Abre: **http://localhost:5000**

### Qué encontrarás en MLflow

- **Experimento:** `recycling-yolov8-registration`
- **Run:** `yolov8n-coco-pretrained`
- **Parámetros registrados:** tipo de modelo, pesos, número de clases, tamaño de entrada, framework, modo de entrenamiento.
- **Métricas registradas:** mAP50, mAP50-95, precision, recall (métricas oficiales de YOLOv8n en COCO).
- **Artefacto:** archivo `yolov8n.pt` almacenado bajo la ruta `model/`.

### Re-registrar el modelo

Si necesitas volver a registrar el modelo (por ejemplo después de un fine-tuning):

```bash
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
tests/test_material_classifier.py::test_bottle_es_plastico          PASSED
tests/test_material_classifier.py::test_can_es_metal                PASSED
tests/test_material_classifier.py::test_wine_glass_es_vidrio        PASSED
tests/test_material_classifier.py::test_book_es_carton              PASSED
tests/test_material_classifier.py::test_clase_desconocida           PASSED
tests/test_material_classifier.py::test_case_insensitive            PASSED
tests/test_preprocessing.py::test_imagen_se_comprime_correctamente  PASSED
tests/test_preprocessing.py::test_imagen_thumbnail_respeta_aspecto  PASSED
tests/test_preprocessing.py::test_imagen_se_convierte_a_rgb        PASSED

9 passed in 0.32s
```

### Descripción de las pruebas

**`tests/test_material_classifier.py`** — prueba el módulo `material_classifier`:
- `test_bottle_es_plastico`: verifica que `bottle` se clasifica como `plastico`.
- `test_can_es_metal`: verifica que `can` se clasifica como `metal`.
- `test_wine_glass_es_vidrio`: verifica que `wine glass` se clasifica como `vidrio`.
- `test_book_es_carton`: verifica que `book` se clasifica como `carton`.
- `test_clase_desconocida`: verifica que una clase no mapeada retorna `no clasificado`.
- `test_case_insensitive`: verifica que el clasificador no distingue mayúsculas de minúsculas.

**`tests/test_preprocessing.py`** — prueba el preprocesamiento de imágenes:
- `test_imagen_se_comprime_correctamente`: crea una imagen de 1920x1080, la comprime y verifica que el resultado pesa menos de 4MB (límite de gRPC por defecto).
- `test_imagen_thumbnail_respeta_aspecto`: verifica que `thumbnail(1280, 1280)` no supera el tamaño máximo en ninguna dimensión.
- `test_imagen_se_convierte_a_rgb`: verifica que una imagen RGBA se convierte correctamente a RGB sin errores.

---

## Gestión del Proyecto — Tablero Kanban

El proyecto se gestiona con **Jira** usando las columnas: To Do → In Progress → Review → Done.

Cada funcionalidad del sistema se registra como un issue independiente con su rama `feature/` correspondiente siguiendo la metodología **Gitflow**.

🔗 **Tablero Kanban:** https://recycle-project.atlassian.net/jira/software/projects/KAN/boards/1?atlOrigin=eyJpIjoiNWJlMGExZmIxMGNhNDc2OWI2NmFiMDk1ZTc5NGFiNjgiLCJwIjoiaiJ9 

---

## Decisiones de Diseño

**¿Por qué YOLOv8n y no fine-tuning?**
Se optó por usar el modelo preentrenado en COCO que ya detecta objetos cotidianos reciclables como botellas, tazas y vasos. El valor académico del proyecto está en la arquitectura de software, no en el entrenamiento del modelo. Si en el futuro se quisiera mejorar la precisión, bastaría con reemplazar `yolov8n.pt` por un modelo fine-tuneado sin modificar ningún otro componente del sistema.

**¿Por qué separar inference_service de app_service?**
Para aplicar el principio de bajo acoplamiento: si en el futuro se cambia el modelo de YOLOv8 a otro, solo se modifica `inference_service` sin tocar la interfaz. Además permite escalar el servidor de inferencia de forma independiente del frontend.

**¿Por qué comprimir la imagen antes de enviarla por gRPC?**
gRPC tiene un límite de 4MB por mensaje por defecto. Imágenes de alta resolución (12MB+) superan este límite. La compresión a JPEG calidad 85 con máximo 1280px reduce el tamaño sin pérdida visual significativa para los propósitos de detección de YOLOv8.

**¿Por qué uv en lugar de pip?**
`uv` es un instalador de paquetes Python escrito en Rust, significativamente más rápido que pip. Reduce el tiempo de instalación de dependencias de minutos a segundos, lo que es especialmente útil al construir la imagen Docker o configurar el entorno en una máquina nueva.

---

## Limitaciones Conocidas

- El modelo YOLOv8n fue entrenado en COCO (objetos cotidianos), no en un dataset específico de reciclaje, por lo que algunas clasificaciones de material pueden no ser precisas para todos los objetos.
- La cámara analógica puede ser detectada como `cell phone` por similitud visual en el dataset COCO.
- Objetos grandes o con formas complejas pueden aparecer con detecciones duplicadas de diferente confianza.
- El sistema está diseñado para ejecución local. El despliegue en nube corresponde al Módulo 4 del curso.
- No se reconocen residuos orgánicos ni materiales compuestos.
- La clasificación de material se basa en reglas fijas, no en aprendizaje automático.

---

## Licencia


Este proyecto está distribuido bajo la licencia **MIT**. Consulta el archivo [LICENSE](LICENSE) para más información.
