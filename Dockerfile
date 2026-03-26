FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgl1 \
    libgthread-2.0-0 \
    wget \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY proto/ ./proto/
COPY inference_service/ ./inference_service/
COPY app_service/ ./app_service/

RUN wget -O trashnet.pt "https://github.com/gianlucasposito/YOLO-Waste-Detection/raw/main/best_model.pt"

EXPOSE 8501
EXPOSE 50051

ENV PYTHONPATH=/app/proto

CMD ["sh", "-c", "python inference_service/server.py & sleep 15 && streamlit run app_service/app.py --server.port=8501 --server.address=0.0.0.0"]