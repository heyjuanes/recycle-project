FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgl1 \
    libgthread-2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY proto/ ./proto/
COPY inference_service/ ./inference_service/
COPY app_service/ ./app_service/
COPY yolov8n.pt .

EXPOSE 8501
EXPOSE 50051

ENV PYTHONPATH=/app/proto

CMD ["sh", "-c", "python inference_service/server.py & streamlit run app_service/app.py --server.port=8501 --server.address=0.0.0.0"]