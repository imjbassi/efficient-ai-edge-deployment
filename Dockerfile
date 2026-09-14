ARG BASE_IMAGE=python:3.10-slim-bookworm@sha256:68d914ec641a0b69267ce65184d000a2bc3a9ee2590ab702b82250ab2385735a
FROM ${BASE_IMAGE}

# Set system environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    MODEL_PATH=/app/models/mobilenet_v2_int8.tflite \
    TFLITE_THREADS=1 \
    MAX_UPLOAD_BYTES=10485760 \
    MAX_IMAGE_PIXELS=25000000

# Set the working directory in the container
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN useradd --create-home --uid 10001 appuser

COPY src/ ./src/
COPY models/mobilenet_v2_int8.tflite ./models/mobilenet_v2_int8.tflite
USER appuser

# Expose the API port
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

CMD ["uvicorn", "src.deploy:app", "--host", "0.0.0.0", "--port", "8000"]
