# ===============================================
# Dockerfile para Sistema de Detección de Deepfakes
# Universidad del Bío-Bío - Ingeniería Civil en Informática
# ===============================================

FROM python:3.11-slim

# Variables de entorno para Python
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

WORKDIR /app

# Instalar dependencias del sistema necesarias para OpenCV y PyTorch
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar archivo de requisitos
COPY requirements.txt .

# Instalar PyTorch CPU (versión ligera para contenedor sin GPU)
RUN pip install --no-cache-dir \
    torch==2.1.0 \
    torchvision==0.16.0 \
    torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cpu

# Instalar dependencias del proyecto
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código fuente completo
COPY src/ ./src/
COPY models/ ./models/
COPY data/processed/ffpp/test/real/amigo*.jpeg ./data/test_images/

# Crear directorio para uploads si se necesita
RUN mkdir -p /app/uploads

# Verificar que el modelo existe
RUN test -f /app/models/deepfake_detector_resnet18_ffpp.pth || \
    (echo "ERROR: Modelo ResNet18 no encontrado en /app/models/" && exit 1)

# Exponer puerto para la API
EXPOSE 8000

# Health check para verificar que la API está funcionando
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Comando de inicio - Uvicorn con la API FastAPI
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

