FROM python:3.11-slim

# HF_HOME is set before the model download step so the weights land inside the
# image layer rather than in a per-container cache directory.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf-cache

WORKDIR /app

# curl is only needed for the HEALTHCHECK below.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# sentence-transformers pulls in torch. Installing the CPU-only wheel first
# keeps pip from resolving the default CUDA build (~2.5GB of unused GPU libs).
RUN pip install --upgrade pip \
    && pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt

# Bake the embedding model into the image. Without this, every cold start
# downloads ~90MB from HuggingFace before the API can serve its first request,
# and the container fails outright on a host with no outbound internet.
RUN python -c "from sentence_transformers import SentenceTransformer; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
ENV HF_HUB_OFFLINE=1
COPY . .

RUN useradd --create-home --uid 1000 appuser \
    && chown -R appuser:appuser /app /opt/hf-cache
USER appuser

EXPOSE 8000

# start-period covers model load into memory on boot, which takes a few seconds.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
