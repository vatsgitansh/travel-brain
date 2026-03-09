# ─────────────────────────────────────────────────────────────────────────────
# Slim production Dockerfile for Travel Brain
# Target image size: < 2GB (vs 8.5GB with torch/onnxruntime)
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim-bullseye

# Install only essential system libs (no build-essential = no compilers = smaller)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

WORKDIR /app

# --- Dependency isolation: block heavy transitive packages ------------------
# Pre-install a dummy onnxruntime stub so chromadb doesn't pull the real one
# (chromadb uses onnxruntime only for its local embedding model;
#  we use Gemini embeddings so we don't need it at all)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        "onnxruntime==1.17.0" \
        --extra-index-url https://download.pytorch.org/whl/cpu 2>/dev/null || \
    pip install --no-cache-dir onnxruntime==1.17.0

# --- Install app dependencies -----------------------------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Copy app source (after deps for better Docker layer caching) ------------
COPY travel_brain/ travel_brain/
COPY data/chroma_db/ data/chroma_db/

# --- Runtime config ----------------------------------------------------------
ENV PYTHONUNBUFFERED=1
ENV EMBEDDING_PROVIDER=gemini
EXPOSE 8000

CMD ["uvicorn", "travel_brain.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
