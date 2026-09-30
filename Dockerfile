# Slim serving image — sized for Render free tier (512 MB RAM / 0.1 vCPU).
# torch/sentence-transformers are intentionally absent: the serving path never
# imports them (verified), and their presence alone put the old image at 768 MB
# RSS before the first request.
# For a full local image incl. ingestion/evals deps, use Dockerfile.full.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/tmp/hf_cache \
    NUMEXPR_MAX_THREADS=1 \
    OMP_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false \
    RERANK_CACHE_DIR=/opt/flashrank

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

COPY . .

# Pre-bake the 3.3 MB FlashRank cross-encoder into the image so cold starts
# never download from HuggingFace (avoids startup dependency on an external CDN).
RUN python -c "from flashrank import Ranker; Ranker(cache_dir='/opt/flashrank', log_level='WARNING'); print('FlashRank model baked ->', '/opt/flashrank')"

# Run as non-root for security; appuser owns the model cache
RUN useradd --create-home appuser \
    && chown -R appuser: /app /opt/flashrank
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD curl -fsS "http://localhost:${PORT:-8000}/" || exit 1

# $PORT is injected by Render (default 10000); docker-compose has no PORT so it
# falls back to 8000, matching the compose port mapping.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]