### ---------- builder stage ----------
FROM python:3.12.10-slim AS builder

# Faster, deterministic builds
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 1) create isolated venv in /opt/venv (easier to COPY later)
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 2) requirements first → leverage Docker layer cache
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir huggingface-hub

# 3) pre-download Cross-Encoder model so runtime doesn’t fetch it
ENV HF_HOME=/opt/hf-cache
RUN huggingface-cli download BAAI/bge-reranker-base --repo-type model --local-dir $HF_HOME/bge-reranker-base

### ---------- runtime stage ----------
FROM python:3.12.10-slim

# copy venv & HF cache
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/hf-cache /opt/hf-cache
ENV PATH="/opt/venv/bin:$PATH" \
    HF_HOME=/opt/hf-cache

WORKDIR /app
COPY . .

# expose & run
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]