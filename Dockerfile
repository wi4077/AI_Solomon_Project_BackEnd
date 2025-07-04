### ---------- builder stage ----------
FROM python:3.12-slim AS builder

ARG HF_MODEL=BAAI/bge-reranker-small

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 1) 먼저 requirements.txt 복사 — Docker 레이어 캐시 활용
COPY requirements.txt ./requirements.txt

# 2) 가상환경 생성 및 의존 설치
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --no-compile --prefer-binary -r requirements.txt \
    && /opt/venv/bin/pip install --no-cache-dir huggingface-hub \
    && /opt/venv/bin/pip cache purge

ENV PATH="/opt/venv/bin:$PATH" \
    HF_HOME=/opt/hf-cache

# 3) Cross‑Encoder 모델을 미리 다운로드하고 tar 로 압축
RUN mkdir -p $HF_HOME \
    && huggingface-cli download ${HF_MODEL} --repo-type model --local-dir $HF_HOME/${HF_MODEL} \
    && tar -C $HF_HOME -czf /opt/model.tgz ${HF_MODEL}

### ---------- runtime stage ----------
FROM python:3.12-slim

ENV PATH="/opt/venv/bin:$PATH" HF_HOME=/opt/hf-cache
WORKDIR /app

# 1) venv / 모델 캐시 복사
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/model.tgz /opt/model.tgz
RUN mkdir -p $HF_HOME \
    && tar -C $HF_HOME -xzf /opt/model.tgz \
    && rm /opt/model.tgz

# 2) 애플리케이션 소스 복사
COPY . .

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]