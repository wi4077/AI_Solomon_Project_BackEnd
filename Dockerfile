### -------- builder --------
FROM python:3.12-slim AS builder
ARG HF_MODEL=BAAI/bge-reranker-small

# 설치
RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install --no-cache-dir --no-compile --prefer-binary -r /tmp/requirements.txt \
 && pip cache purge

# 모델만 tar 로 묶기
ENV HF_HOME=/opt/hf-cache
RUN mkdir -p $HF_HOME \
 && huggingface-cli download $HF_MODEL --repo-type model --local-dir $HF_HOME/$HF_MODEL \
 && tar -C $HF_HOME -czf /opt/model.tgz $HF_MODEL

### -------- runner --------
FROM python:3.12-slim
ENV PATH="/opt/venv/bin:$PATH" HF_HOME=/opt/hf-cache
WORKDIR /app

# 복사: venv + 모델 tar
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/model.tgz /opt/model.tgz
RUN mkdir -p $HF_HOME && tar -C $HF_HOME -xzf /opt/model.tgz && rm /opt/model.tgz

COPY . .
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]