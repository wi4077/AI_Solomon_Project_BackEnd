FROM python:3.12.10 AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
WORKDIR /app

RUN python -m venv .venv
COPY requirements.txt ./
RUN .venv/bin/pip install -r requirements.txt

FROM python:3.12.10-slim
WORKDIR /app
COPY --from=builder /app/.venv .venv/
COPY . .

# (선택) 포트 명시
EXPOSE 8000

# main.py 파일의 app 객체를 실행 (파일명에 맞게 수정)
CMD ["/app/.venv/bin/uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]