FROM python:3.11-slim

# 시스템 유틸
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && \
    rm -rf /var/lib/apt/lists/*

ENV TZ=Asia/Seoul \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# MLflow + Postgres 드라이버 + S3용 boto3
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir \
      mlflow==2.15.1 \
      psycopg2-binary==2.9.9 \
      boto3==1.34.162

WORKDIR /srv
EXPOSE 8080
CMD ["bash", "-lc", "mlflow --help"]

