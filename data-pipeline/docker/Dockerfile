# COVID-19 전염병 예측 및 백신 효과성 분석을 위한 Docker 이미지
FROM python:3.11-slim

# 환경 변수 설정
ENV PYTHONUNBUFFERED=1
ENV JUPYTER_ENABLE_LAB=yes

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    vim \
    build-essential \
    libgdal-dev \
    gdal-bin \
    libspatialindex-dev \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 생성
WORKDIR /workspace

# Python 라이브러리 설치용 requirements.txt 생성
COPY requirements.txt .

# Python 패키지 설치
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Jupyter 설정 디렉토리 생성
RUN mkdir -p /root/.jupyter

# Jupyter 설정 파일 생성
RUN echo "c.ServerApp.token = ''" > /root/.jupyter/jupyter_server_config.py && \
    echo "c.ServerApp.password = ''" >> /root/.jupyter/jupyter_server_config.py && \
    echo "c.ServerApp.allow_root = True" >> /root/.jupyter/jupyter_server_config.py && \
    echo "c.ServerApp.ip = '0.0.0.0'" >> /root/.jupyter/jupyter_server_config.py

# 프로젝트 디렉토리 구조 생성
RUN mkdir -p /workspace/data/raw && \
    mkdir -p /workspace/data/processed && \
    mkdir -p /workspace/notebooks && \
    mkdir -p /workspace/src && \
    mkdir -p /workspace/models && \
    mkdir -p /workspace/results

# 포트 노출 (Jupyter Lab용)
EXPOSE 8888

# 기본 명령어
CMD ["jupyter", "lab", "--port=8888", "--no-browser", "--allow-root", "--ip=0.0.0.0"]
