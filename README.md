# COVID-19 MLOps Pipeline Project

<br>

## 💻 프로젝트 소개
### 📊 프로젝트 개요

COVID-19 감염 확산 데이터를 활용한 **시계열 예측 모델 개발** 및 **MLOps 파이프라인 구축** 프로젝트입니다.

- 실시간 데이터 수집부터 모델 학습, 예측, 서빙까지 전체 ML 라이프사이클을 자동화
- 시계열 분석을 통한 COVID-19 확산 추세 예측 및 인사이트 제공
- Production 환경을 고려한 안정적인 MLOps 인프라 구축

### ✨ 주요 특징

- **완전 자동화된 ML 파이프라인**: Airflow 기반 오케스트레이션으로 데이터 수집부터 모델 배포까지 자동화
- **실험 추적 및 모델 관리**: MLflow를 활용한 체계적인 실험 관리 및 모델 버전 관리
- **시계열 예측 모델**: LSTM, Prophet 등 다양한 시계열 모델 앙상블
- **실시간 예측 서비스**: FastAPI 기반 웹 대시보드를 통한 예측 결과 시각화
- **컨테이너화된 환경**: Docker Compose로 일관된 개발/운영 환경 제공

<br>

## 👨‍👩‍👦‍👦 팀 구성원

| ![최해혁](https://avatars.githubusercontent.com/u/156163982?v=4) | ![이재윤](https://avatars.githubusercontent.com/u/156163982?v=4) | ![박준영](https://avatars.githubusercontent.com/u/156163982?v=4) | ![이동건](https://avatars.githubusercontent.com/u/156163982?v=4) | 
|:-------------------------------------------------------------:|:-------------------------------------------------------------:|:-------------------------------------------------------------:|:-------------------------------------------------------------:| 
|            [최해혁](https://github.com/hyuk12)             |            [이재윤](https://github.com/LEEJY0126)             |            [박준영](https://github.com/juny79)             |            [이동건](https://github.com/dg5407)             |
|                    팀장, 데이터 파이프라인 & 시계열 모델링                    |                MLflow 서버 구축 및 API 서빙 및 웹 대시보드                 |                             MLflow 서버 구축 및 API 서빙 및 웹 대시보드                             |                             데이터 파이프라인 & 시계열 모델링                             |                            |

<br>

## 🔨 기술 스택

### Core

- **Language**: Python 3.11+
- **Containerization**: Docker, Docker Compose
- **Cloud Services**: AWS S3, RDS (PostgreSQL)

### ML/MLOps

- **Workflow Orchestration**: Apache Airflow 2.0+
- **Experiment Tracking**: MLflow
- **Machine Learning**: scikit-learn, XGBoost, LightGBM
- **Time Series**: Prophet, LSTM, statsmodels
- **Data Processing**: Pandas, NumPy, PySpark

### Backend & API

- **Web Framework**: FastAPI
- **Database**: PostgreSQL
- **Visualization**: Plotly, Matplotlib, Seaborn

### DevOps & Collaboration

- **Version Control**: Git, GitHub
- **Collaboration**: GitHub Projects, Notion

<br>

## 📁 프로젝트 구조
```
covid-mlops-project/
├── data-pipeline/              # 데이터 수집 및 전처리 파이프라인
│   ├── src/
│   │   ├── collectors/         # 데이터 수집 모듈
│   │   ├── data_processing/    # 전처리 모듈
│   │   ├── feature_engineering/# 피처 엔지니어링
│   │   ├── modeling/           # 모델 학습 및 평가
│   │   ├── pipeline/           # MLflow 파이프라인
│   │   ├── config/             # 설정 파일
│   │   └── utils/              # 유틸리티 함수
│   └── dags/                   # Airflow DAG 정의
│       ├── covid_pipeline_dag.py      # 학습 파이프라인
│       └── batch_prediction_dag.py    # 배치 예측
│
├── models/                     # 모델 개발 및 실험
│   ├── src/                    # 모델 코드
│   └── notebooks/              # EDA 및 실험 노트북
│
├── mlflow-server/             # MLflow 서버 구성
│   └── docker/
│
├── api-service/               # FastAPI 서빙 서비스
│   ├── src/                   # API 코드
│   └── templates/             # 웹 대시보드 템플릿
│
├── infrastructure/            # 인프라 설정
│   └── docker/
│       ├── Dockerfile.analysis
│       ├── docker-compose.yml
│       └── requirements.txt
│
└── docs/                      # 프로젝트 문서
    ├── FE-GUIDE.md
    ├── data-processing-guide.md
    └── 도커_사용방법.md
```

<br>

## 💻 주요 기능

### 1. 자동화된 데이터 파이프라인

- **데이터 수집**: Our World in Data의 COVID-19 데이터를 자동으로 수집
- **전처리**: 결측치 처리, 이상치 제거, 데이터 정규화
- **피처 엔지니어링**:
    - 시간 기반 피처 (요일, 주차, 월)
    - Lag 피처 (1일, 7일, 14일 전)
    - Rolling Statistics (이동평균, 표준편차)
    - 차분 및 변화율 계산
    - Future-Aware 피처 (미래 예측 시점에도 계산 가능한 피처만 사용)

### 2. 시계열 예측 모델

- **모델 앙상블**:
    - Random Forest
    - Gradient Boosting (XGBoost, LightGBM)
    - Linear Regression
    - Ridge Regression
- **예측 전략**:
    - 재귀적 다단계 예측 (H-step ahead)
    - 동적 피처 업데이트
    - 최근 60일 홀드아웃 검증

### 3. MLflow 실험 관리

- 모든 실험 자동 로깅 (파라미터, 메트릭, 아티팩트)
- 모델 버전 관리 및 스테이지 관리 (Staging, Production)
- 데이터셋 추적 및 계보 관리
- S3 기반 아티팩트 저장소

### 4. 실시간 예측 서비스

- FastAPI 기반 RESTful API
- 웹 대시보드를 통한 예측 결과 시각화
-
<br>

## 🛠️ 작품 아키텍처(필수X)
- #### _아래 이미지는 예시입니다_
![이미지 설명](https://miro.medium.com/v2/resize:fit:4800/format:webp/1*ub_u88a4MB5Uj-9Eb60VNA.jpeg)

<br>

## 🚨 트러블 슈팅

### 1. MLflow 서버 연결 실패

**문제**

`MlflowException: Failed to connect to MLflow tracking server`

**해결**

- `.env` 파일의 `MLFLOW_TRACKING_URI` 확인
- MLflow 컨테이너 상태 확인: `docker-compose ps`
- 로컬 파일 시스템으로 폴백: `MLFLOW_TRACKING_URI=file:/tmp/mlruns`

### 2. AWS S3 권한 오류

**문제**

`botocore.exceptions.NoCredentialsError`

**해결**

- AWS 자격 증명 확인 (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`)
- IAM 정책에서 S3 버킷 접근 권한 확인
- 로컬 개발 시 파일 시스템 사용 권장

### 3. 메모리 부족 오류

**문제**

`MemoryError: Unable to allocate array`

**해결**

- Docker 메모리 할당 증가 (Docker Desktop 설정)
- 데이터 청크 처리 사용: `CHUNK_SIZE` 설정 조정
- 피처 개수 줄이기 또는 샘플링 적용

<br>

## 📌 프로젝트 회고
### 박패캠
- _프로젝트 회고를 작성해주세요_

<br>

## 📚 참고 자료

### 데이터 출처

- [Our World in Data - COVID-19 Dataset](https://ourworldindata.org/covid-cases)
- [WHO COVID-19 Dashboard](https://covid19.who.int/)

### 기술 문서

- [Airflow Documentation](https://airflow.apache.org/docs/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Prophet Documentation](https://facebook.github.io/prophet/)
