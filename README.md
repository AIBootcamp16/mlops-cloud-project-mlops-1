# COVID-19 MLOps Pipeline Project

<br>

## 💻 프로젝트 소개
### <프로젝트 소개>
- COVID-19 감염 확산 데이터를 활용한 예측 모델 개발 및 MLOps 파이프라인 구축 프로젝트
- 시계열 데이터 분석을 통한 감염률 예측과 백신 효과성 분석을 수행
- 전체 ML 라이프사이클을 자동화하여 실시간 예측 서비스 제공

### <작품 소개>
- Airflow 기반 데이터 파이프라인과 MLflow를 활용한 모델 관리 시스템
- 시계열 예측 모델(LSTM, Prophet)을 통한 COVID-19 확산 예측
- FastAPI 기반 웹 대시보드를 통한 실시간 예측 결과 시각화

<br>

## 👨‍👩‍👦‍👦 팀 구성원

| ![최해혁](https://avatars.githubusercontent.com/u/156163982?v=4) | ![이재윤](https://avatars.githubusercontent.com/u/156163982?v=4) | ![박준영](https://avatars.githubusercontent.com/u/156163982?v=4) | ![이동건](https://avatars.githubusercontent.com/u/156163982?v=4) | 
|:-------------------------------------------------------------:|:-------------------------------------------------------------:|:-------------------------------------------------------------:|:-------------------------------------------------------------:| 
|            [최해혁](https://github.com/hyuk12)             |            [이재윤](https://github.com/LEEJY0126)             |            [박준영](https://github.com/juny79)             |            [이동건](https://github.com/dg5407)             |
|                    팀장, 데이터 파이프라인 & 시계열 모델링                    |                MLflow 서버 구축 및 API 서빙 및 웹 대시보드                 |                             MLflow 서버 구축 및 API 서빙 및 웹 대시보드                             |                             데이터 파이프라인 & 시계열 모델링                             |                            |

<br>

## 🔨 개발 환경 및 기술 스택
- 주 언어 : Python 3.11+
- 컨테이너 : Docker, Docker Compose
- 오케스트레이션 : Airflow
- 모델 관리 : MLflow
- 데이터베이스 : PostgreSQL
- 웹 프레임워크 : FastAPI
- 버전 및 이슈관리 : GitHub
- 협업 툴 : GitHub, Notion

<br>

## 📁 프로젝트 구조
```
covid-mlops-project/
├── data-pipeline/              # 팀 1: 데이터 수집 및 전처리
│   ├── airflow/
│   └── src/
├── models/                     # 팀 1: 모델 개발
│   ├── src/
│   └── configs/
├── mlflow-server/             # 팀 2: MLflow 구성
│   └── docker/
├── api-service/               # 팀 2: API 서빙
│   ├── src/
│   └── templates/
├── infrastructure/            # 공통: 인프라 설정
└── docker-compose.yml
```

<br>

## 💻​ 구현 기능
### 기능1
- _작품에 대한 주요 기능을 작성해주세요_
### 기능2
- _작품에 대한 주요 기능을 작성해주세요_
### 기능3
- _작품에 대한 주요 기능을 작성해주세요_

<br>

## 🛠️ 작품 아키텍처(필수X)
- #### _아래 이미지는 예시입니다_
![이미지 설명](https://miro.medium.com/v2/resize:fit:4800/format:webp/1*ub_u88a4MB5Uj-9Eb60VNA.jpeg)

<br>

## 🚨​ 트러블 슈팅
### 1. OOO 에러 발견

#### 설명
- _프로젝트 진행 중 발생한 트러블에 대해 작성해주세요_

#### 해결
- _프로젝트 진행 중 발생한 트러블 해결방법 대해 작성해주세요_

<br>

## 📌 프로젝트 회고
### 박패캠
- _프로젝트 회고를 작성해주세요_

<br>

## 📰​ 참고자료
- _참고자료를 첨부해주세요_
