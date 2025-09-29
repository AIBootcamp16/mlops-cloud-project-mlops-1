# Robust Integrated Covid Trainer — Feature Engineering Guide

미래 시점에도 계산 가능한 future‑aware 피처만으로 안정적 시계열 예측을 수행하는 프로젝트의 FE-GUIDE 입니다.

<p align="center">

<img src="https://img.shields.io/badge/ML-Feature%20Engineering-blue" />

<img src="https://img.shields.io/badge/Time%20Series-Leakage%20Safe-green" />

<img src="https://img.shields.io/badge/MLflow-Tracking-orange" />

<img src="https://img.shields.io/badge/License-MIT-lightgrey" />

</p>

## 목차

- [개요](#개요)
- [핵심 설계](#핵심-설계)
- [파이프라인](#파이프라인)
- [데이터 스키마](#데이터-스키마)
- [설치](#설치)
- [빠른 시작](#빠른-시작)
- [구성 옵션](#구성-옵션)
- [예측과 운영](#예측과-운영)
- [프로젝트 구조](#프로젝트-구조)
- [성능 팁](#성능-팁)
- [자주 묻는 질문](#자주-묻는-질문)
- [라이선스](#라이선스)

---

## 개요

- 목표: 시계열 코로나 지표(기본 `new_cases`)를 누수 없이 예측
- 접근: 미래에도 계산 가능한 future‑aware 피처만 사용
- 운영: 최근 N일 홀드아웃 평가와 H‑step 재귀 예측으로 실제 환경을 모사

주요 사용처

- 베이스라인 모델링
- MLOps 파이프라인의 피처 엔지니어링 표준화
- 운영 일관성 검증과 실험 추적(MLflow)

---

## 핵심 설계

- 데이터 누수 방지
    - 당일 또는 미래 값을 피처에 사용하지 않음
    - lag, rolling, diff, pct, 캘린더 사인/코사인 임베딩만 허용
- 강건성
    - 파일 인코딩/구분자 자동 추론
    - 잘못된 경로 보정
    - MLflow 서버 불가 시 file store로 자동 폴백
- 실전성
    - 최근 N일 고정 평가 + 필요 시 TimeSeriesSplit
    - H‑step 재귀 예측으로 다단계 예측 구현

---

## 파이프라인

1) 입력 로드

- 기본 입력: `covid_processed.csv`
- 탐색 경로: `/mnt/data`, `./`, `./data`, `./dataset`
- CSV/TSV/Excel 자동 파싱

2) 기본 정리

- 날짜 컬럼 정규화 및 파싱 실패값 NaT 처리
- `date` 기준 정렬

3) 피처 엔지니어링

- 캘린더 피처
    - dow, weekofyear, dayofyear, month
    - 주기 임베딩: `dow_sin`, `dow_cos`, `month_sin`, `month_cos`
- 타깃 유도 피처
    - lag: 1, 7, 14
    - rolling mean/std: 7, 14, 28 창
    - diff1, pct 변화율
    - 선형 보간 + ffill/bfill로 연속성 보정

4) 타깃과 학습

- `y_next = target.shift(-1)`로 다음날 예측
- 최근 `test_days` 고정 평가
- 후보 모델: RF, GBRT, Linear, Ridge, (옵션) XGBoost
- 테스트 RMSE 최소 모델 선택

5) 예측

- `recursive_forecast_dynamic()`로 H‑step 재귀 예측
- 예측값을 순차적으로 lag/rolling에 반영하여 누수 방지

---

## 데이터 스키마

기본/키

- date: datetime
- target: float, 당일 타깃 값. 모델 입력에서 제외
- y_next: float, 다음날 타깃. 학습 타깃

시간 피처

- dow_sin, dow_cos
- weekofyear, dayofyear
- month_sin, month_cos

타깃 유도 피처({t}는 타깃 접두 예: new_cases_)

- {t}_lag1, {t}_lag7, {t}_lag14
- {t}_rollmean7,14,28
- {t}_rollstd7,14,28
- {t}_diff1, {t}_pct

미래 안전 화이트리스트

- 허용: 위 시간 피처 + `{target}_*` lag/rolling/diff/pct
- 제외: 당일 `target`, `y_next`

---

## 설치

```bash
# 필수
pip install numpy pandas scikit-learn mlflow matplotlib

# 선택
# pip install xgboost
```

---

## 빠른 시작

```bash
python train_[integrated.py](http://integrated.py) \
  --features_path ./covid_processed.csv \
  --target new_cases \
  --test_days 60 \
  --horizon 30
```

MLflow UI 실행 예

- 폴백 파일 스토어: `mlflow ui --backend-store-uri [file:/root/mlruns](file:/root/mlruns) --port 5001`
- 서버 사용 시: `--tracking_uri http://mlflow:8080` 형태로 지정

체크리스트

- [ ]  date 컬럼 정상 파싱 및 정렬
- [ ]  shift(1) 적용으로 누수 방지
- [ ]  결측 보간과 ffill/bfill 적용
- [ ]  test_days가 데이터 길이와 운영 기간에 적정
- [ ]  시계열 CV 분할 수 적정
- [ ]  H가 커질수록 창 길이와 모델 복잡도 조정

---

## 구성 옵션

대표 인자

- features_path: 입력 데이터 경로
- target: 기본 `new_cases`
- test_days: 최근 홀드아웃 일수. 기본 60
- horizon: 예측 지평. 예: 30
- lags: 기본 1, 7, 14
- rolls: 기본 7, 14, 28
- tracking_uri: MLflow 서버. 미지정 시 파일 폴백

추천 규칙

- 데이터가 짧을수록 lags와 rolls를 보수적으로
- H가 커질수록 긴 창 평균 등 안정성 비중 상향

---

## 예측과 운영

- 1‑step 학습(y_next) + 재귀 예측으로 H‑step 확장
- 예측값을 피처 재계산에 반영하여 실제 운영과 동일한 제약 보장
- 주기성은 사인/코사인으로 표현해 모델 유형에 상관없이 활용 가능

수식 참고

- dow_sin = sin(2π  *dow / 7), dow_cos = cos(2π*  dow / 7)
- month_sin = sin(2π  *month / 12), month_cos = cos(2π*  month / 12)
- diff1(t) = target[t] - target[t-1]
- pct(t) = (target[t] - target[t-1]) / max(|target[t-1]|, 1e-9)
- rollmean_w(t) = mean(target[t-1..t-w])

---

## 프로젝트 구조

예시

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

---

## 성능 팁

- Feature Count vs Sample Size
    - lags와 rolls는 데이터 길이에 맞춰 과대적합 방지
- 정규화
    - Ridge 등 선형 모델은 표준화로 이득 가능
- 모델 선택
    - RF/GBRT: 비선형 패턴과 상호작용 포착, 결측·스케일에 둔감
    - Ridge/Linear: 빠르고 해석 용이
    - XGBoost: 대규모·복잡 패턴에 유리하나 시간 비용 증가
- Horizon 전략
    - H가 길수록 보수적 피처 설계 권장

---

## 자주 묻는 질문

Q. 왜 y_next를 쓰나요?

A. 다음날 예측을 명확히 정의해 1‑step 문제로 단순화합니다. H‑step은 재귀로 확장합니다.

Q. 왜 사인/코사인 임베딩인가요?

A. 요일/월처럼 순환적 범주는 각도 공간 연속성을 주면 모델이 주기를 더 잘 학습합니다.

Q. rolling에 shift(1)가 필요한가요?

A. 당일 값을 포함하면 누수입니다. 항상 전일까지 관측으로 창 통계를 만듭니다.
