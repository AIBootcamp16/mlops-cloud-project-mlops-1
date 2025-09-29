# Robust Integrated Covid Trainer — Feature Engineering Guide

**어떤 피처를 어떻게 만들고**, **왜 그렇게 설계했는지**

미래 시점에도 계산 가능한 future‑aware 피처만으로 안정적 시계열 예측을 수행하는 프로젝트의 FE-GUIDE 입니다.




<p align="center">

<img src="https://img.shields.io/badge/ML-Feature%20Engineering-blue" />

<img src="https://img.shields.io/badge/Time%20Series-Leakage%20Safe-green" />

<img src="https://img.shields.io/badge/MLflow-Tracking-orange" />

<img src="https://img.shields.io/badge/License-MIT-lightgrey" />

</p>

---

## 목표와 설계 철학

- **목표**: 시계열(일자별) 코로나 지표(기본값: `new_cases`)를 예측하기 위해 **미래 시점에서도 계산 가능한(future-aware)** 피처만으로 안정적인 모델을 학습·서빙한다.
- **핵심 원칙**
    1. **데이터 누수(look-ahead/leakage) 방지**: 예측 시점에 사용할 수 없는 정보 제거 → lag/rolling, trigonometric date features만 사용.
    2. **강건성(robustness)**: 파일 인코딩/구분자 자동 탐지, 잘못된 경로 보정, MLflow 서버 불능 시 **로컬 파일 스토어로 자동 폴백**.
    3. **실전성(practical)**: 최근 N일 고정 평가(set-aside) + (옵션) 시계열 교차검증, **동적 재귀 예측(H-step)** 로 실제 운영 조건을 모사.

---

## 전체 변환 흐름 (High-level Pipeline)

### 1) 원시 데이터 → 기본 정리

- **원시 데이터**: `covid_processed.csv` (기본 탐색 경로: `/mnt/data`, `./`, `./data`, `./dataset`)
    - 자동 로더가 CSV/TSV/Excel을 인코딩/구분자 추론으로 파싱 (`read_table_robust`)
- **기본 정리**
    - 날짜 컬럼 정규화: `add_time_features()`에서 `date` 존재 확인 및 대체명(`Date`, `ds`, `DATE`, `날짜`) 자동 rename
    - [`pd.to](http://pd.to)_datetime(..., errors="coerce")`로 파싱 실패값 NaT 처리
    - `date` 기준 정렬

### 2) 피처 엔지니어링

- **시간 기반 기본 피처 (주기·계절성)**
    
    `add_time_features()`가 아래를 생성:
    
    - `dow`: 요일(0–6)
    - `weekofyear`: ISO 주차(1–53)
    - `dayofyear`: 연중 일수(1–366)
    - `month`: 월(1–12)
    - **사인/코사인 임베딩** (주기 정보 손실 최소화)
        - `dow_sin = sin(2π * dow / 7)`, `dow_cos = cos(2π * dow / 7)`
        - `month_sin = sin(2π * month / 12)`, `month_cos = cos(2π * month / 12)`
- **타깃 유도 시계열 피처**
    
    `add_lag_roll(target="new_cases")`가 아래를 생성:
    
    - **Lag** (지연): `new_cases_lag1`, `lag7`, `lag14`
    - **Rolling stats** (shift(1) 이후 창 평균/표준편차):
        
        `new_cases_rollmean7/14/28`, `new_cases_rollstd7/14/28`
        
    - **변화율·차분**: `new_cases_diff1`(1일차분), `new_cases_pct`(전일 대비 증감률)
    - **결측/연속성 보정**: 수치형 전체에 대해 `interpolate("linear")`, `bfill().ffill()` 적용
- **미래 안전(future-aware) 화이트리스트**
    
    `select_future_aware_features()`가 **미래 시점에도 계산 가능한 열만 선택**
    
    - 허용: `dow_*`, `month_*`, `weekofyear`, `dayofyear` **+** `target` 접두(`new_cases_*`)로 시작하는 lag/rolling/diff/pct
    - 제외: `target` 그 자체(당일 값)와 `y_next`(학습용 타깃)

### 3) 학습/검증/예측

- **타깃 시프트**: 다음날 예측을 위해 `y_next = target.shift(-1)`
- **홀드아웃**: 최근 `test_days`(기본 60일) 고정 평가 + (옵션) `TimeSeriesSplit`
- **모델 앙상블 후보**: RF, GBRT, Linear, Ridge, (옵션) XGBoost
- **베스트 모델 선택**: 테스트 RMSE 최소 모델
- **H-step 재귀 예측**: `recursive_forecast_dynamic()`
    - 마지막 관측 이후 `horizon`일 만큼 날짜를 생성
    - 각 미래일에 대해 **직전까지의 예측값**으로 lag/rolling/diff/pct **동적 업데이트** → 누수 없이 다단계 예측

---

## 데이터셋 스키마 (카테고리별)

> 실제 열 이름은 target 파라미터에 따라 new_cases_* 또는 사용자가 지정한 타깃 접두로 생성됩니다.
> 

### A. 기본/키 컬럼

| 컬럼 | 타입 | 설명 |
| --- | --- | --- |
| `date` | datetime64 | 일자(정렬 및 기준 컬럼) |
| `target` | float | 당일 타깃 값(예: `new_cases`) — **모델 입력에서 제외**, 타깃 생성에만 사용 |
| `y_next` | float | 다음날 타깃 (`target.shift(-1)`) — **학습 타깃** |

### B. 시간 기본 피처 (미래 계산 가능)

| 컬럼 | 타입 | 설명 |
| --- | --- | --- |
| `dow_sin`, `dow_cos` | float | 요일 주기 사인/코사인 |
| `weekofyear` | int | ISO 주차 |
| `dayofyear` | int | 연중 일수 |
| `month_sin`, `month_cos` | float | 월 주기 사인/코사인 |

### C. 타깃 유도 피처 (미래 계산 가능)

| 컬럼 | 타입 | 설명 |
| --- | --- | --- |
| `{t}_lag1`, `{t}_lag7`, `{t}_lag14` | float | t일 전 값 |
| `{t}_rollmean7/14/28` | float | 전일까지 이동평균(창 7/14/28) |
| `{t}_rollstd7/14/28` | float | 전일까지 이동표준편차(창 7/14/28) |
| `{t}_diff1` | float | 전일 대비 차분 |
| `{t}_pct` | float | 전일 대비 증감률 |

> {t}는 타깃 접두(예: new_cases_)를 의미.
> 

### D. 확장 제안(옵션)

- **지역 기반 피처**: 지역/행정구역/이동량/POI 지표(예: `region_id`, `population_density`, `mobility_index`)
    
    → *단, 미래 시점에 이용 가능한 형태(예: 과거 관측으로만 구성된 lag/rolling 또는 캘린더성 static 정보)로 설계해야 함.*
    
- **정책/이벤트 피처**: 방역 단계, 공휴일/연휴, 대형 이벤트 더미 등(캘린더성 또는 공개되는 고정 스케줄)

---

## 실제 변환 예시

원시 입력(간소화):

| date | new_cases |
| --- | --- |
| 2025-01-01 | 100 |
| 2025-01-02 | 120 |
| 2025-01-03 | 90 |
| 2025-01-04 | NaN |
| 2025-01-05 | 110 |
1. `add_time_features` 적용:
- `dow` = [2,3,4,5,6], `month` = [1,1,1,1,1]
- `dow_sin = sin(2π * dow/7)`, `dow_cos = cos(2π * dow/7)`
- `month_sin = sin(2π * 1/12)`, `month_cos = cos(2π * 1/12)` 등 추가
1. `add_lag_roll("new_cases")` 적용 (결측 보정 포함):
- `new_cases_lag1` = [NaN,100,120,90,90] → 보간/ffill 후 [100,100,120,90,90]
- `new_cases_rollmean7` 등: `shift(1)` 뒤 창 평균
- `new_cases_diff1` = [NaN,20,-30,?,?] → 보간 후 연속화
- `new_cases_pct` = [(120-100)/100=0.2, (90-120)/120=-0.25, …]
1. 학습 데이터 생성:
- `y_next = new_cases.shift(-1)`
    - (2025-01-01 행의 타깃은 2025-01-02의 `new_cases`)
1. **미래 안전 피처만 선택**: `dow_*`, `month_*`, `weekofyear`, `dayofyear`, `new_cases_*` 접두 열 → **`y_next`**, **`new_cases`**(당일값) 제외

---

## 수식/핵심 함수 해설

### 주기 임베딩 (누수 없이 계절성 인코딩)

- 요일/월은 범주형이지만 **각도 공간의 연속성**을 보존하기 위해 사인/코사인 사용:
    - `dow_sin = sin(2π * dow / 7)`, `dow_cos = cos(2π * dow / 7)`
    - `month_sin = sin(2π * month / 12)`, `month_cos = cos(2π * month / 12)`

### 이동 통계 (shift(1)로 미래 누수 차단)

- `rollmean_w(t) = mean( target[t-1], target[t-2], …, target[t-w] )`
- `rollstd_w(t) = std( same_window )`

### 변화율/차분

- `diff1(t) = target[t] - target[t-1]`
- `pct(t) = (target[t] - target[t-1]) / max(|target[t-1]|, 1e-9)`

### 재귀 예측(`recursive_forecast_dynamic`)

- 미래일 `t+1`을 예측하면, 그 값을 다시 lag/roll/diff/pct 계산에 반영 → `t+2` 예측 … 반복
- 미래에서도 계산 가능한 구조를 **코드로 강제**하여 운영 일관성 확보

---

## 왜 이런 피처 구성이 성능에 유리한가?

1. **누수 방지**: 모델이 보지 못할 미래 데이터를 제거 → 실사용 성능과 오프라인 점수의 괴리 최소화
2. **주기·계절성 반영**: 전염병 데이터는 주간/월간 주기가 강함(보고 지연/휴일 효과 등) → 사인/코사인 임베딩으로 비선형 모델/선형 모델 모두에 유리
3. **최근 추세 반영**: lag/rolling/diff/pct은 최근 레벨/변동성/추세를 요약 → 급증/급감 포착
4. **강건성**: 선형 보간 + ffill/bfill로 학습 불안정(결측/이상) 최소화

---

## 성능 최적화를 위한 고려 사항

- **Feature Count vs. Sample Size**: 창 크기(7/14/28)와 lag 수(1/7/14)는 과대적합 위험과 데이터 길이 균형 고려
- **정규화/스케일링**: 선형류 모델(Ridge)은 표준화 시 유리할 수 있음(본 스크립트는 안전·단순성을 위해 생략)
- **시계열 CV**: `TimeSeriesSplit`으로 데이터 누수 없는 검증. 분할 수(`cv_splits`)는 데이터 길이에 맞게 조정
- **모델 선택**:
    - **RF/GBRT**: 비선형·상호작용 포착, 결측/스케일 민감도 낮음
    - **Ridge/Linear**: 빠르고 해석 용이, 적은 피처에도 견고
    - **XGBoost**(옵션): 큰 데이터/복잡 패턴에서 추가 이득 가능(학습시간↑)
- **예측 지평(Horizon)**: 멀리 갈수록 오차 누적 → H가 크면 보수적 피처(긴 창 평균 등) 비중을 늘리는 것도 전략

---

## 확장 가이드 (도메인 피처)

- **정책/캘린더**: 공휴일/연휴/주말 더미, 등교/재택 정책 단계
    - *주의*: 미래 사용 가능성(고정/사전공개 스케줄) 보장 필요
- **이동/집객**: 교통량, 대중교통 승하차, 상권 유동인구 — 과거 데이터로 **lag/rolling** 변환하여 future-aware로 만들기
- **기상**: 기온/습도/강수량 → 지역 종속 시 **지역 ID + lag/roll** 조합

---

## 코드에서 참고할 핵심 함수

```python
# 날짜 파싱 및 주기 임베딩
df = add_time_features(df, date_col="date")

# 누수 방지 시계열 피처
df = add_lag_roll(df, target="new_cases", lags=(1,7,14), rolls=(7,14,28))

# 미래에도 계산 가능한 컬럼만 화이트리스트링
feat_list = select_future_aware_features(df, target="new_cases")

# y_next 타깃(다음날 값) 생성
df["y_next"] = df["new_cases"].shift(-1)
```

---

## 빠른 확인 체크리스트

- [ ]  `date` 컬럼 정상 파싱 및 정렬
- [ ]  shift(1) 적용된 rolling으로 누수 방지
- [ ]  결측 보간과 ffill/bfill 적용
- [ ]  미래 계산 가능 피처만 최종 선택
- [ ]  H가 길수록 보수적 창 설계 고려

---

## 결론

본 구성은 **미래에도 계산 가능한 정보만** 사용하도록 설계되어, 오프라인 점수와 운영 환경 성능의 **일치**를 최대화합니다. 주기성(요일/월), 최근 레벨·변동성(lag/rolling), 추세(diff/pct)를 결합해 **간결하고 강건한 베이스라인**을 제공합니다. 도메인 피처를 future‑aware 방식으로 확장하면 추가 향상을 기대할 수 있습니다.
