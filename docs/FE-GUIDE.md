# Feature Engineering Guide 
— Robust Integrated Covid Trainer

**어떤 피처를 어떻게 만들고**, **왜 그렇게 설계했는지**
---

## 목표와 설계 철학

**목표**  
시계열(일자별) 코로나 지표(기본값: `new_cases`)를 예측하기 위해 **미래 시점에서도 계산 가능한(future-aware)** 피처만으로 안정적인 모델을 학습·서빙합니다.

**핵심 원칙**
- **데이터 누수(look-ahead/leakage) 방지:** 예측 시점에 사용할 수 없는 정보 제거 → `lag/rolling`, 삼각함수 기반 달력 피처만 사용
- **강건성(robustness):** 파일 인코딩/구분자 자동 탐지, 잘못된 경로 보정, MLflow 서버 불능 시 **로컬 파일 스토어 자동 폴백**
- **실전성(practical):** 최근 N일 고정 평가(set-aside) + (옵션) 시계열 교차검증, **동적 재귀 예측(H-step)** 으로 운영 조건 모사

---

## 전체 변환 흐름 (High-level Pipeline)

**1.원시 데이터 → 기본 정리**

read_table_robust: CSV/TSV/Excel 자동 파싱(구분자/인코딩)

add_time_features: date 정규화/파싱/주기 피처

**2.피처 엔지니어링**

시간 기반 기본 피처(주기·계절성)

타깃 유도 시계열 피처(lag/roll/diff/pct)

미래 안전(future-aware) 화이트리스트 선택

**3.학습/검증/예측**

y_next = target.shift(-1)

최근 test_days 고정 평가 + (옵션) TimeSeriesSplit

모델 비교 후 베스트 선택

H-step 재귀 예측(recursive_forecast_dynamic)


---

## 1) 원시 데이터 → 기본 정리

- **원시 데이터:** `covid_processed.csv`  
  (기본 탐색 경로: `/mnt/data`, `./`, `./data`, `./dataset`)
- **자동 로더:** `read_table_robust`  
  - CSV/TSV/Excel 지원, 인코딩·구분자 추론, JSON 오인 방지
- **기본 정리**
  - **날짜 컬럼 정규화:** `add_time_features()`에서 `date` 존재 확인 및 대체명(`Date`, `ds`, `DATE`, `날짜`) 자동 rename  
  - `pd.to_datetime(..., errors="coerce")`로 파싱 실패값 `NaT` 처리  
  - `date` 기준 정렬

---

## 2) 피처 엔지니어링

### 2-A. 시간 기반 기본 피처 (주기·계절성)

`add_time_features()`가 생성:
- `dow`(요일 0–6), `weekofyear`(ISO 주차 1–53), `dayofyear`(1–366), `month`(1–12)
- **사인/코사인 임베딩(주기 정보 보존)**
  - `dow_sin = sin(2π * dow / 7)`, `dow_cos = cos(2π * dow / 7)`
  - `month_sin = sin(2π * month / 12)`, `month_cos = cos(2π * month / 12)`

### 2-B. 타깃 유도 시계열 피처

`add_lag_roll(target="new_cases")`가 생성:
- **Lag(지연):** `new_cases_lag1`, `new_cases_lag7`, `new_cases_lag14`
- **Rolling 통계(전일로 shift(1) 후 창 통계):**
  - `new_cases_rollmean7/14/28`, `new_cases_rollstd7/14/28`
- **변화율·차분:** `new_cases_diff1`(1일 차분), `new_cases_pct`(전일 대비 증감률)
- **결측/연속성 보정:** 수치형 전체에 `interpolate("linear")` + `bfill()` + `ffill()`

### 2-C. 미래 안전(future-aware) 화이트리스트

`select_future_aware_features()`가 **미래 시점에도 계산 가능한 열만** 선택
- **허용:** `dow_*`, `month_*`, `weekofyear`, `dayofyear` + 타깃 접두(`new_cases_`)로 시작하는 `lag/rolling/diff/pct`
- **제외:** `target` 그 자체(당일 값)와 `y_next`(학습용 타깃)

---

## 3) 학습·검증·예측

- **타깃 시프트:** `y_next = target.shift(-1)` (다음날 예측 학습)
- **평가:** 최근 `test_days`(기본 60일) 고정 평가 + (옵션) `TimeSeriesSplit`
- **모델 후보:** RF, GBRT, Linear, Ridge, (옵션) XGBoost
- **베스트 선택:** 테스트 RMSE 최소 모델
- **H-step 재귀 예측:** `recursive_forecast_dynamic()`
  - 마지막 관측 이후 `horizon`일 만큼 날짜 생성
  - 각 미래일에 대해 **직전까지의 예측값**으로 `lag/rolling/diff/pct` 동적 업데이트 → 누수 없이 다단계 예측

---

## 데이터셋 스키마 (카테고리별)

> 실제 열 이름은 `target` 파라미터에 따라 `new_cases_*` 또는 사용자 지정 접두로 생성됩니다.


### A. 기본/키 컬럼

| 컬럼      | 타입       | 설명                                                        |
|-----------|------------|-------------------------------------------------------------|
| `date`    | datetime64 | 일자(정렬 및 기준 컬럼)                                     |
| `target`  | float      | 당일 타깃 값(예: `new_cases`) — **모델 입력에서 제외**, 타깃 생성용 |
| `y_next`  | float      | 다음날 타깃 (`target.shift(-1)`) — **학습 타깃**            |

### B. 시간 기본 피처 (미래 계산 가능)

| 컬럼                      | 타입  | 설명                    |
|--------------------------|-------|-------------------------|
| `dow_sin`, `dow_cos`     | float | 요일 주기 사인/코사인   |
| `weekofyear`             | int   | ISO 주차                |
| `dayofyear`              | int   | 연중 일수               |
| `month_sin`, `month_cos` | float | 월 주기 사인/코사인     |

### C. 타깃 유도 피처 (미래 계산 가능)

| 컬럼                             | 타입  | 설명                              |
|----------------------------------|-------|-----------------------------------|
| `{t}_lag1`,`{t}_lag7`,`{t}_lag14`| float | t일 전 값                         |
| `{t}_rollmean7/14/28`            | float | 전일까지 이동평균(창 7/14/28)      |
| `{t}_rollstd7/14/28`             | float | 전일까지 이동표준편차(창 7/14/28)  |
| `{t}_diff1`                      | float | 전일 대비 차분                    |
| `{t}_pct`                        | float | 전일 대비 증감률                  |

> `{t}`는 타깃 접두(예: `new_cases_`)를 의미합니다.

---

## 실제 변환 예시

**원시 입력(간소화)**

| date       | new_cases |
|------------|-----------|
| 2025-01-01 | 100       |
| 2025-01-02 | 120       |
| 2025-01-03 | 90        |
| 2025-01-04 | NaN       |
| 2025-01-05 | 110       |

- `add_time_features` 적용:  
  `dow=[2,3,4,5,6]`, `month=[1,1,1,1,1]` → `dow_sin/cos`, `month_sin/cos`, `weekofyear`, `dayofyear`
- `add_lag_roll("new_cases")` (결측 보정 포함):  
  `new_cases_lag1 = [NaN,100,120,90,90] → 보간/ffill 후 [100,100,120,90,90]`  
  `new_cases_rollmean7`: `shift(1)` 뒤 창 평균, `rollstd` 동일  
  `new_cases_diff1`, `new_cases_pct`: 전일 변화/증감률 계산

**학습 타깃:** `y_next = new_cases.shift(-1)`  

**미래 안전 피처만 선택:** 
`dow_*`, `month_*`, `weekofyear`, `dayofyear`, `new_cases_*`  
  *(→ `y_next`, 당일 `new_cases`는 제외)*

---

## 수식 & 핵심 함수

**주기 임베딩(누수 없이 계절성 인코딩)**
dow_sin = sin(2π * dow / 7), 
dow_cos = cos(2π * dow / 7)
month_sin = sin(2π * month / 12), 
month_cos = cos(2π * month / 12)

**이동 통계(shift(1)로 미래 누수 차단)**
rollmean_w(t) = mean(target[t-1], target[t-2], …, target[t-w])
rollstd_w(t) = std(same_window)

**변화율/차분**
diff1(t) = target[t] - target[t-1]
pct(t) = (target[t] - target[t-1]) / max(|target[t-1]|, 1e-9)

**재귀 예측(`recursive_forecast_dynamic`)**
- 미래일 `t+1`을 예측 → 그 값을 다시 `lag/roll/diff/pct` 계산에 반영 → `t+2` 예측 … 반복  
- **미래에도 계산 가능한 구조**를 코드로 강제하여 운영 일관성 확보

---

## 왜 이런 피처 구성이 성능에 유리한가?

- **누수 방지:** 실사용 성능과 오프라인 점수의 괴리 최소화  
- **주기·계절성 반영:** 보고 지연/휴일 효과 등 주간·월간 패턴을 sin/cos 임베딩으로 반영  
- **최근 추세 반영:** `lag/rolling/diff/pct`가 최근 레벨·변동성·추세를 요약해 급증/급감 포착  
- **강건성:** 보간 + `ffill/bfill`로 결측/이상에 덜 민감

---

## 성능 최적화를 위한 고려 사항

- **Feature 수 ↔ 표본 수 균형:** 창(7/14/28)·lag(1/7/14) 규모 조정으로 과대적합 방지  
- **정규화/스케일링:** 선형류(Ridge)에서 표준화가 유리할 수 있음(본 스크립트는 기본 미적용)  
- **시계열 CV:** `TimeSeriesSplit` 사용, `cv_splits`는 데이터 길이에 맞게 조절  
- **모델 선택 가이드:**  
  - RF/GBRT: 비선형·상호작용 포착, 스케일 민감도 낮음  
  - Ridge/Linear: 빠르고 해석 용이  
  - XGBoost: 데이터 충분·복잡 패턴에서 추가 이득(학습시간 증가)
- **예측 지평(Horizon):** H가 커질수록 오차 누적 → 더 긴 창 평균 비중 확대 등 보수적 전략 고려




