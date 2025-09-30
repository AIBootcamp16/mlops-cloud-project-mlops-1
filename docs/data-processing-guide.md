# COVID-19 데이터 전처리 파이프라인

`data_processing.ipynb`: 노트북에서 데이터 전처리 로직 개발
`data_preprocessor.py`: 자동화를 위해 스크립트로 모듈화

이 파이프라인은 원본 데이터를 머신러닝 모델 학습에 적합한 형태로 가공하며, 모든 전처리 과정과 결과물을 **MLflow**로 추적하고 관리합니다.

---

## 주요 기능

- **자동화된 전처리 파이프라인**: 원본 CSV 데이터를 입력받아 데이터를 전처리하여 출력하는 전 과정을 자동화합니다.
- **MLflow 통합**: 전처리 과정의 모든 단계, 파라미터, 결과물(데이터셋, 메타데이터)을 MLflow에 자동으로 로깅하여 실험의 재현성과 추적성을 보장합니다.
- **유연한 데이터 로딩**: MLflow 아티팩트, 로컬 CSV 파일 경로, 가장 최신 데이터 수집 런 등 다양한 소스에서 데이터를 유연하게 로드할 수 있습니다.
- **체계적인 결측치 처리**: 데이터의 특성을 고려하여 다양한 결측치 처리 전략(제거, 보간, 채우기)을 순차적으로 적용합니다.
- **피처 엔지니어링**: 날짜 데이터로부터 `년`, `월`, `일`, `요일`, `주말 여부` 등 다양한 파생 변수를 생성하여 모델 성능 향상을 꾀합니다.

---

## 전처리 파이프라인 실행 단계

`data_preprocessor.py`의 `_execute_preprocessing_pipeline` 메소드에 정의된 전처리 단계는 다음과 같습니다.

1.  **날짜 처리 및 시간 특성 추가 (`_process_datetime_features`)**
    - `date` 컬럼을 `datetime` 객체로 변환합니다.
    - `year`, `month`, `day`, `day_of_week` (요일), `is_weekend` (주말 여부) 컬럼을 생성합니다.

2.  **결측치가 많은 컬럼 제거 (`_remove_high_missing_columns`)**
    - 결측치 비율이 **50% 이상**인 컬럼들을 제거하여 데이터의 노이즈를 줄입니다.
    - 모델링에 필수적인 주요 컬럼들은 제거 대상에서 제외됩니다.

3.  **누적 데이터 결측치 보완 (`_fill_cumulative_columns`)**
    - `total_cases`와 같이 시간이 지남에 따라 누적되는 값의 결측치는 앞/뒤 데이터를 사용하여 채워넣습니다 (`bfill` -> `ffill`). 이는 누적 데이터의 연속성을 보존하는 데 도움을 줍니다.

4.  **불필요한 메타데이터 컬럼 제거 (`_remove_metadata_columns`)**
    - `collected_at`, `data_source`, `collector`와 같이 분석에 직접적으로 사용되지 않는 메타데이터 컬럼을 제거합니다.

5.  **범주형 데이터 인코딩 (`_encode_categorical_columns`)**
    - `country`, `continent` 등 문자열로 이루어진 범주형 데이터를 머신러닝 모델이 이해할 수 있도록 **원-핫 인코딩** 방식으로 변환합니다.

6.  **나머지 결측치 보간 (`_interpolate_missing_values`)**
    - 위 단계를 거치고도 남아있는 소수의 수치형 데이터 결측치는 **선형 보간(Linear Interpolation)**을 통해 채워넣어 데이터의 연속성을 유지합니다.

---

## 사용 방법

### 1. 환경 설정

필요한 라이브러리를 설치합니다.

```bash
pip install pandas mlflow numpy
```

MLflow 트래킹 서버를 실행합니다.

```bash
mlflow ui
```

### 2. 스크립트 실행

`data_preprocessor.py` 파일은 직접 실행할 수 있도록 구성되어 있습니다. 터미널에서 아래와 같이 실행하면, MLflow의 가장 최신 데이터 수집 실험(`covid_data_collection`)에서 원본 데이터를 자동으로 가져와 전처리를 수행합니다.

```bash
python -m src.data_processing.data_preprocessor
```

### 3. 다양한 실행 옵션

`CovidDataPreprocessor` 클래스의 `run` 메서드를 통해 다양한 옵션으로 전처리를 수행할 수 있습니다.

-   **최신 데이터로 실행**:
    ```python
    from src.data_processing.data_preprocessor import CovidDataPreprocessor

    preprocessor = CovidDataPreprocessor()
    processed_df = preprocessor.run(
        run_name="my_preprocessing_run",
        from_latest=True
    )
    ```

-   **로컬 CSV 파일로 실행**:
    ```python
    processed_df = preprocessor.run(
        run_name="local_csv_processing",
        input_csv_path="./data/raw/my_covid_data.csv"
    )
    ```

-   **MLflow 아티팩트 URI로 실행**:
    ```python
    processed_df = preprocessor.run(
        run_name="artifact_processing",
        input_artifact_uri="runs:/<RUN_ID>/raw_data"
    )
    ```

전처리가 완료되면 최종 데이터프레임이 반환되고, 모든 과정과 결과물은 MLflow 트래킹 서버에 기록됩니다.