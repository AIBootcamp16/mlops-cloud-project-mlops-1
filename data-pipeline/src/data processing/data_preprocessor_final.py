# src/data_processing/data_preprocessor.py
import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
import mlflow
import mlflow.data
from mlflow.tracking import MlflowClient
from contextlib import contextmanager

from ..config.settings import ProjectConfig
from ..utils.mlflow_utils import MLflowManager, create_data_metadata

class CovidDataPreprocessor:
    """
    MLOps 파이프라인을 위한 데이터 전처리기 (데이터 소싱 + 전처리 로직 통합)
    """

    def __init__(self, config: Optional[ProjectConfig] = None, tracking_uri: Optional[str] = None):
        self.config = config or ProjectConfig(tracking_uri=tracking_uri)
        mlflow.set_tracking_uri(self.config.mlflow.TRACKING_URI)
        self.mlflow_manager = MLflowManager(
            self.config.mlflow.TRACKING_URI, "covid_data_preprocessing"
        )

    @contextmanager
    def safe_mlflow_run(self, run_name: Optional[str] = None):
        if mlflow.active_run():
            print(f"Using existing active run: {mlflow.active_run().info.run_id}")
            yield
        else:
            with mlflow.start_run(run_name=run_name) as run:
                print(f"Started new run '{run_name}' with ID: {run.info.run_id}")
                yield

    def run(self, input_csv_path: Optional[str] = None, from_latest: bool = True, run_name: Optional[str] = None) -> pd.DataFrame:
        """
        주어진 입력 소스에서 데이터를 로드하고 전체 전처리 파이프라인을 실행합니다.
        Airflow 등에서 호출하기 위한 표준 진입점입니다.
        """
        with self.safe_mlflow_run(run_name=run_name or f"preprocessing_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # 1. 데이터 소싱
            raw_df = self._load_data(input_csv_path, from_latest)
            if raw_df is None:
                raise ValueError("Failed to load raw data. Halting pipeline.")

            # 2. 전처리 파이프라인 실행
            processed_df = self._execute_preprocessing_pipeline(raw_df)
            
            # 3. 결과 로깅
            self._log_processing_results(raw_df, processed_df)
            
            print(f"\nPreprocessing finished! Check MLflow UI for run details.")
            return processed_df

    def _load_data(self, input_csv_path: Optional[str], from_latest: bool) -> Optional[pd.DataFrame]:
        """다양한 소스에서 원본 데이터를 로드합니다."""
        if input_csv_path and os.path.exists(input_csv_path):
            print(f"Loading raw data from local path: {input_csv_path}")
            df = pd.read_csv(input_csv_path)
            mlflow.log_param("data_source_type", "local_path")
            mlflow.log_param("input_data_path", input_csv_path)
            return df
        
        if from_latest:
            print("Attempting to load data from the latest MLflow collection run...")
            df = self._load_from_latest_collection()
            if df is not None:
                mlflow.log_param("data_source_type", "mlflow_latest_run")
                return df

        print("Warning: No valid data source found. Creating dummy data for demonstration.")
        mlflow.log_param("data_source_type", "dummy_fallback")
        return self._create_dummy_data()

    def _load_from_latest_collection(self) -> Optional[pd.DataFrame]:
        # 이 메소드는 원본 파일의 복잡한 로직을 그대로 사용하거나,
        # Airflow XComs 등을 통해 명시적으로 URI를 전달받는 방식으로 단순화할 수 있습니다.
        # 여기서는 간단한 버전으로 구현합니다.
        try:
            collection_experiment = mlflow.get_experiment_by_name("covid_data_collection")
            if not collection_experiment:
                print("MLflow experiment 'covid_data_collection' not found.")
                return None
            
            runs = mlflow.search_runs(
                experiment_ids=[collection_experiment.experiment_id],
                filter_string="attributes.status = 'FINISHED'",
                order_by=["start_time DESC"],
                max_results=1
            )
            if runs.empty:
                print("No finished runs found in 'covid_data_collection'.")
                return None
            
            latest_run_id = runs.iloc[0].run_id
            print(f"Found latest collection run: {latest_run_id}")
            mlflow.log_param("source_collection_run_id", latest_run_id)

            # 아티팩트 다운로드
            artifact_path = "raw_data" # 수집 단계에서 저장한 아티팩트 경로
            local_dir = mlflow.artifacts.download_artifacts(run_id=latest_run_id, artifact_path=artifact_path)
            
            csv_files = [f for f in os.listdir(local_dir) if f.endswith(".csv")]
            if not csv_files:
                print(f"No CSV files found in artifact path '{artifact_path}' for run {latest_run_id}.")
                return None

            data_path = os.path.join(local_dir, csv_files[0])
            return pd.read_csv(data_path)
        except Exception as e:
            print(f"Error loading from MLflow: {e}")
            return None

    def _execute_preprocessing_pipeline(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Jupyter Notebook의 전체 전처리 파이프라인을 순서대로 실행합니다."""
        print("\n--- Starting Data Preprocessing Pipeline ---")
        
        # 1-6단계: 기본 전처리 (Notebook 전반부)
        data = (raw_data.copy()
                .pipe(self._process_datetime_features)
                .pipe(lambda df: self._remove_high_missing_columns(df, threshold=0.5)[0])
                .pipe(self._fill_cumulative_columns)
                .pipe(self._remove_metadata_columns)
                .pipe(self._encode_categorical_columns)
                .pipe(self._interpolate_missing_values))

        # 7단계: 유효하지 않은 최신 데이터 컬럼 제거 (Notebook 중반부)
        data = self._remove_recent_zero_columns(data, days=730, threshold=0.5)

        # 8단계: 이상치 처리 (Notebook 후반부)
        data = self._handle_reproduction_rate_outliers(data, method='iqr', multiplier=1.5)
        
        print("--- Pipeline Execution Finished ---")
        return data

    # ... (개별 전처리 메소드들은 이전 답변과 동일하게 유지) ...
    def _process_datetime_features(self, data: pd.DataFrame) -> pd.DataFrame:
        print("Step 1/8: Processing datetime features...")
        data['date'] = pd.to_datetime(data['date'])
        data['year'] = data['date'].dt.year
        data['month'] = data['date'].dt.month
        data['day'] = data['date'].dt.day
        return data

    def _remove_high_missing_columns(self, data: pd.DataFrame, threshold: float) -> Tuple[pd.DataFrame, list]:
        print(f"Step 2/8: Removing columns with >{threshold*100}% missing values.")
        missing_ratio = data.isnull().sum() / len(data)
        cols_to_drop = missing_ratio[missing_ratio > threshold].index.tolist()
        if cols_to_drop: mlflow.log_text("\n".join(cols_to_drop), "removed_high_missing_columns.txt")
        return data.drop(columns=cols_to_drop), cols_to_drop

    def _fill_cumulative_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        print("Step 3/8: Filling missing values in cumulative columns.")
        cumulative_cols = [col for col in data.columns if 'total' in col.lower()]
        if cumulative_cols: data[cumulative_cols] = data[cumulative_cols].bfill().ffill()
        return data

    def _remove_metadata_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        print("Step 4/8: Removing metadata columns.")
        metadata_cols = ['collected_at', 'data_source', 'collector']
        cols_to_remove = [col for col in metadata_cols if col in data.columns]
        if cols_to_remove: data = data.drop(columns=cols_to_remove)
        return data

    def _encode_categorical_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        print("Step 5/8: One-hot encoding categorical columns.")
        cols_to_encode = [col for col in ['country', 'code', 'continent'] if col in data.columns]
        if cols_to_encode: data = pd.get_dummies(data, columns=cols_to_encode, dtype=int)
        return data

    def _interpolate_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        print("Step 6/8: Interpolating remaining missing values.")
        numeric_cols = data.select_dtypes(include=np.number).columns
        data[numeric_cols] = data[numeric_cols].interpolate(method='linear').bfill().ffill()
        return data

    def _remove_recent_zero_columns(self, data: pd.DataFrame, days: int, threshold: float) -> pd.DataFrame:
        print(f"Step 7/8: Removing columns with >{threshold*100}% zeros in the last {days} days.")
        latest_date = data['date'].max()
        cutoff_date = latest_date - pd.Timedelta(days=days)
        recent_data = data[data['date'] >= cutoff_date]
        numeric_cols = recent_data.select_dtypes(include=np.number)
        zero_ratio = (numeric_cols == 0).mean()
        cols_to_drop = zero_ratio[zero_ratio > threshold].index.tolist()
        if cols_to_drop:
            print(f"  - Dropping: {cols_to_drop}")
            data = data.drop(columns=cols_to_drop)
            mlflow.log_param("recent_zero_days_period", days)
            mlflow.log_param("recent_zero_threshold", threshold)
            mlflow.log_text("\n".join(cols_to_drop), "removed_recent_zero_columns.txt")
        return data

    def _handle_reproduction_rate_outliers(self, data: pd.DataFrame, method: str, multiplier: float) -> pd.DataFrame:
        col = 'reproduction_rate'
        print(f"Step 8/8: Handling outliers in '{col}' using {method} method.")
        if col not in data.columns: return data
        q1, q3 = data[col].quantile(0.25), data[col].quantile(0.75)
        upper_bound = q3 + multiplier * (q3 - q1)
        outliers_mask = data[col] > upper_bound
        num_outliers = outliers_mask.sum()
        if num_outliers > 0:
            data.loc[outliers_mask, col] = upper_bound
            mlflow.log_param(f"{col}_outlier_method", method); mlflow.log_param(f"{col}_outlier_multiplier", multiplier); mlflow.log_metric(f"{col}_outliers_handled", num_outliers)
        return data

    def _log_processing_results(self, raw_data: pd.DataFrame, processed_data: pd.DataFrame):
        mlflow.log_metrics({
            "input_rows": len(raw_data), "output_rows": len(processed_data),
            "input_columns": len(raw_data.columns), "output_columns": len(processed_data.columns),
            "final_missing_values": processed_data.isnull().sum().sum()
        })
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.mlflow_manager.save_and_log_artifact(
            processed_data, f"covid_processed_{timestamp}.csv", "processed_data"
        )
    
    def _create_dummy_data(self) -> pd.DataFrame:
        # 더미 데이터 생성 로직 (필요 시 사용)
        dates = pd.date_range(start='2024-01-01', periods=100)
        return pd.DataFrame({'date': dates, 'new_cases': np.random.randint(0, 1000, 100)})


# --- 스크립트 직접 실행을 위한 부분 ---
if __name__ == "__main__":
    RAW_DATA_PATH = "korea_covid_raw_20250923_025636.csv"
    if os.path.exists(RAW_DATA_PATH):
        preprocessor = CovidDataPreprocessor()
        # 로컬 파일로부터 직접 실행
        processed_data = preprocessor.run(input_csv_path=RAW_DATA_PATH, run_name="manual_preprocessing_run")
        print("\n--- Final Processed Data Head ---")
        print(processed_data.head())
    else:
        print(f"Error: Raw data file not found at '{RAW_DATA_PATH}'")