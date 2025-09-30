# src/data_processing/data_preprocessor.py - 수정된 버전
import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
import mlflow
import mlflow.data
from mlflow.tracking import MlflowClient
from contextlib import contextmanager

from config.settings import ProjectConfig
from utils.mlflow_utils import MLflowManager, create_data_metadata


class CovidDataPreprocessor:
    """수정된 MLflow 통합 데이터 전처리기"""

    def __init__(self, config: Optional[ProjectConfig] = None, tracking_uri: Optional[str] = None):
        self.config = config or ProjectConfig(tracking_uri=tracking_uri)
        if tracking_uri:
            self.config.mlflow.TRACKING_URI = tracking_uri

        # if self.config.mlflow.MLRUNS_DIR:
        #     mlruns_dir = self.config.mlflow.MLRUNS_DIR
        # else :
        #     mlruns_dir = None
        
        mlflow.set_tracking_uri(self.config.mlflow.TRACKING_URI)

        self.mlflow_manager = MLflowManager(
            self.config.mlflow.TRACKING_URI,
            "covid_data_preprocessing"
        )

    @contextmanager
    def safe_mlflow_run(self, run_name: Optional[str] = None):
        """MLflow 런 안전 관리 컨텍스트 매니저"""
        run_started = False
        try:
            # 이미 활성 런이 있는지 확인
            active_run = mlflow.active_run()

            if active_run is None:
                # 활성 런이 없으면 새로 시작
                mlflow.start_run(run_name=run_name)
                run_started = True
                print(f"[mlflow] Started new run: {run_name}")
            else:
                # 활성 런이 있으면 기존 런 사용
                print(f"[mlflow] Using existing active run: {active_run.info.run_id}")

            yield

        except Exception as e:
            print(f"[mlflow] Run error: {e}")
            raise
        finally:
            # 우리가 시작한 런만 종료
            if run_started and mlflow.active_run():
                mlflow.end_run()
                print("[mlflow] Ended run")

    def process_raw_data(self, raw_data: pd.DataFrame, run_name: Optional[str] = None) -> pd.DataFrame:
        """원시 데이터를 처리하여 모델링용 데이터 생성 - 수정된 버전"""

        with self.safe_mlflow_run(run_name):
            print("Starting MLflow-tracked data preprocessing...")

            # 전처리 파라미터 로깅
            self._log_preprocessing_parameters(raw_data)

            # 단계별 전처리 수행
            processed_data = self._execute_preprocessing_pipeline(raw_data)

            # 결과를 MLflow에 로깅
            self._log_processing_results(raw_data, processed_data)

            print(f"Data preprocessing completed! Check MLflow UI: {self.config.mlflow.TRACKING_URI}")
            return processed_data

    def run(
            self,
            run_name: Optional[str] = None,
            from_latest: bool = False,
            input_artifact_uri: Optional[str] = None,
            input_csv_path: Optional[str] = None,
    ):
        """
        Airflow/CLI 표준 진입점 - MLflow 런 충돌 해결 버전
        """
        mlflow.set_experiment("covid_data_preprocessing")

        # 기존 활성 런이 있으면 종료
        if mlflow.active_run():
            print("[preprocess] Ending existing active run...")
            mlflow.end_run()

        with self.safe_mlflow_run(run_name):
            raw_df = None

            # 1) 명시적 MLflow 아티팩트 URI
            if input_artifact_uri:
                try:
                    local_path = mlflow.artifacts.download_artifacts(input_artifact_uri)
                    raw_df = pd.read_csv(local_path)
                    print(f"[preprocess] loaded from artifact: {input_artifact_uri}")
                except Exception as e:
                    print(f"[preprocess] artifact load failed: {e}")

            # 2) 최신 수집 런에서 자동 탐색
            if raw_df is None and from_latest:
                raw_df = self._load_from_latest_collection_safe()

            # 3) 로컬 경로
            if raw_df is None:
                path = input_csv_path or os.environ.get("RAW_CSV_PATH")
                if path and os.path.exists(path):
                    raw_df = pd.read_csv(path)
                    print(f"[preprocess] loaded from local path: {path}")

            # 4) 최종 폴백: 더미 데이터 생성
            if raw_df is None:
                print("[preprocess] No input found, creating dummy data for pipeline continuity")
                raw_df = self._create_dummy_data()
                mlflow.log_param("data_source", "dummy_fallback")

            # 실제 전처리 수행 (이미 런 컨텍스트 안에 있으므로 run_name=None)
            processed = self._execute_preprocessing_pipeline(raw_df)
            self._log_processing_results(raw_df, processed)

            return processed

    def _load_from_latest_collection_safe(self) -> Optional[pd.DataFrame]:
        """MLflow 런 컨텍스트 외부에서 안전하게 데이터 로드"""
        try:
            client = MlflowClient(tracking_uri=self.config.mlflow.TRACKING_URI)

            # 수집 실험 후보들
            candidate_exp_names = [
                "covid_data_collection_v3.1",
                "covid_data_collection_v3",
                "covid_data_collection",
            ]

            exp_ids = []
            for name in candidate_exp_names:
                try:
                    exp = client.get_experiment_by_name(name)
                    if exp and exp.lifecycle_stage == "active":
                        exp_ids.append(exp.experiment_id)
                        print(f"[preprocess] found experiment: {name} (id={exp.experiment_id})")
                except:
                    continue

            if not exp_ids:
                print("[preprocess] No collection experiments found")
                return None

            # 모든 런 검색
            df = mlflow.search_runs(
                experiment_ids=exp_ids,
                order_by=["start_time DESC"],
                max_results=10,  # 줄여서 더 빠르게
            )

            if df.empty:
                print("[preprocess] No runs found")
                return None

            print(f"[preprocess] found {len(df)} collection runs")

            # FINISHED 런만 시도 (RUNNING/FAILED 런은 스킵)
            finished_runs = df[df["status"] == "FINISHED"]

            if finished_runs.empty:
                print("[preprocess] No FINISHED runs found")
                return None

            print(f"[preprocess] trying {len(finished_runs)} FINISHED runs")

            for _, row in finished_runs.iterrows():
                run_id = row["run_id"]
                try:
                    # 여러 아티팩트 경로 시도
                    artifact_paths = ["raw_data", "data", "artifacts"]

                    for artifact_path in artifact_paths:
                        try:
                            # MLflow 런 컨텍스트 없이 아티팩트 다운로드
                            dir_path = mlflow.artifacts.download_artifacts(f"runs:/{run_id}/{artifact_path}")
                            csv_files = [f for f in os.listdir(dir_path) if f.endswith(".csv")]

                            if csv_files:
                                csv_file = os.path.join(dir_path, sorted(csv_files)[-1])
                                raw_df = pd.read_csv(csv_file)
                                print(f"[preprocess] loaded from run {run_id}: {os.path.basename(csv_file)}")
                                return raw_df

                        except Exception as e:
                            print(f"[preprocess] artifact path {artifact_path} failed: {e}")
                            continue

                except Exception as e:
                    print(f"[preprocess] run {run_id} failed: {e}")
                    continue

            print("[preprocess] No usable data found in FINISHED runs")
            return None

        except Exception as e:
            print(f"[preprocess] latest collection load failed: {e}")
            return None

    def _load_from_latest_collection(self) -> Optional[pd.DataFrame]:
        """최신 수집 런에서 데이터 로드 - 개선된 버전"""
        try:
            client = MlflowClient(tracking_uri=self.config.mlflow.TRACKING_URI)

            # 수집 실험 후보들
            candidate_exp_names = [
                "covid_data_collection_v3.1",
                "covid_data_collection_v3",
                "covid_data_collection",
            ]

            exp_ids = []
            for name in candidate_exp_names:
                try:
                    exp = client.get_experiment_by_name(name)
                    if exp and exp.lifecycle_stage == "active":
                        exp_ids.append(exp.experiment_id)
                        print(f"[preprocess] found experiment: {name} (id={exp.experiment_id})")
                except:
                    continue

            if not exp_ids:
                print("[preprocess] No collection experiments found")
                return None

            # 모든 런 검색 (FAILED 포함)
            df = mlflow.search_runs(
                experiment_ids=exp_ids,
                order_by=["start_time DESC"],
                max_results=50,
            )

            if df.empty:
                print("[preprocess] No runs found")
                return None

            print(f"[preprocess] found {len(df)} collection runs")

            # FINISHED 런부터 시도, 없으면 FAILED 런도 시도
            status_priority = ["FINISHED", "RUNNING", "FAILED"]

            for status in status_priority:
                candidates = df[df["status"] == status] if status in df["status"].values else pd.DataFrame()

                if candidates.empty:
                    continue

                print(f"[preprocess] trying {len(candidates)} {status} runs")

                for _, row in candidates.iterrows():
                    run_id = row["run_id"]
                    try:
                        # 아티팩트 경로들 시도
                        artifact_paths = ["raw_data", "data", "artifacts"]

                        for artifact_path in artifact_paths:
                            try:
                                dir_path = mlflow.artifacts.download_artifacts(f"runs:/{run_id}/{artifact_path}")
                                csv_files = [f for f in os.listdir(dir_path) if f.endswith(".csv")]

                                if csv_files:
                                    csv_file = os.path.join(dir_path, sorted(csv_files)[-1])  # 최신 파일
                                    raw_df = pd.read_csv(csv_file)
                                    print(f"[preprocess] loaded from run {run_id}: {os.path.basename(csv_file)}")
                                    return raw_df

                            except Exception:
                                continue

                    except Exception as e:
                        print(f"[preprocess] run {run_id} failed: {e}")
                        continue

            print("[preprocess] No usable data found in any collection run")
            return None

        except Exception as e:
            print(f"[preprocess] latest collection load failed: {e}")
            return None

    def _create_dummy_data(self) -> pd.DataFrame:
        """더미 데이터 생성"""
        print("Creating dummy data for preprocessing...")

        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
        dummy_data = pd.DataFrame({
            'date': dates,
            'country': 'South Korea',
            'new_cases': np.random.randint(100, 1000, len(dates)),
            'total_cases': np.cumsum(np.random.randint(100, 1000, len(dates))),
            'new_deaths': np.random.randint(0, 50, len(dates)),
            'total_deaths': np.cumsum(np.random.randint(0, 50, len(dates))),
            'stringency_index': np.random.uniform(0, 100, len(dates)),
            'reproduction_rate': np.random.uniform(0.5, 2.0, len(dates)),
            'collected_at': datetime.now(),
            'data_source': 'DUMMY',
            'collector': 'DummyDataGenerator'
        })

        return dummy_data

    def _log_preprocessing_parameters(self, raw_data: pd.DataFrame) -> None:
        """전처리 파라미터 로깅"""
        mlflow.log_param("input_rows", len(raw_data))
        mlflow.log_param("input_columns", len(raw_data.columns))
        mlflow.log_param("preprocessing_timestamp", datetime.now().isoformat())
        mlflow.log_param("missing_threshold", 0.5)
        mlflow.log_param("interpolation_method", "linear")
        mlflow.log_param("encoding_method", "one_hot")

    def _execute_preprocessing_pipeline(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """전처리 파이프라인 실행"""

        # 1단계: 날짜 처리 및 시간 특성 추가
        print("Step 1: Processing datetime features...")
        data = self._process_datetime_features(raw_data.copy())
        mlflow.log_metric("step1_output_columns", len(data.columns))

        # 2단계: 결측치가 많은 컬럼 제거
        print("Step 2: Removing high-missing columns...")
        data, removed_cols = self._remove_high_missing_columns(data)
        mlflow.log_metric("step2_removed_columns", len(removed_cols))
        if removed_cols:
            mlflow.log_param("removed_columns", removed_cols[:10])  # 처음 10개만

        # 3단계: 누적 데이터 결측치 보완
        print("Step 3: Filling cumulative columns...")
        data = self._fill_cumulative_columns(data)

        # 4단계: 불필요한 메타데이터 컬럼 제거
        print("Step 4: Removing metadata columns...")
        data = self._remove_metadata_columns(data)

        # 5단계: 범주형 데이터 인코딩
        print("Step 5: Encoding categorical columns...")
        data = self._encode_categorical_columns(data)

        # 6단계: 나머지 결측치 보간
        print("Step 6: Interpolating remaining missing values...")
        data = self._interpolate_missing_values(data)
        mlflow.log_metric("final_missing_values", data.isnull().sum().sum())

        return data

    def _process_datetime_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """날짜 컬럼 처리 및 시간 특성 추가"""
        data['date'] = pd.to_datetime(data['date'])
        data['year'] = data['date'].dt.year
        data['month'] = data['date'].dt.month
        data['day'] = data['date'].dt.day
        data['day_of_week'] = data['date'].dt.dayofweek
        data['is_weekend'] = data['day_of_week'].isin([5, 6]).astype(int)
        return data

    def _remove_high_missing_columns(self, data: pd.DataFrame, threshold: float = 0.5) -> Tuple[pd.DataFrame, list]:
        """결측치 비율이 임계값 이상인 컬럼 제거"""
        missing_ratio = data.isnull().sum() / len(data)
        cols_to_drop = missing_ratio[missing_ratio > threshold].index.tolist()

        # 중요한 컬럼은 보존
        important_cols = ['date', 'country', 'new_cases', 'total_cases', 'new_deaths', 'total_deaths']
        cols_to_drop = [col for col in cols_to_drop if col not in important_cols]

        if cols_to_drop:
            # 제거할 컬럼 로깅
            mlflow.log_text("\n".join(cols_to_drop), "removed_columns.txt")

        return data.drop(columns=cols_to_drop), cols_to_drop

    def _fill_cumulative_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """누적 수치 컬럼의 결측치를 이전/이후 값으로 채우기"""
        cumulative_cols = [col for col in data.columns if 'total' in col.lower()]

        if cumulative_cols:
            data[cumulative_cols] = data[cumulative_cols].bfill().ffill()
            mlflow.log_param("filled_cumulative_columns", cumulative_cols)

        return data

    def _remove_metadata_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """메타데이터 컬럼 제거"""
        metadata_cols = ['collected_at', 'data_source', 'collector']
        cols_to_remove = [col for col in metadata_cols if col in data.columns]

        if cols_to_remove:
            data = data.drop(columns=cols_to_remove)
            mlflow.log_param("removed_metadata_columns", cols_to_remove)

        return data

    def _encode_categorical_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """범주형 데이터를 원-핫 인코딩"""
        categorical_cols = ['country', 'code', 'continent']
        cols_to_encode = [col for col in categorical_cols if col in data.columns]

        if cols_to_encode:
            data = pd.get_dummies(data, columns=cols_to_encode, dtype=int)
            mlflow.log_param("encoded_columns", cols_to_encode)

        return data

    def _interpolate_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """나머지 결측치를 선형 보간으로 처리"""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        data[numeric_cols] = data[numeric_cols].interpolate(method='linear').bfill().ffill()
        return data

    def _log_processing_results(self, raw_data: pd.DataFrame, processed_data: pd.DataFrame) -> None:
        """전처리 결과를 MLflow에 로깅"""


        # 기본 메트릭
        mlflow.log_metric("input_rows", len(raw_data))
        mlflow.log_metric("output_rows", len(processed_data))
        mlflow.log_metric("input_columns", len(raw_data.columns))
        mlflow.log_metric("output_columns", len(processed_data.columns))
        mlflow.log_metric("data_reduction_ratio", 1 - (len(processed_data.columns) / len(raw_data.columns)))

        # 데이터 품질 메트릭
        input_missing = raw_data.isnull().sum().sum()
        output_missing = processed_data.isnull().sum().sum()

        mlflow.log_metric("input_missing_values", input_missing)
        mlflow.log_metric("output_missing_values", output_missing)
        mlflow.log_metric("missing_reduction_ratio", 1 - (output_missing / max(input_missing, 1)))

        # 데이터셋 등록
        dataset = mlflow.data.from_pandas(
            processed_data,
            source="covid_data_preprocessing",
            name="korea_covid_processed_dataset"
        )
        mlflow.log_input(dataset, context="preprocessing")

        # CSV 아티팩트 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"covid_processed_{timestamp}.csv"

        self.mlflow_manager.save_and_log_artifact(
            data=processed_data,
            filename=csv_filename,
            mlruns_dir = self.config.mlflow.PATH,
            artifact_path="processed_data"
        )

        # 전처리 요약 메타데이터
        metadata = self._create_preprocessing_metadata(raw_data, processed_data)
        metadata_filename = f"preprocessing_metadata_{timestamp}.json"
        self.mlflow_manager.log_metadata(metadata, metadata_filename)

        print(f"Processing results logged to MLflow:")
        print(f"  - Processed CSV: processed_data/{csv_filename}")
        print(f"  - Metadata: metadata/{metadata_filename}")
        print(f"  - Dataset registered: korea_covid_processed_dataset")

    def _create_preprocessing_metadata(self, raw_data: pd.DataFrame, processed_data: pd.DataFrame) -> Dict[str, Any]:
        """전처리 메타데이터 생성"""
        return {
            'preprocessing_info': {
                'timestamp': datetime.now().isoformat(),
                'processor': 'MLflowCovidPreprocessor',
                'version': 'v1.1_fixed'
            },
            'transformation_summary': {
                'input_shape': raw_data.shape,
                'output_shape': processed_data.shape,
                'columns_removed': raw_data.shape[1] - processed_data.shape[1],
                'missing_values_eliminated': int(raw_data.isnull().sum().sum() - processed_data.isnull().sum().sum())
            },
            'processing_steps': [
                'datetime_feature_engineering',
                'high_missing_column_removal',
                'cumulative_value_imputation',
                'metadata_column_removal',
                'categorical_encoding',
                'linear_interpolation'
            ],
            'data_quality': {
                'completeness_ratio': float(
                    1 - (processed_data.isnull().sum().sum() / (processed_data.shape[0] * processed_data.shape[1]))),
                'numerical_columns': len(processed_data.select_dtypes(include=[np.number]).columns),
                'categorical_columns': len(processed_data.select_dtypes(include=['bool', 'object']).columns)
            }
        }


# 사용 예시 함수
def run_preprocessing_pipeline():
    """전처리 파이프라인 실행"""

    # 전처리기 생성 및 실행
    preprocessor = CovidDataPreprocessor()
    processed_data = preprocessor.run(
        run_name="preprocessing_pipeline_v1_fixed",
        from_latest=True
    )

    print(f"\nPreprocessing Summary:")
    print(f"  Output: {processed_data.shape}")
    print(f"  Missing values: {processed_data.isnull().sum().sum()}")

    return processed_data


if __name__ == "__main__":
    processed_data = run_preprocessing_pipeline()