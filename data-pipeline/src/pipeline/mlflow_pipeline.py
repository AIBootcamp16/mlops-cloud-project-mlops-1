import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
import mlflow
import mlflow.data
from mlflow.tracking import MlflowClient
import pandas as pd
from typing import Optional, Dict, Any
from datetime import datetime
import os

from ..collectors.covid_collector import CovidDataCollector
from ..data_processing.data_preprocessor import CovidDataPreprocessor
from ..config.settings import ProjectConfig


class MLflowCovidPipeline:
    """MLflow 기반 COVID 데이터 전체 파이프라인"""
    
    def __init__(self, config: Optional[ProjectConfig] = None, tracking_uri: Optional[str] = None):
        self.config = config or ProjectConfig(tracking_uri=tracking_uri)

        if tracking_uri:
            self.config.mlflow.TRACKING_URI = tracking_uri

        # ★ 추가: collector와 동일한 폴백 가드
        self._ensure_local_tracking_if_no_s3_creds()

        # ★ 변경: mlflow 글로벌 세팅 먼저
        mlflow.set_tracking_uri(self.config.mlflow.TRACKING_URI)

        # ★ 변경: 하위 컴포넌트는 반드시 self.config로 "한 번만" 생성
        self.collector = CovidDataCollector(config=self.config)
        self.preprocessor = CovidDataPreprocessor(config=self.config)

    def _ensure_local_tracking_if_no_s3_creds(self):
        uri = (self.config.mlflow.TRACKING_URI or "").strip()

        if uri.startswith("http://") or uri.startswith("https://"):
            return

        if uri.startswith("file:"):
            if uri.startswith("file:/workspace/"):
                self.config.mlflow.TRACKING_URI = "file:/tmp/mlruns"
            os.makedirs("/tmp/mlruns", exist_ok=True)
            return

        try:
            import boto3
            boto3.client("sts").get_caller_identity()
            return
        except Exception:
            self.config.mlflow.TRACKING_URI = "file:/tmp/mlruns"
            os.makedirs("/tmp/mlruns", exist_ok=True)

    # ★ 추가: collect.py가 먼저 찾는 엔트리포인트
    def run_collection(self, run_name: Optional[str] = None):
        return self.collector.collect_raw_data(run_name=run_name or "collection")

    def run_full_pipeline(self, collection_run_name: Optional[str] = None, 
                         preprocessing_run_name: Optional[str] = None) -> pd.DataFrame:
        """전체 파이프라인 실행: 데이터 수집 → 전처리"""
        print("Starting full MLflow COVID data pipeline...")
        print("=" * 60)

        # 1단계: 데이터 수집
        print("\n[STEP 1] Data Collection")
        raw_data = self.collector.collect_raw_data(run_name=collection_run_name)
        collection_run_id = mlflow.active_run().info.run_id if mlflow.active_run() else None

        # 2단계: 데이터 전처리
        print("\n[STEP 2] Data Preprocessing")
        processed_data = self.preprocessor.process_raw_data(
            raw_data, 
            run_name=preprocessing_run_name
        )

        print("\n" + "=" * 60)
        print("Pipeline completed successfully!")
        print(f"MLflow UI: {self.config.mlflow.TRACKING_URI}")
        
        return processed_data

    def run_preprocessing_from_artifact(self, collection_run_id: str, 
                                        artifact_path: str = None,
                                        preprocessing_run_name: Optional[str] = None) -> pd.DataFrame:
         # ★ 변경: 기본 아티팩트 경로는 설정값과 일치시킴
        artifact_path = artifact_path or self.config.mlflow.ARTIFACT_PATH

        print(f"Loading raw data from MLflow artifact... Run ID: {collection_run_id}")
        artifact_uri = f"runs:/{collection_run_id}/{artifact_path}"
        local_path = mlflow.artifacts.download_artifacts(artifact_uri)

        csv_files = [f for f in os.listdir(local_path) if f.endswith('.csv')]
        if not csv_files:
            raise ValueError(f"No CSV files found in artifact path: {local_path}")

        csv_file = csv_files[0]
        raw_data = pd.read_csv(os.path.join(local_path, csv_file))

        print(f"Successfully loaded raw data: {raw_data.shape} from {csv_file}")

        processed_data = self.preprocessor.process_raw_data(
            raw_data,
            run_name=preprocessing_run_name
        )

        if mlflow.active_run():
            mlflow.log_param("source_run_id", collection_run_id)
            mlflow.log_param("source_artifact_path", artifact_path)
            mlflow.log_param("source_csv_file", csv_file)

        return processed_data

    def get_latest_collection_run(self, experiment_name: Optional[str] = None) -> Optional[str]:
        """가장 최근의 데이터 수집 런 ID 가져오기"""
        # ★ 변경: 기본값은 설정값 사용
        experiment_name = experiment_name or self.config.mlflow.EXPERIMENT_NAME

        experiment = mlflow.get_experiment_by_name(experiment_name)
        if not experiment:
            print(f"Experiment '{experiment_name}' not found")
            return None

        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=1
        )
        if runs.empty:
            print(f"No runs found in experiment '{experiment_name}'")
            return None

        latest_run_id = runs.iloc[0]['run_id']
        start_time = runs.iloc[0]['start_time']
        print(f"Latest collection run found: {latest_run_id} at {start_time}")
        return latest_run_id

    def list_available_artifacts(self, run_id: str) -> Dict[str, Any]:
        """특정 런의 아티팩트 목록 조회"""
        client = mlflow.tracking.MlflowClient()
        artifacts = client.list_artifacts(run_id)

        artifact_info = {}
        for artifact in artifacts:
            if artifact.is_dir:
                sub_artifacts = client.list_artifacts(run_id, artifact.path)
                artifact_info[artifact.path] = [sub.path for sub in sub_artifacts]
            else:
                artifact_info[artifact.path] = "file"

        return artifact_info

    # 🔑 권장 리팩터링: 클래스 메서드 추가
    def run_preprocessing_from_latest_collection(self, run_name: Optional[str] = None):
        """가장 최근 수집 데이터로 전처리 실행 - 시그니처 수정"""
        latest_run_id = self.get_latest_collection_run()
        if not latest_run_id:
            print("No collection run found. Running full pipeline instead...")
            return self.run_full_pipeline(
                collection_run_name="collection_v1",
                preprocessing_run_name=run_name or "preprocessing_v1"
            )

        artifacts = self.list_available_artifacts(latest_run_id)
        print(f"Available artifacts: {artifacts}")

        processed_data = self.run_preprocessing_from_artifact(
            collection_run_id=latest_run_id,
            artifact_path="raw_data",
            preprocessing_run_name=run_name or "preprocessing_from_artifact_v1"
        )
        return processed_data

    def load_latest_processed(self) -> Optional[pd.DataFrame]:
        """최신 전처리 데이터 로드"""
        try:
            client = MlflowClient(tracking_uri=self.config.mlflow.TRACKING_URI)

            # 전처리 실험에서 최신 성공 런 찾기
            preprocessing_exp = client.get_experiment_by_name("covid_data_preprocessing")
            if not preprocessing_exp:
                return None

            runs_df = mlflow.search_runs(
                [preprocessing_exp.experiment_id],
                filter_string="attribute.status = 'FINISHED'",
                order_by=["start_time DESC"],
                max_results=5
            )

            if runs_df.empty:
                return None

            # 최신 런에서 processed_data 아티팩트 찾기
            for _, run in runs_df.iterrows():
                run_id = run["run_id"]
                try:
                    # processed_data 폴더에서 CSV 찾기
                    dir_path = mlflow.artifacts.download_artifacts(f"runs:/{run_id}/processed_data")
                    csv_files = [f for f in os.listdir(dir_path) if f.endswith(".csv")]

                    if csv_files:
                        csv_file = os.path.join(dir_path, sorted(csv_files)[-1])
                        processed_data = pd.read_csv(csv_file)
                        print(f"[pipeline] loaded processed data: {processed_data.shape}")
                        return processed_data

                except Exception as e:
                    print(f"[pipeline] run {run_id} artifact load failed: {e}")
                    continue

            return None

        except Exception as e:
            print(f"[pipeline] load_latest_processed failed: {e}")
            return None

    def run(
            self,
            run_name: Optional[str] = None,
            from_latest: bool = False,
            input_artifact_uri: Optional[str] = None,
    ):
        """
        Airflow CLI에서 호출하는 표준 진입점.
        - from_latest=True면 '수집(run) 최신 아티팩트'를 찾아 전처리
        - 아니면 input_artifact_uri(로컬 경로/MLflow URI)로부터 전처리
        """
        # 실험 고정 (프로젝트 설정에 따라 조정)
        mlflow.set_experiment("covid_data_preprocessing")

        with mlflow.start_run(run_name=run_name):
            if from_latest:
                # 프로젝트 내 존재하는 메서드명에 맞춰 연결하세요.
                # 예: self.run_from_latest_collection(...) 또는 self.preprocess_from_latest_collection(...)
                return self.run_from_latest_collection(run_name=run_name)

            # 직접 입력 경로로 전처리 (없으면 예외)
            if not input_artifact_uri:
                raise ValueError("input_artifact_uri가 필요합니다 (from_latest=False).")

            return self.preprocess_and_log(input_artifact_uri, run_name=run_name)


# -------------------------
# 모듈 레벨 wrapper 함수들
# -------------------------
def run_collection_and_preprocessing(tracking_uri: Optional[str] = None):
    pipeline = MLflowCovidPipeline(tracking_uri=tracking_uri)
    return pipeline.run_full_pipeline(
        collection_run_name="collection_v1",
        preprocessing_run_name="preprocessing_v1"
    )

def run_preprocessing_from_latest_collection(tracking_uri: Optional[str] = None):
    pipeline = MLflowCovidPipeline(tracking_uri=tracking_uri)
    return pipeline.run_preprocessing_from_latest_collection()

def run_preprocessing_from_specific_run(run_id: str, tracking_uri: Optional[str] = None):
    pipeline = MLflowCovidPipeline(tracking_uri=tracking_uri)
    return pipeline.run_preprocessing_from_artifact(
        collection_run_id=run_id,
        preprocessing_run_name=f"preprocessing_from_{run_id[:8]}"
    )


if __name__ == "__main__":
    processed_data = run_collection_and_preprocessing()
    print(f"\nFinal processed data shape: {processed_data.shape}")
    print("Pipeline completed!")
