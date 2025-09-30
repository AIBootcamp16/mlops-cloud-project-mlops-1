"""COVID-19 데이터 수집기 - 수정된 버전"""
import pandas as pd
import requests
import os, tempfile
import boto3
from datetime import datetime, date
from botocore.exceptions import NoCredentialsError, PartialCredentialsError, ClientError
from typing import Optional, Tuple
import mlflow

from config.settings import ProjectConfig
from utils.mlflow_utils import MLflowManager, create_data_metadata


class CovidDataCollector:
    """수정된 COVID-19 데이터 수집기"""

    def __init__(self, config: Optional[ProjectConfig] = None, tracking_uri: Optional[str] = None):
        self.config = config or ProjectConfig(tracking_uri=tracking_uri)
        if tracking_uri:
            self.config.mlflow.TRACKING_URI = tracking_uri

        # S3 자격증명 확인 및 폴백
        self._ensure_local_tracking_if_no_s3_creds()

        mlflow.set_tracking_uri(self.config.mlflow.TRACKING_URI)

        self.mlflow_manager = MLflowManager(
            self.config.mlflow.TRACKING_URI,
            self.config.mlflow.EXPERIMENT_NAME
        )

        print(f"[collector] tracking_uri = {mlflow.get_tracking_uri()}")

    def _ensure_local_tracking_if_no_s3_creds(self):
        """S3 크리덴셜 없으면 로컬 파일 스토어로 폴백"""
        uri = (self.config.mlflow.TRACKING_URI or "").strip()

        # 서버 프록시 모드(HTTP/HTTPS)면 폴백 금지
        if uri.startswith("http://") or uri.startswith("https://"):
            return

        if uri.startswith("file:"):
            if uri.startswith("file:/workspace/"):
                self.config.mlflow.TRACKING_URI = "file:/tmp/mlruns"
            os.makedirs("/tmp/mlruns", exist_ok=True)
            return

        # S3 직접 접근 시도
        try:
            import boto3
            boto3.client("sts").get_caller_identity()
            return
        except Exception:
            self.config.mlflow.TRACKING_URI = "file:/tmp/mlruns"
            os.makedirs("/tmp/mlruns", exist_ok=True)
            print("[collector] S3 credentials not found, using local file store")

    def collect_raw_data(self, run_name: Optional[str] = None) -> pd.DataFrame:
        """원시 데이터 수집 및 MLflow 로깅"""

        with self.mlflow_manager.start_run(run_name):
            print("Starting MLflow-tracked COVID data collection...")

            try:
                # 수집 파라미터 로깅
                self._log_collection_parameters()

                # 데이터 수집 실행
                korea_data = self._fetch_korea_data()

                # 데이터 검증
                if korea_data is None or len(korea_data) == 0:
                    raise ValueError("No data collected")

                print(f"Data collected successfully: {korea_data.shape}")

                # MLflow 로깅
                self._log_to_mlflow(korea_data)

                print(f"Data collection completed! Check MLflow UI: {self.config.mlflow.TRACKING_URI}")
                return korea_data

            except Exception as e:
                print(f"Collection failed: {e}")
                mlflow.log_param("collection_error", str(e))

                # 오류 발생시 더미 데이터라도 반환 (파이프라인 중단 방지)
                dummy_data = self._create_dummy_data()
                self._log_to_mlflow(dummy_data)
                return dummy_data

    def _create_dummy_data(self) -> pd.DataFrame:
        """수집 실패시 더미 데이터 생성"""
        print("Creating dummy data for pipeline continuity...")

        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
        dummy_data = pd.DataFrame({
            'date': dates,
            'country': 'South Korea',
            'new_cases': range(len(dates)),
            'total_cases': range(1000, 1000 + len(dates)),
            'new_deaths': [1] * len(dates),
            'total_deaths': range(100, 100 + len(dates)),
            'collected_at': datetime.now(),
            'data_source': 'DUMMY',
            'collector': 'MLflowCovidCollector'
        })

        mlflow.log_param("data_type", "dummy")
        return dummy_data

    def _log_collection_parameters(self) -> None:
        """수집 파라미터를 MLflow에 로깅"""
        mlflow.log_param("data_source_url", self.config.data.OWID_URL)
        mlflow.log_param("target_country", self.config.data.TARGET_COUNTRY)
        mlflow.log_param("chunk_size", self.config.data.CHUNK_SIZE)
        mlflow.log_param("filter_future_dates", self.config.data.FILTER_FUTURE_DATES)
        mlflow.log_param("collection_timestamp", datetime.now().isoformat())
        mlflow.log_param("collector_version", "v1.1_fixed")

    def _fetch_korea_data(self) -> pd.DataFrame:
        """한국 COVID 데이터 수집"""
        print("Fetching Korea data from OWID...")

        korea_chunks = []
        chunks_processed = 0

        try:
            # 데이터 소스 연결 테스트
            response = requests.head(self.config.data.OWID_URL, timeout=30)
            if response.status_code != 200:
                raise ConnectionError(f"Cannot access data source: {response.status_code}")

            # 청크 단위로 데이터 읽기
            for chunk in pd.read_csv(self.config.data.OWID_URL, chunksize=self.config.data.CHUNK_SIZE):
                chunks_processed += 1
                print(f"Processing chunk {chunks_processed}...")

                # 한국 데이터 필터링
                korea_in_chunk = chunk[
                    chunk[self.config.data.COUNTRY_COLUMN] == self.config.data.TARGET_COUNTRY
                ].copy()

                if len(korea_in_chunk) > 0:
                    korea_chunks.append(korea_in_chunk)
                    print(f"Found Korea data in chunk {chunks_processed}: {len(korea_in_chunk)} rows")

                # 최대 10개 청크만 처리 (메모리 절약)
                # if chunks_processed >= 10:
                #     break

            # 청크 병합
            if korea_chunks:
                korea_raw = pd.concat(korea_chunks, ignore_index=True)
                print(f"Total Korea data collected: {len(korea_raw)} rows")
            else:
                raise ValueError("No Korea data found in the dataset")

            # 데이터 전처리
            korea_raw = self._preprocess_data(korea_raw)

            # 수집 통계 로깅
            mlflow.log_metric("chunks_processed", chunks_processed)
            mlflow.log_metric("chunks_with_korea_data", len(korea_chunks))

            return korea_raw

        except Exception as e:
            print(f"Data fetching error: {e}")
            raise

    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """데이터 전처리"""
        print("Preprocessing collected data...")

        # 날짜 변환
        data['date'] = pd.to_datetime(data['date'])

        # 미래 날짜 필터링
        if self.config.data.FILTER_FUTURE_DATES:
            before_filter = len(data)
            today = date.today()
            data = data[data['date'].dt.date <= today].copy()
            after_filter = len(data)

            # 필터링 결과 로깅
            mlflow.log_metric("rows_before_date_filter", before_filter)
            mlflow.log_metric("rows_after_date_filter", after_filter)
            mlflow.log_metric("future_dates_removed", before_filter - after_filter)

            print(f"Filtered future dates: {before_filter} → {after_filter} rows")

        # 메타데이터 컬럼 추가
        data['collected_at'] = datetime.now()
        data['data_source'] = 'OWID'
        data['collector'] = 'MLflowCovidCollector'

        return data

    def _log_to_mlflow(self, data: pd.DataFrame) -> None:
        """데이터를 MLflow에 로깅"""
        print("Logging data to MLflow...")

        try:
            # 기본 데이터 메트릭 로깅
            self.mlflow_manager.log_data_metrics(data)

            # 주요 컬럼 가용성 로깅
            key_columns = [
                'new_cases', 'total_cases', 'new_deaths', 'total_deaths',
                'people_vaccinated_per_hundred', 'people_fully_vaccinated_per_hundred',
                'stringency_index', 'reproduction_rate'
            ]
            self.mlflow_manager.log_column_availability(data, key_columns)

            # 날짜 범위 정보
            if 'date' in data.columns:
                date_range_days = (data['date'].max() - data['date'].min()).days
                mlflow.log_metric("date_range_days", date_range_days)
                mlflow.log_param("date_start", data['date'].min().strftime('%Y-%m-%d'))
                mlflow.log_param("date_end", data['date'].max().strftime('%Y-%m-%d'))

            # 데이터셋 등록
            self.mlflow_manager.log_dataset(
                data=data,
                source_url=self.config.data.OWID_URL,
                name="korea_covid_raw_dataset",
                context="raw_data_collection"
            )

            # CSV 아티팩트 저장
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_filename = f"korea_covid_raw_{timestamp}.csv"

            # 안전한 임시 디렉토리 사용
            tmp_dir = tempfile.gettempdir()
            local_csv_path = os.path.join(tmp_dir, csv_filename)

            # 저장
            data.to_csv(local_csv_path, index=False)

            # MLflow에 아티팩트로 업로드
            artifact_path = self.config.mlflow.ARTIFACT_PATH
            mlflow.log_artifact(local_csv_path, artifact_path=artifact_path)

            # 메타데이터 생성 및 저장
            metadata = create_data_metadata(
                data=data,
                collection_info={
                    'source_url': self.config.data.OWID_URL,
                    'target_country': self.config.data.TARGET_COUNTRY,
                    'chunk_size': self.config.data.CHUNK_SIZE
                }
            )

            metadata_filename = f"metadata_{timestamp}.json"
            local_meta_path = os.path.join(tmp_dir, metadata_filename)

            # 저장
            import json
            with open(local_meta_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

            # 로깅
            mlflow.log_artifact(local_meta_path, artifact_path=os.path.join(artifact_path, "metadata"))

            # 임시 파일 정리
            os.remove(local_csv_path)
            os.remove(local_meta_path)

            print("Data logged to MLflow successfully:")
            print(f"  - CSV artifact: {artifact_path}/{csv_filename}")
            print(f"  - Metadata: {artifact_path}/metadata/{metadata_filename}")

        except Exception as e:
            print(f"MLflow logging error: {e}")
            # 로깅 실패해도 데이터는 반환

    def run(self, run_name: Optional[str] = None):
        return self.collect_raw_data(run_name=run_name)


def main():
    """메인 실행 함수"""
    collector = CovidDataCollector()
    data = collector.collect_raw_data()

    print(f"\nCollection Summary:")
    print(f"  Rows: {len(data):,}")
    print(f"  Columns: {len(data.columns)}")
    print(f"  Date range: {data['date'].min()} ~ {data['date'].max()}")
    print(f"  Missing values: {data.isnull().sum().sum():,}")


if __name__ == "__main__":
    main()