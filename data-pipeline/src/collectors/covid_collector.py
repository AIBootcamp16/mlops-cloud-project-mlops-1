# # data-pipeline/src/collectors/covid_collector.py
# """COVID-19 데이터 수집기 - 2022-2023 OWID 실제 데이터"""
# import pandas as pd
# import requests
# import os, tempfile
# from datetime import datetime
# from typing import Optional
# import mlflow
#
# from ..config.settings import ProjectConfig
# from ..utils.mlflow_utils import MLflowManager, create_data_metadata
#
#
# class CovidDataCollector:
#     """OWID 실제 데이터 수집기"""
#
#     def __init__(self, config: Optional[ProjectConfig] = None, tracking_uri: Optional[str] = None):
#         self.config = config or ProjectConfig(tracking_uri=tracking_uri)
#         if tracking_uri:
#             self.config.mlflow.TRACKING_URI = tracking_uri
#
#         self._ensure_local_tracking_if_no_s3_creds()
#         mlflow.set_tracking_uri(self.config.mlflow.TRACKING_URI)
#
#         self.mlflow_manager = MLflowManager(
#             self.config.mlflow.TRACKING_URI,
#             self.config.mlflow.EXPERIMENT_NAME
#         )
#
#         print(f"[collector] Tracking URI: {mlflow.get_tracking_uri()}")
#         print(f"[collector] Train period: {self.config.data.TRAIN_START_DATE} ~ {self.config.data.TRAIN_END_DATE}")
#         print(f"[collector] Target: {self.config.data.TARGET_COUNTRY}")
#
#     def _ensure_local_tracking_if_no_s3_creds(self):
#         """S3 크레덴셜 없으면 로컬 파일 스토어로 폴백"""
#         uri = (self.config.mlflow.TRACKING_URI or "").strip()
#
#         if uri.startswith("http://") or uri.startswith("https://"):
#             return
#
#         if uri.startswith("file:"):
#             if uri.startswith("file:/workspace/"):
#                 self.config.mlflow.TRACKING_URI = "file:/tmp/mlruns"
#             os.makedirs("/tmp/mlruns", exist_ok=True)
#             return
#
#         try:
#             import boto3
#             boto3.client("sts").get_caller_identity()
#             return
#         except Exception:
#             self.config.mlflow.TRACKING_URI = "file:/tmp/mlruns"
#             os.makedirs("/tmp/mlruns", exist_ok=True)
#             print("[collector] S3 credentials not found, using local file store")
#
#     def collect_raw_data(self, run_name: Optional[str] = None) -> pd.DataFrame:
#         """실제 OWID 데이터 수집"""
#
#         with self.mlflow_manager.start_run(run_name):
#             print("\n" + "="*60)
#             print("Starting OWID COVID-19 data collection...")
#             print("="*60)
#
#             try:
#                 self._log_collection_parameters()
#                 korea_data = self._fetch_korea_data_from_owid()
#
#                 if korea_data is None or len(korea_data) == 0:
#                     print("[WARNING] No data collected from OWID, using fallback")
#                     korea_data = self._create_fallback_data()
#
#                 print(f"\n[SUCCESS] Data collected: {korea_data.shape}")
#                 print(f"Date range: {korea_data['date'].min().date()} ~ {korea_data['date'].max().date()}")
#
#                 self._log_to_mlflow(korea_data)
#                 return korea_data
#
#             except Exception as e:
#                 print(f"\n[ERROR] Collection failed: {e}")
#                 mlflow.log_param("collection_error", str(e))
#
#                 # 폴백: 더미 데이터
#                 fallback_data = self._create_fallback_data()
#                 self._log_to_mlflow(fallback_data)
#                 return fallback_data
#
#     def _fetch_korea_data_from_owid(self) -> pd.DataFrame:
#         """OWID에서 한국 데이터 수집"""
#         print("\n[Step 1] Fetching data from OWID...")
#         print(f"URL: {self.config.data.OWID_URL}")
#
#         try:
#             # 연결 테스트
#             print("Testing connection...")
#             response = requests.head(self.config.data.OWID_URL, timeout=30)
#             if response.status_code != 200:
#                 raise ConnectionError(f"Cannot access OWID (status {response.status_code})")
#             print("Connection OK")
#
#             # 전체 데이터 다운로드 (OWID는 상대적으로 작음)
#             print("\nDownloading full dataset...")
#             all_data = pd.read_csv(self.config.data.OWID_URL, low_memory=False)
#             print(f"Downloaded: {len(all_data):,} rows, {len(all_data.columns)} columns")
#
#             # 컬럼명 확인
#             print(f"\nColumn check - 'location' exists: {'location' in all_data.columns}")
#             if 'location' not in all_data.columns:
#                 print(f"Available columns: {list(all_data.columns)[:10]}")
#                 # 'country' 컬럼이 있으면 사용
#                 if 'country' in all_data.columns:
#                     all_data.rename(columns={'country': 'location'}, inplace=True)
#                     print("Renamed 'country' to 'location'")
#
#             # 한국 데이터 필터링
#             print(f"\n[Step 2] Filtering for '{self.config.data.TARGET_COUNTRY}'...")
#             korea_data = all_data[
#                 all_data['location'] == self.config.data.TARGET_COUNTRY
#             ].copy()
#
#             print(f"Korea data found: {len(korea_data):,} rows")
#
#             if len(korea_data) == 0:
#                 # 다른 이름 시도
#                 alternative_names = ['Korea, South', 'Republic of Korea', 'South Korea']
#                 for name in alternative_names:
#                     print(f"Trying alternative name: '{name}'...")
#                     korea_data = all_data[all_data['location'] == name].copy()
#                     if len(korea_data) > 0:
#                         print(f"Found with '{name}': {len(korea_data)} rows")
#                         break
#
#             if len(korea_data) == 0:
#                 print("[WARNING] No Korea data found in OWID dataset")
#                 # 사용 가능한 국가 일부 출력
#                 if 'location' in all_data.columns:
#                     available = all_data['location'].unique()[:20]
#                     print(f"Sample countries: {list(available)}")
#                 return None
#
#             # 날짜 변환 및 전처리
#             korea_data = self._preprocess_data(korea_data)
#
#             # 날짜 범위 필터링
#             korea_data = self._filter_by_date_range(korea_data)
#
#             return korea_data
#
#         except Exception as e:
#             print(f"\n[ERROR] OWID fetch failed: {e}")
#             import traceback
#             traceback.print_exc()
#             return None
#
#     def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
#         """데이터 전처리"""
#         print("\n[Step 3] Preprocessing...")
#
#         # 날짜 변환
#         data['date'] = pd.to_datetime(data['date'], errors='coerce')
#
#         # 날짜 실패한 행 제거
#         invalid_dates = data['date'].isna().sum()
#         if invalid_dates > 0:
#             print(f"Removing {invalid_dates} rows with invalid dates")
#             data = data.dropna(subset=['date'])
#
#         # 날짜 정렬
#         data = data.sort_values('date').reset_index(drop=True)
#
#         # 메타데이터 추가
#         data['collected_at'] = datetime.now()
#         data['data_source'] = 'OWID'
#         data['collector'] = 'CovidCollector_v2'
#
#         return data
#
#     def _filter_by_date_range(self, data: pd.DataFrame) -> pd.DataFrame:
#         """학습 기간으로 데이터 필터링"""
#         print("\n[Step 4] Filtering by date range...")
#
#         train_start = pd.to_datetime(self.config.data.TRAIN_START_DATE)
#         train_end = pd.to_datetime(self.config.data.TRAIN_END_DATE)
#
#         before_filter = len(data)
#
#         data = data[
#             (data['date'] >= train_start) &
#             (data['date'] <= train_end)
#         ].copy()
#
#         after_filter = len(data)
#
#         print(f"Filtered: {before_filter:,} -> {after_filter:,} rows")
#         print(f"Date range: {data['date'].min().date()} ~ {data['date'].max().date()}")
#
#         # MLflow 로깅
#         mlflow.log_param("train_start_date", str(train_start.date()))
#         mlflow.log_param("train_end_date", str(train_end.date()))
#         mlflow.log_metric("rows_before_filter", before_filter)
#         mlflow.log_metric("rows_after_filter", after_filter)
#
#         # 주요 컬럼 통계
#         if 'new_cases' in data.columns:
#             stats = data['new_cases'].describe()
#             print(f"\nnew_cases statistics:")
#             print(f"  Count: {stats['count']:.0f}")
#             print(f"  Mean: {stats['mean']:.2f}")
#             print(f"  Std: {stats['std']:.2f}")
#             print(f"  Min: {stats['min']:.0f}")
#             print(f"  Max: {stats['max']:.0f}")
#
#             mlflow.log_metric("new_cases_mean", float(stats['mean']))
#             mlflow.log_metric("new_cases_std", float(stats['std']))
#             mlflow.log_metric("new_cases_min", float(stats['min']))
#             mlflow.log_metric("new_cases_max", float(stats['max']))
#
#         return data
#
#     def _create_fallback_data(self) -> pd.DataFrame:
#         """폴백 데이터 생성 (설정 기간에 맞춤)"""
#         print("\n[FALLBACK] Creating synthetic data...")
#
#         import numpy as np
#
#         train_start = pd.to_datetime(self.config.data.TRAIN_START_DATE)
#         train_end = pd.to_datetime(self.config.data.TRAIN_END_DATE)
#
#         dates = pd.date_range(start=train_start, end=train_end, freq='D')
#         n = len(dates)
#
#         # 현실적인 패턴 생성
#         np.random.seed(42)
#         base_cases = 5000
#         trend = np.linspace(0, 2000, n)  # 증가 추세
#         seasonal = 1000 * np.sin(np.linspace(0, 4*np.pi, n))  # 계절성
#         noise = np.random.normal(0, 500, n)
#
#         new_cases = np.maximum(0, base_cases + trend + seasonal + noise)
#
#         fallback_data = pd.DataFrame({
#             'date': dates,
#             'location': 'South Korea',
#             'new_cases': new_cases.astype(int),
#             'total_cases': np.cumsum(new_cases).astype(int),
#             'new_deaths': np.maximum(0, (new_cases * 0.01 + np.random.normal(0, 5, n))).astype(int),
#             'total_deaths': None,  # 나중에 cumsum
#             'stringency_index': np.random.uniform(30, 70, n),
#             'reproduction_rate': np.random.uniform(0.8, 1.5, n),
#             'collected_at': datetime.now(),
#             'data_source': 'SYNTHETIC',
#             'collector': 'FallbackGenerator'
#         })
#
#         fallback_data['total_deaths'] = fallback_data['new_deaths'].cumsum()
#
#         mlflow.log_param("data_type", "synthetic_fallback")
#
#         print(f"Generated: {len(fallback_data)} rows")
#         print(f"Date range: {fallback_data['date'].min().date()} ~ {fallback_data['date'].max().date()}")
#
#         return fallback_data
#
#     def _log_collection_parameters(self) -> None:
#         """수집 파라미터 로깅"""
#         mlflow.log_param("data_source_url", self.config.data.OWID_URL)
#         mlflow.log_param("target_country", self.config.data.TARGET_COUNTRY)
#         mlflow.log_param("train_start", self.config.data.TRAIN_START_DATE)
#         mlflow.log_param("train_end", self.config.data.TRAIN_END_DATE)
#         mlflow.log_param("collection_timestamp", datetime.now().isoformat())
#         mlflow.log_param("collector_version", "v2.0_owid_real")
#
#     def _log_to_mlflow(self, data: pd.DataFrame) -> None:
#         """MLflow 로깅"""
#         print("\n[Step 5] Logging to MLflow...")
#
#         try:
#             self.mlflow_manager.log_data_metrics(data)
#
#             key_columns = [
#                 'new_cases', 'total_cases', 'new_deaths', 'total_deaths',
#                 'stringency_index', 'reproduction_rate'
#             ]
#             self.mlflow_manager.log_column_availability(data, key_columns)
#
#             if 'date' in data.columns:
#                 date_range_days = (data['date'].max() - data['date'].min()).days
#                 mlflow.log_metric("date_range_days", date_range_days)
#
#             # 데이터셋 등록
#             self.mlflow_manager.log_dataset(
#                 data=data,
#                 source_url=self.config.data.OWID_URL,
#                 name="korea_covid_2022_2023_dataset",
#                 context="raw_data_collection"
#             )
#
#             # CSV 저장
#             timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#             csv_filename = f"korea_covid_raw_{timestamp}.csv"
#
#             tmp_dir = tempfile.gettempdir()
#             local_csv_path = os.path.join(tmp_dir, csv_filename)
#
#             data.to_csv(local_csv_path, index=False)
#             mlflow.log_artifact(local_csv_path, artifact_path=self.config.mlflow.ARTIFACT_PATH)
#
#             # 메타데이터 저장
#             metadata = create_data_metadata(
#                 data=data,
#                 collection_info={
#                     'source_url': self.config.data.OWID_URL,
#                     'target_country': self.config.data.TARGET_COUNTRY,
#                     'train_start': self.config.data.TRAIN_START_DATE,
#                     'train_end': self.config.data.TRAIN_END_DATE,
#                     'data_source': data['data_source'].iloc[0] if 'data_source' in data.columns else 'UNKNOWN'
#                 }
#             )
#
#             metadata_filename = f"metadata_{timestamp}.json"
#             local_meta_path = os.path.join(tmp_dir, metadata_filename)
#
#             import json
#             with open(local_meta_path, "w", encoding="utf-8") as f:
#                 json.dump(metadata, f, ensure_ascii=False, indent=2)
#
#             mlflow.log_artifact(
#                 local_meta_path,
#                 artifact_path=os.path.join(self.config.mlflow.ARTIFACT_PATH, "metadata")
#             )
#
#             # 정리
#             os.remove(local_csv_path)
#             os.remove(local_meta_path)
#
#             print(f"Logged to MLflow:")
#             print(f"  - CSV: {self.config.mlflow.ARTIFACT_PATH}/{csv_filename}")
#             print(f"  - Metadata: {self.config.mlflow.ARTIFACT_PATH}/metadata/{metadata_filename}")
#             print("="*60 + "\n")
#
#         except Exception as e:
#             print(f"[ERROR] MLflow logging failed: {e}")
#             import traceback
#             traceback.print_exc()
#
#     def run(self, run_name: Optional[str] = None):
#         return self.collect_raw_data(run_name=run_name)
#
#
# def main():
#     """메인 실행 함수"""
#     collector = CovidDataCollector()
#     data = collector.collect_raw_data()
#
#     print(f"\n{'='*60}")
#     print("Collection Summary:")
#     print(f"{'='*60}")
#     print(f"  Rows: {len(data):,}")
#     print(f"  Columns: {len(data.columns)}")
#     print(f"  Date range: {data['date'].min().date()} ~ {data['date'].max().date()}")
#     print(f"  Missing values: {data.isnull().sum().sum():,}")
#     print(f"  Data source: {data['data_source'].iloc[0] if 'data_source' in data.columns else 'Unknown'}")
#     print(f"{'='*60}\n")
#
#
# if __name__ == "__main__":
#     main()

# data-pipeline/src/collectors/covid_collector.py
# -*- coding: utf-8 -*-
"""
OWID COVID-19 데이터 수집기 (탄탄한 네트워크/스키마 일관성/MLflow 로깅 포함)

- 전체 CSV 단일 로드(OWID compact.csv는 충분히 가벼움)
- 네트워크 안정성: User-Agent, timeout, 지수형 백오프 리트라이
- 스키마 표준화: location 기준, 전처리 호환을 위해 country 보조 컬럼 생성
- 원본(raw) 보존 권장: 기본값은 전체 기간 저장, 필요 시 수집 단계 필터 옵션 지원
- MLflow: 파라미터/메트릭/아티팩트(CSV, 메타데이터 JSON) 로깅

필요 패키지: pandas, requests, mlflow
"""

from __future__ import annotations

import os
import json
import time
import random
import tempfile
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List

import pandas as pd
import requests
import mlflow

# 프로젝트 내부 유틸
from ..config.settings import ProjectConfig
from ..utils.mlflow_utils import MLflowManager, create_data_metadata


# ---------------------------
# 설정 데이터클래스 (옵션 확장)
# ---------------------------
@dataclass
class CollectorOptions:
    """수집기 동작 옵션"""
    # True면 수집 단계에서 학습 구간으로 잘라 저장, False면 전체 기간 그대로 저장
    filter_in_collector: bool = False
    # 네트워크 리트라이 최대 시도
    max_retries: int = 3
    # requests 타임아웃 (connect, read)
    timeout_connect: int = 5
    timeout_read: int = 30
    # OWID 한국 대체명 후보
    alt_korea_names: List[str] = None

    def __post_init__(self):
        if self.alt_korea_names is None:
            self.alt_korea_names = ["Republic of Korea", "Korea, South", "South Korea"]


# ---------------------------
# 수집기 본체
# ---------------------------
class CovidDataCollector:
    """OWID 실제 데이터 수집기(탄탄 버전)"""

    def __init__(
        self,
        config: Optional[ProjectConfig] = None,
        tracking_uri: Optional[str] = None,
        options: Optional[CollectorOptions] = None,
    ):
        self.config = config or ProjectConfig(tracking_uri=tracking_uri)
        if tracking_uri:
            self.config.mlflow.TRACKING_URI = tracking_uri

        self.options = options or CollectorOptions()

        self._ensure_local_tracking_if_no_s3_creds()
        mlflow.set_tracking_uri(self.config.mlflow.TRACKING_URI)

        self.mlflow_manager = MLflowManager(
            self.config.mlflow.TRACKING_URI,
            self.config.mlflow.EXPERIMENT_NAME
        )

        print(f"[collector] Tracking URI: {mlflow.get_tracking_uri()}")
        print(f"[collector] Train period: {self.config.data.TRAIN_START_DATE} ~ {self.config.data.TRAIN_END_DATE}")
        print(f"[collector] Target: {self.config.data.TARGET_COUNTRY}")
        print(f"[collector] Filter in collector: {self.options.filter_in_collector}")

    # ---------------------------
    # 내부 유틸
    # ---------------------------
    def _ensure_local_tracking_if_no_s3_creds(self):
        """S3 크레덴셜 없으면 로컬 파일 스토어로 폴백"""
        uri = (self.config.mlflow.TRACKING_URI or "").strip()

        if uri.startswith(("http://", "https://")):
            return

        if uri.startswith("file:"):
            if uri.startswith("file:/workspace/"):
                self.config.mlflow.TRACKING_URI = "file:/tmp/mlruns"
            os.makedirs("/tmp/mlruns", exist_ok=True)
            return

        try:
            import boto3  # noqa
            boto3.client("sts").get_caller_identity()
            return
        except Exception:
            self.config.mlflow.TRACKING_URI = "file:/tmp/mlruns"
            os.makedirs("/tmp/mlruns", exist_ok=True)
            print("[collector] S3 credentials not found, using local file store")

    def _download_df(self, url: str) -> pd.DataFrame:
        """리트라이/타임아웃/UA 적용한 다운로드 → pandas 로드"""
        headers = {"User-Agent": "covid-collector/1.0 (+https://your.domain)"}
        last_exc: Optional[Exception] = None
        for i in range(self.options.max_retries):
            try:
                # HEAD 대신 GET으로 접근성 개선 (일부 CDN에서 HEAD 차단 사례)
                with requests.get(
                    url,
                    headers=headers,
                    stream=True,
                    timeout=(self.options.timeout_connect, self.options.timeout_read),
                ) as r:
                    r.raise_for_status()
                # 스트림 테스트가 OK면 pandas가 직접 URL에서 읽도록 함
                return pd.read_csv(url, low_memory=False)
            except Exception as e:
                last_exc = e
                backoff = 1.5 * (i + 1) + random.random()
                print(f"[download] attempt {i+1}/{self.options.max_retries} failed: {e} → retry in {backoff:.1f}s")
                time.sleep(backoff)
        raise last_exc  # 최종 실패

    def _log_collection_parameters(self) -> None:
        """수집 파라미터 로깅"""
        mlflow.log_param("data_source_url", self.config.data.OWID_URL)
        mlflow.log_param("target_country", self.config.data.TARGET_COUNTRY)
        mlflow.log_param("train_start", self.config.data.TRAIN_START_DATE)
        mlflow.log_param("train_end", self.config.data.TRAIN_END_DATE)
        mlflow.log_param("collection_timestamp", datetime.now().isoformat())
        mlflow.log_param("collector_version", "v3.0_owid_robust")
        mlflow.log_param("filter_in_collector", self.options.filter_in_collector)

    def _preprocess_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """스키마 표준화: location 보장 + 전처리 호환 country 보조 컬럼"""
        if "location" not in df.columns:
            if "country" in df.columns:
                df = df.rename(columns={"country": "location"})
            else:
                raise ValueError(f"'location' column not found. Columns: {list(df.columns)[:20]}")
        # 전처리 단계에서 country 인코딩을 기대한다면 보조 컬럼 생성
        if "country" not in df.columns:
            df["country"] = df["location"]
        return df

    def _filter_korea(self, df: pd.DataFrame) -> pd.DataFrame:
        """한국 데이터 필터링 (대체명 포함)"""
        target = self.config.data.TARGET_COUNTRY
        korea = df[df["location"] == target]
        if korea.empty:
            for alt in self.options.alt_korea_names:
                tmp = df[df["location"] == alt]
                if not tmp.empty:
                    print(f"[filter] Found '{alt}' instead of '{target}': {len(tmp)} rows")
                    korea = tmp
                    break
        if korea.empty:
            # 디버깅을 위해 상위 국가명 일부 보여주기
            sample = df["location"].dropna().unique().tolist()[:20]
            raise ValueError(f"No Korea rows found. Sample locations: {sample}")
        return korea.copy()

    def _postprocess_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """날짜 파싱/정렬 + 메타 컬럼"""
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        invalid = df["date"].isna().sum()
        if invalid:
            print(f"[preprocess] drop rows with invalid date: {invalid}")
            df = df.dropna(subset=["date"])
        df = df.sort_values("date").reset_index(drop=True)

        df["collected_at"] = pd.Timestamp.now()
        df["data_source"] = "OWID"
        df["collector"] = "CovidCollector_v3"
        return df

    def _maybe_filter_by_train_range(self, df: pd.DataFrame) -> pd.DataFrame:
        """옵션에 따라 학습 기간으로 필터링 (기본 False: 원본 보존)"""
        if not self.options.filter_in_collector:
            return df
        start = pd.to_datetime(self.config.data.TRAIN_START_DATE)
        end = pd.to_datetime(self.config.data.TRAIN_END_DATE)
        before = len(df)
        df = df[(df["date"] >= start) & (df["date"] <= end)].copy()
        after = len(df)
        print(f"[range] filtered by train period: {before:,} -> {after:,} rows")
        mlflow.log_param("collector_date_filter_applied", True)
        mlflow.log_metric("rows_before_filter", before)
        mlflow.log_metric("rows_after_filter", after)
        return df

    def _log_basic_stats(self, df: pd.DataFrame) -> None:
        """핵심 통계 로깅"""
        if "new_cases" in df.columns:
            stats = df["new_cases"].describe()
            mlflow.log_metric("new_cases_count", float(stats.get("count", 0)))
            mlflow.log_metric("new_cases_mean", float(stats.get("mean", 0)))
            mlflow.log_metric("new_cases_std", float(stats.get("std", 0)))
            mlflow.log_metric("new_cases_min", float(stats.get("min", 0)))
            mlflow.log_metric("new_cases_max", float(stats.get("max", 0)))

        if "date" in df.columns and len(df) > 0:
            mlflow.log_metric("date_range_days", int((df["date"].max() - df["date"].min()).days))

    def _log_artifacts(self, df: pd.DataFrame) -> None:
        """CSV/메타데이터를 MLflow 아티팩트로 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"korea_covid_raw_{timestamp}.csv"
        meta_filename = f"metadata_{timestamp}.json"

        tmp_dir = tempfile.gettempdir()
        csv_path = os.path.join(tmp_dir, csv_filename)
        meta_path = os.path.join(tmp_dir, meta_filename)

        # CSV 저장
        df.to_csv(csv_path, index=False)
        mlflow.log_artifact(csv_path, artifact_path=self.config.mlflow.ARTIFACT_PATH)

        # 메타데이터
        metadata = create_data_metadata(
            data=df,
            collection_info={
                "source_url": self.config.data.OWID_URL,
                "target_country": self.config.data.TARGET_COUNTRY,
                "train_start": self.config.data.TRAIN_START_DATE,
                "train_end": self.config.data.TRAIN_END_DATE,
                "data_source": df.get("data_source", pd.Series(["UNKNOWN"])).iloc[0],
            },
        )
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        mlflow.log_artifact(meta_path, artifact_path=f"{self.config.mlflow.ARTIFACT_PATH}/metadata")

        # 정리
        try:
            os.remove(csv_path)
            os.remove(meta_path)
        except Exception:
            pass

        print("[mlflow] logged artifacts:")
        print(f"  - {self.config.mlflow.ARTIFACT_PATH}/{csv_filename}")
        print(f"  - {self.config.mlflow.ARTIFACT_PATH}/metadata/{meta_filename}")

    # ---------------------------
    # 공개 API
    # ---------------------------
    def collect_raw_data(self, run_name: Optional[str] = None) -> pd.DataFrame:
        """OWID → 한국 데이터 수집(원본 보존 권장), MLflow 로깅 포함"""
        with self.mlflow_manager.start_run(run_name):
            print("\n" + "=" * 60)
            print("Starting OWID COVID-19 data collection...")
            print("=" * 60)

            try:
                self._log_collection_parameters()

                print(f"[step] download: {self.config.data.OWID_URL}")
                df_all = self._download_df(self.config.data.OWID_URL)
                print(f"[ok] downloaded: {len(df_all):,} rows, {len(df_all.columns)} columns")

                df_all = self._preprocess_schema(df_all)
                df_kr = self._filter_korea(df_all)
                print(f"[ok] korea rows: {len(df_kr):,}")

                df_kr = self._postprocess_frame(df_kr)
                df_kr = self._maybe_filter_by_train_range(df_kr)

                # 간단 콘솔 요약
                if len(df_kr):
                    print(f"[range] {df_kr['date'].min().date()} ~ {df_kr['date'].max().date()}")

                # MLflow 로깅
                self.mlflow_manager.log_data_metrics(df_kr)
                self.mlflow_manager.log_column_availability(
                    df_kr,
                    [
                        "new_cases", "total_cases", "new_deaths", "total_deaths",
                        "stringency_index", "reproduction_rate",
                    ],
                )
                self._log_basic_stats(df_kr)

                self.mlflow_manager.log_dataset(
                    data=df_kr,
                    source_url=self.config.data.OWID_URL,
                    name="korea_covid_dataset_raw",
                    context="raw_data_collection",
                )

                self._log_artifacts(df_kr)

                print("=" * 60 + "\n")
                return df_kr

            except Exception as e:
                print(f"\n[ERROR] collection failed: {e}")
                mlflow.log_param("collection_error", str(e))
                raise

    def run(self, run_name: Optional[str] = None) -> pd.DataFrame:
        return self.collect_raw_data(run_name=run_name)


# ---------------------------
# 스크립트 실행 엔트리포인트
# ---------------------------
def main():
    collector = CovidDataCollector(
        # ProjectConfig() 생략 시 내부 기본값 사용
        options=CollectorOptions(
            filter_in_collector=False  # 원본(raw) 보존 권장. 필요시 True로.
        )
    )
    df = collector.collect_raw_data(run_name="collect_owid_korea")

    print("\n" + "=" * 60)
    print("Collection Summary")
    print("=" * 60)
    print(f"Rows:        {len(df):,}")
    print(f"Columns:     {len(df.columns)}")
    print(f"Date range:  {df['date'].min().date()} ~ {df['date'].max().date()}" if len(df) else "N/A")
    print(f"Missing sum: {int(df.isnull().sum().sum()) if len(df) else 0:,}")
    print(f"Source:      {df['data_source'].iloc[0] if len(df) and 'data_source' in df.columns else 'Unknown'}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
