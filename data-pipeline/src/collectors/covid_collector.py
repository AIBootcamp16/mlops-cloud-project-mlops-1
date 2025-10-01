# data-pipeline/src/collectors/covid_collector.py
# -*- coding: utf-8 -*-
"""
OWID COVID-19 데이터 수집기 (탄탄한 네트워크/스키마 일관성/MLflow 로깅 포함)
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

from ..config.settings import ProjectConfig
from ..utils.mlflow_utils import MLflowManager, create_data_metadata


@dataclass
class CollectorOptions:
    """수집기 동작 옵션"""
    filter_in_collector: bool = False
    max_retries: int = 3
    timeout_connect: int = 5
    timeout_read: int = 30
    alt_korea_names: List[str] = None

    def __post_init__(self):
        if self.alt_korea_names is None:
            self.alt_korea_names = ["Republic of Korea", "Korea, South", "South Korea"]


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
        print(
            f"[collector] Train period: {self.config.data.TRAIN_START_DATE} ~ {self.config.data.TRAIN_END_DATE or 'auto'}")
        print(f"[collector] Target: {self.config.data.TARGET_COUNTRY}")
        print(f"[collector] Filter in collector: {self.options.filter_in_collector}")

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
            import boto3
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
                with requests.get(
                        url,
                        headers=headers,
                        stream=True,
                        timeout=(self.options.timeout_connect, self.options.timeout_read),
                ) as r:
                    r.raise_for_status()
                return pd.read_csv(url, low_memory=False)
            except Exception as e:
                last_exc = e
                backoff = 1.5 * (i + 1) + random.random()
                print(f"[download] attempt {i + 1}/{self.options.max_retries} failed: {e} → retry in {backoff:.1f}s")
                time.sleep(backoff)
        raise last_exc

    def _log_collection_parameters(self) -> None:
        """수집 파라미터 로깅"""
        mlflow.log_param("data_source_url", self.config.data.OWID_URL)
        mlflow.log_param("target_country", self.config.data.TARGET_COUNTRY)
        mlflow.log_param("train_start", self.config.data.TRAIN_START_DATE)

        # TRAIN_END_DATE 처리
        train_end = self.config.data.TRAIN_END_DATE
        if not train_end or not train_end.strip():
            train_end = (pd.Timestamp.now() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        mlflow.log_param("train_end", train_end)

        mlflow.log_param("collection_timestamp", datetime.now().isoformat())
        mlflow.log_param("collector_version", "v3.1_future_filter")
        mlflow.log_param("filter_in_collector", self.options.filter_in_collector)

    def _preprocess_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """스키마 표준화: location 보장 + 전처리 호환 country 보조 컬럼"""
        if "location" not in df.columns:
            if "country" in df.columns:
                df = df.rename(columns={"country": "location"})
            else:
                raise ValueError(f"'location' column not found. Columns: {list(df.columns)[:20]}")
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
            sample = df["location"].dropna().unique().tolist()[:20]
            raise ValueError(f"No Korea rows found. Sample locations: {sample}")
        return korea.copy()

    def _postprocess_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """날짜 파싱/정렬 + 메타 컬럼 + 미래 날짜 제거"""
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        invalid = df["date"].isna().sum()
        if invalid:
            print(f"[preprocess] drop rows with invalid date: {invalid}")
            df = df.dropna(subset=["date"])

        # ✅ 미래 날짜 필터링
        today = pd.Timestamp.now().normalize()
        future_mask = df["date"] > today
        future_count = future_mask.sum()

        if future_count > 0:
            print(f"[preprocess] ⚠️  Removing {future_count} future date rows (after {today.date()})")
            df = df[~future_mask].copy()

        df = df.sort_values("date").reset_index(drop=True)

        df["collected_at"] = pd.Timestamp.now()
        df["data_source"] = "OWID"
        df["collector"] = "CovidCollector_v3.1"

        print(f"[preprocess] Final date range: {df['date'].min().date()} ~ {df['date'].max().date()}")

        return df

    def _maybe_filter_by_train_range(self, df: pd.DataFrame) -> pd.DataFrame:
        """✅ 수정: 미래 날짜 제거 + 선택적 날짜 범위 필터링"""

        # ✅ 1단계: 항상 미래 날짜는 제거
        today = pd.Timestamp.now().normalize()
        before_future_filter = len(df)
        df = df[df["date"] <= today].copy()
        after_future_filter = len(df)

        future_removed = before_future_filter - after_future_filter
        if future_removed > 0:
            print(f"[range] Removed {future_removed} future date rows")
            mlflow.log_metric("future_rows_removed", future_removed)

        # ✅ 2단계: filter_in_collector=False면 전체 데이터 보존
        if not self.options.filter_in_collector:
            print(f"[range] Keeping all historical data: {len(df):,} rows")
            print(f"[range] Date range: {df['date'].min().date()} ~ {df['date'].max().date()}")

            # 메트릭만 로깅
            start = pd.to_datetime(self.config.data.TRAIN_START_DATE)
            mlflow.log_param("data_start_date", str(df['date'].min().date()))
            mlflow.log_param("data_end_date", str(df['date'].max().date()))
            mlflow.log_param("config_train_start", str(start.date()))
            mlflow.log_metric("total_rows", len(df))

            return df

        # ✅ 3단계: 필터링 활성화된 경우에만 날짜 범위 필터링
        start = pd.to_datetime(self.config.data.TRAIN_START_DATE)

        # TRAIN_END_DATE 처리
        if self.config.data.TRAIN_END_DATE and self.config.data.TRAIN_END_DATE.strip():
            end = pd.to_datetime(self.config.data.TRAIN_END_DATE)
        else:
            # 빈 값이면 어제 날짜
            end = today - pd.Timedelta(days=1)

        before = len(df)
        df = df[(df["date"] >= start) & (df["date"] <= end)].copy()
        after = len(df)

        print(f"[range] Filtered by train period: {before:,} -> {after:,} rows")
        print(f"[range] Train range: {start.date()} ~ {end.date()}")
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

        tmp_dir = tempfile.gettempdir()
        csv_path = os.path.join(tmp_dir, csv_filename)

        # CSV 저장
        df.to_csv(csv_path, index=False)

        # ✅ 고정된 경로로 저장 (raw_data 대신 data 사용)
        mlflow.log_artifact(csv_path, artifact_path="data")  # 변경

        # 메타데이터도 동일하게
        metadata = create_data_metadata(...)
        meta_path = os.path.join(tmp_dir, f"metadata_{timestamp}.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        mlflow.log_artifact(meta_path, artifact_path="data/metadata")  # 변경

        # 정리
        try:
            os.remove(csv_path)
            os.remove(meta_path)
        except Exception:
            pass

        print("[mlflow] logged artifacts:")
        print(f"  - data/{csv_filename}")  # 변경
        print(f"  - data/metadata/metadata_{timestamp}.json")  # 변경

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


def main():
    collector = CovidDataCollector(
        options=CollectorOptions(
            filter_in_collector=False
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