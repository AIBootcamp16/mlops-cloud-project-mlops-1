# data-pipeline/src/config/settings.py
"""프로젝트 설정 파일 - 실시간 예측 지원"""
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class DataConfig:
    """데이터 관련 설정"""
    OWID_URL: str = "https://catalog.ourworldindata.org/garden/covid/latest/compact/compact.csv"
    TARGET_COUNTRY: str = "South Korea"
    COUNTRY_COLUMN: str = "location"
    CHUNK_SIZE: int = 50000

    # ✅ 실시간 예측을 위한 날짜 설정
    TRAIN_START_DATE: str = "2020-01-01"
    TRAIN_END_DATE: str = ""  # 빈 값이면 어제 날짜 자동 사용

    TEST_DAYS: int = 60
    PREDICT_HORIZON: int = 30
    PREDICT_START_DATE: str = ""

    FILTER_FUTURE_DATES: bool = True  # ✅ 미래 날짜 필터링 활성화

    def get_train_end_date(self) -> str:
        """✅ 학습 종료 날짜 계산 (어제)"""
        if self.TRAIN_END_DATE and self.TRAIN_END_DATE.strip():
            return self.TRAIN_END_DATE

        # 어제 날짜 반환
        import pandas as pd
        yesterday = pd.Timestamp.now().normalize() - pd.Timedelta(days=1)
        return yesterday.strftime("%Y-%m-%d")

    def get_predict_start_date(self, last_train_date: str = None) -> str:
        """✅ 예측 시작 날짜 계산"""
        if self.PREDICT_START_DATE and self.PREDICT_START_DATE.strip():
            return self.PREDICT_START_DATE

        # 학습 데이터 마지막 날 다음날
        if last_train_date:
            from datetime import datetime, timedelta
            last_date = datetime.strptime(last_train_date, "%Y-%m-%d")
            return (last_date + timedelta(days=1)).strftime("%Y-%m-%d")

        # 기본값: 오늘
        import pandas as pd
        today = pd.Timestamp.now().normalize()
        return today.strftime("%Y-%m-%d")


@dataclass
class MLflowConfig:
    """MLflow 관련 설정"""
    TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    EXPERIMENT_NAME: str = "covid_prediction_realtime"
    ARTIFACT_PATH: str = "raw_data"


class ProjectConfig:
    """전체 프로젝트 설정"""

    def __init__(self, tracking_uri: Optional[str] = None):
        self.data = DataConfig()
        self.mlflow = MLflowConfig()
        if tracking_uri:
            self.mlflow.TRACKING_URI = tracking_uri

        self.DATA_DIR = "/workspace/data"
        self.RAW_DATA_DIR = f"{self.DATA_DIR}/raw"
        self.PROCESSED_DATA_DIR = f"{self.DATA_DIR}/processed"