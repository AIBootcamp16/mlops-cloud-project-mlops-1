# data-pipeline/src/config/settings.py
"""프로젝트 설정 파일 - 2022-2023 학습 시나리오"""
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class DataConfig:
    """데이터 관련 설정"""
    OWID_URL: str = "https://catalog.ourworldindata.org/garden/covid/latest/compact/compact.csv"
    TARGET_COUNTRY: str = "South Korea"
    COUNTRY_COLUMN: str = "location"  # OWID는 'location' 사용
    CHUNK_SIZE: int = 50000  # 청크 크기 증가

    # 학습 기간: 2022년 전체 + 2023년 Q1
    TRAIN_START_DATE: str = "2022-01-01"
    TRAIN_END_DATE: str = "2023-03-31"

    # 검증/예측 기간: 2023년 Q2-Q3
    PREDICT_START_DATE: str = "2023-04-01"
    PREDICT_END_DATE: str = "2023-09-30"

    FILTER_FUTURE_DATES: bool = True


@dataclass
class MLflowConfig:
    """MLflow 관련 설정"""
    TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    EXPERIMENT_NAME: str = "covid_prediction_2022_2023"
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