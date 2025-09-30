"""프로젝트 설정 파일"""
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class DataConfig:
    """데이터 관련 설정"""
    OWID_URL: str = "https://catalog.ourworldindata.org/garden/covid/latest/compact/compact.csv"
    TARGET_COUNTRY: str = "South Korea"
    COUNTRY_COLUMN: str = "country"
    CHUNK_SIZE: int = 10000
    FILTER_FUTURE_DATES: bool = True


@dataclass
class MLflowConfig:
    """MLflow 관련 설정"""
    # MLRUNS_DIR = r"C:\AIBootCamp\project\mlops\workspace"
    TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "file:///C:/AIBootCamp/project/mlops/workspace")

    PREPATH = TRACKING_URI.replace("file:///", "")
    PREPATH = PREPATH + "/predata"
    PREPATH = os.path.normpath(PREPATH)
    os.makedirs(PREPATH, exist_ok=True)
    PREPATH = PREPATH + "/predata.json"
    PREPATH = os.path.normpath(PREPATH)

    TRACKING_URI = TRACKING_URI + "/mlruns"
    PATH = TRACKING_URI.replace("file:///", "")  # remove URI prefix
    PATH = os.path.normpath(PATH)
    # mlruns 자동 생성
    os.makedirs(PATH, exist_ok=True)

    EXPERIMENT_NAME: str = "covid_data_collection_v3.1"
    ARTIFACT_PATH: str = "raw_data"
    # SKIP_PROCESS: int = 3


class ProjectConfig:
    """전체 프로젝트 설정 - 일반 클래스로 변경"""

    def __init__(self, tracking_uri: Optional[str] = None):
        self.data = DataConfig()
        self.mlflow = MLflowConfig()
        if tracking_uri:  # ★ 추가: 외부에서 URI 주입 허용
            self.mlflow.TRACKING_URI = tracking_uri

        # 디렉토리 설정
        self.DATA_DIR = "C:\AIBootCamp\project\mlops\workspace\mlops-cloud-project-mlops-1\data"
        self.RAW_DATA_DIR = f"{self.DATA_DIR}/raw"
        self.PROCESSED_DATA_DIR = f"{self.DATA_DIR}/processed"