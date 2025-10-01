# data-pipeline/src/dags/covid_pipeline_dag.py
"""
COVID ML 학습 파이프라인: 주기적으로 전체 데이터로 모델 재학습
"""
from datetime import datetime, timedelta
import os
from airflow import DAG
from airflow.operators.bash import BashOperator

# ----- Config -----
PROJECT_PATH = os.environ.get("PROJECT_PATH", "/workspace")
PYTHON = os.environ.get("PYTHON_BIN", "python")
MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")

# ✅ 환경변수에서 설정 읽기
TARGET = os.environ.get("COVID_TARGET", "new_cases")
TEST_DAYS = os.environ.get("COVID_TEST_DAYS", "60")
HORIZON = os.environ.get("COVID_HORIZON", "30")

TRAIN_START = os.environ.get("TRAIN_START_DATE", "2020-01-01")
TRAIN_END = os.environ.get("TRAIN_END_DATE", "")

default_args = {
    "owner": "mlops",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}

with DAG(
        dag_id="covid_ml_training_pipeline",
        start_date=datetime(2025, 10, 1),
        schedule_interval="0 3 * * 0",  # 매주 일요일 새벽 3시 (재학습)
        catchup=False,
        default_args=default_args,
        tags=["mlflow", "covid", "training", "realtime"],
        description="Weekly model training with full historical data"
) as dag:
    env = {
        "MLFLOW_TRACKING_URI": MLFLOW_URI,
        "PROJECT_PATH": PROJECT_PATH,
        "PYTHONPATH": "/workspace",

        # AWS
        "AWS_ACCESS_KEY_ID": os.getenv("AWS_ACCESS_KEY_ID", ""),
        "AWS_SECRET_ACCESS_KEY": os.getenv("AWS_SECRET_ACCESS_KEY", ""),
        "AWS_DEFAULT_REGION": os.getenv("AWS_DEFAULT_REGION", ""),

        # ✅ 날짜 설정
        "TRAIN_START_DATE": TRAIN_START,
        "TRAIN_END_DATE": TRAIN_END,

        # COVID 설정
        "COVID_TARGET": TARGET,
        "COVID_TEST_DAYS": TEST_DAYS,
        "COVID_HORIZON": HORIZON,
    }

    PRE = f"""cd {PROJECT_PATH} && export PYTHONPATH=/workspace && \
        RUN_NAME=$(echo "$AIRFLOW_CTX_DAG_RUN_ID" | tr -cd 'A-Za-z0-9_')"""

    # 1. 데이터 수집
    collect = BashOperator(
        task_id="collect",
        bash_command=f"""{PRE} && python -m src.pipeline.collect \
            --run-name "training_collection_$RUN_NAME" \
            --tracking-uri {MLFLOW_URI}
        """,
        env=env,
    )

    # 2. 전처리
    preprocess = BashOperator(
        task_id="preprocess",
        bash_command=f"""{PRE} && python -m src.pipeline.preprocess \
            --from-latest \
            --run-name "training_preprocessing_$RUN_NAME" \
            --tracking-uri {MLFLOW_URI}
        """,
        env=env,
    )

    # 3. 피처 엔지니어링 (✅ 전체 데이터)
    fe = BashOperator(
        task_id="feature_engineering",
        bash_command=f"""{PRE} && python -m src.pipeline.fe \
            --run-name "training_fe_$RUN_NAME" \
            --target {TARGET} \
            --tracking-uri {MLFLOW_URI}
        """,
        env=env,
    )

    # 4. 모델 훈련
    train = BashOperator(
        task_id="train",
        bash_command=f"""{PRE} && python -m src.pipeline.train \
            --target {TARGET} \
            --test-days {TEST_DAYS} \
            --horizon {HORIZON} \
            --tracking-uri {MLFLOW_URI}
        """,
        env=env,
    )

    # Task 의존성
    collect >> preprocess >> fe >> train