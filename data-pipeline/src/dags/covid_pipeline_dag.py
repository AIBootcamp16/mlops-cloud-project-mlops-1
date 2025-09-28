# dags/covid_pipeline_dag.py
# -*- coding: utf-8 -*-
"""
Fixed COVID ML pipeline with proper task dependencies and error handling.
"""

from datetime import datetime, timedelta
import os

from airflow import DAG
from airflow.operators.bash import BashOperator

# ----- Config -----
PROJECT_PATH = os.environ.get("PROJECT_PATH", "/opt/airflow/project")
PYTHON = os.environ.get("PYTHON_BIN", "python")
MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
TARGET = os.environ.get("COVID_TARGET", "new_cases")
TEST_DAYS = os.environ.get("COVID_TEST_DAYS", "60")
HORIZON = os.environ.get("COVID_HORIZON", "30")

default_args = {
    "owner": "mlops",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
    "email_on_retry": False,
}

with DAG(
        dag_id="covid_ml_pipeline_fixed",
        start_date=datetime(2025, 9, 1),
        schedule_interval="@daily",  # 시간당 실행에서 일일 실행으로 변경
        catchup=False,
        default_args=default_args,
        tags=["mlflow", "covid", "fixed"],
        description="Fixed COVID ML pipeline with proper dependencies"
) as dag:
    env = {
        "MLFLOW_TRACKING_URI": MLFLOW_URI,
        "PROJECT_PATH": PROJECT_PATH,
        "PYTHONPATH": "/workspace",
        "AWS_ACCESS_KEY_ID": os.getenv("AWS_ACCESS_KEY_ID", ""),
        "AWS_SECRET_ACCESS_KEY": os.getenv("AWS_SECRET_ACCESS_KEY", ""),
        "AWS_DEFAULT_REGION": os.getenv("AWS_DEFAULT_REGION", ""),
    }

    # 공통 프리앰블
    PRE = f"""cd {PROJECT_PATH} && \
    export PYTHONPATH=/workspace && \
    RUN_NAME=$(echo "$AIRFLOW_CTX_DAG_RUN_ID" | tr -cd 'A-Za-z0-9_')"""

    # 1. 데이터 수집
    collect = BashOperator(
        task_id="collect",
        bash_command=f"""{PRE} && python -m src.pipeline.collect --run-name "collection_$RUN_NAME" """,
        env=env,
    )

    # 2. 전처리 (수집 완료 후 실행)
    preprocess = BashOperator(
        task_id="preprocess",
        bash_command=f"""{PRE} && python -m src.pipeline.preprocess --from-latest --run-name "preprocessing_$RUN_NAME" """,
        env=env,
    )

    # 3. 특성 엔지니어링 (전처리 완료 후 실행)
    fe = BashOperator(
        task_id="feature_engineering",
        bash_command=f"""{PRE} && python -m src.pipeline.fe --run-name "fe_$RUN_NAME" --target {TARGET} """,
        env=env,
    )

    # 4. 모델 훈련 (특성 엔지니어링 완료 후 실행)
    train = BashOperator(
        task_id="train",
        bash_command=f"""{PRE} && python -m src.pipeline.train --target {TARGET} --test-days {TEST_DAYS} --horizon {HORIZON} """,
        env=env,
    )

    # 태스크 의존성 명시
    collect >> preprocess >> fe >> train