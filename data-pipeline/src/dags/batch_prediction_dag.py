# data-pipeline/src/dags/batch_prediction_dag.py
"""
배치 예측 DAG: 정기적으로 최신 모델로 미래 예측 수행 (실시간 버전)
"""
from datetime import datetime, timedelta
import os
from airflow import DAG
from airflow.operators.bash import BashOperator

# ----- Config (환경변수에서 읽기) -----
PROJECT_PATH = os.environ.get("PROJECT_PATH", "/workspace")
PYTHON = os.environ.get("PYTHON_BIN", "python")
MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")

# ✅ 환경변수에서 날짜 설정 읽기
TRAIN_START = os.environ.get("TRAIN_START_DATE", "2020-01-01")
TRAIN_END = os.environ.get("TRAIN_END_DATE", "")
PREDICT_START = os.environ.get("PREDICT_START_DATE", "")

HORIZON = os.environ.get("COVID_HORIZON", "30")
TARGET = os.environ.get("COVID_TARGET", "new_cases")
TEST_DAYS = os.environ.get("COVID_TEST_DAYS", "60")

default_args = {
    "owner": "mlops",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}

with DAG(
        dag_id="covid_batch_prediction",
        start_date=datetime(2025, 10, 1),
        schedule_interval="0 2 * * *",  # 매일 새벽 2시 실행
        catchup=False,
        default_args=default_args,
        tags=["mlflow", "covid", "batch", "prediction", "realtime"],
        description="Daily batch prediction using latest production model (realtime)"
) as dag:
    env = {
        "MLFLOW_TRACKING_URI": MLFLOW_URI,
        "PROJECT_PATH": PROJECT_PATH,
        "PYTHONPATH": "/workspace",

        # AWS
        "AWS_ACCESS_KEY_ID": os.getenv("AWS_ACCESS_KEY_ID", ""),
        "AWS_SECRET_ACCESS_KEY": os.getenv("AWS_SECRET_ACCESS_KEY", ""),
        "AWS_DEFAULT_REGION": os.getenv("AWS_DEFAULT_REGION", "ap-northeast-2"),
        "S3_BUCKET": os.getenv("S3_BUCKET", ""),

        # ✅ 날짜 설정 전달
        "TRAIN_START_DATE": TRAIN_START,
        "TRAIN_END_DATE": TRAIN_END,
        "PREDICT_START_DATE": PREDICT_START,

        # COVID 설정
        "COVID_TARGET": TARGET,
        "COVID_TEST_DAYS": TEST_DAYS,
        "COVID_HORIZON": HORIZON,

        # Email
        "SMTP_HOST": os.getenv("SMTP_HOST", "smtp.gmail.com"),
        "SMTP_PORT": os.getenv("SMTP_PORT", "587"),
        "SMTP_USER": os.getenv("SMTP_USER", ""),
        "SMTP_PASSWORD": os.getenv("SMTP_PASSWORD", ""),
        "EMAIL_TO": os.getenv("EMAIL_TO", ""),
    }

    PRE = f"""cd {PROJECT_PATH} && export PYTHONPATH=/workspace"""

    # S3 경로
    S3_BUCKET = os.getenv("S3_BUCKET", "")
    if S3_BUCKET:
        S3_OUTPUT = f"s3://{S3_BUCKET}/predictions/"
        S3_FEATURES = f"s3://{S3_BUCKET}/features/"
    else:
        S3_OUTPUT = "/workspace/data/predictions/"
        S3_FEATURES = "/workspace/data/features/"

    # 1. 최신 데이터 수집
    collect_latest = BashOperator(
        task_id="collect_latest_data",
        bash_command=f"""{PRE} && python -m src.pipeline.collect \
            --run-name "realtime_collect_$(date +%Y%m%d)" \
            --tracking-uri {MLFLOW_URI}
        """,
        env=env,
    )

    # 2. 전처리
    preprocess_latest = BashOperator(
        task_id="preprocess_latest",
        bash_command=f"""{PRE} && python -m src.pipeline.preprocess \
            --from-latest \
            --run-name "realtime_preprocess_$(date +%Y%m%d)" \
            --tracking-uri {MLFLOW_URI}
        """,
        env=env,
    )

    # 3. 피처 엔지니어링 (✅ train-window-days 제거)
    feature_engineering = BashOperator(
        task_id="feature_engineering",
        bash_command=f"""{PRE} && python -m src.pipeline.fe \
            --run-name "realtime_fe_$(date +%Y%m%d)" \
            --target {TARGET} \
            --output {S3_FEATURES} \
            --tracking-uri {MLFLOW_URI}
        """,
        env=env,
    )

    # 4. 배치 예측 (✅ 실시간)
    batch_predict_and_save = BashOperator(
        task_id="batch_predict_and_save",
        bash_command=f"""{PRE} && python -m src.pipeline.batch_predict \
            --horizon {HORIZON} \
            --model-stage Production \
            --feature-path {S3_FEATURES} \
            --output {S3_OUTPUT}
        """,
        env=env,
    )

    # 5. 성공 알림
    notify_success = BashOperator(
        task_id="notify_success",
        bash_command=f"""{PRE} && python -m src.utils.send_notification \
            --message "실시간 배치 예측이 성공적으로 완료되었습니다. Horizon: {HORIZON}일, Output: {S3_OUTPUT}" \
            --subject "COVID-19 실시간 예측 완료 ($(date +%Y-%m-%d))" \
            --status success \
            --channel email
        """,
        env=env,
    )


    # 6. 실패 시 이메일 알림
    def send_failure_notification(context):
        """Task 실패 시 알림 전송"""
        import subprocess
        task_instance = context.get('task_instance')
        exception = context.get('exception')

        message = f"""
        배치 예측 파이프라인 실패

        Task: {task_instance.task_id}
        DAG: {task_instance.dag_id}
        Execution Date: {context.get('execution_date')}
        Error: {str(exception)[:200]}
        """

        subprocess.run([
            "python3", "-m", "src.utils.send_notification",
            "--message", message,
            "--subject", "COVID-19 배치 예측 실패 알림",
            "--status", "error",
            "--channel", "email"
        ], env=env, cwd=PROJECT_PATH)


    # 모든 Task에 failure callback 추가
    for task in [collect_latest, preprocess_latest, feature_engineering, batch_predict_and_save]:
        task.on_failure_callback = send_failure_notification

    # Task 의존성
    collect_latest >> preprocess_latest >> feature_engineering >> batch_predict_and_save >> notify_success