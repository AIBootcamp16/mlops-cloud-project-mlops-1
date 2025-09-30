# data-pipeline/src/dags/batch_prediction_dag.py
"""
배치 예측 DAG: 정기적으로 최신 모델로 미래 예측 수행
"""
from datetime import datetime, timedelta
import os
from airflow import DAG
from airflow.operators.bash import BashOperator

# ----- Config -----
PROJECT_PATH = os.environ.get("PROJECT_PATH", "/workspace")
PYTHON = os.environ.get("PYTHON_BIN", "python")
MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
HORIZON = os.environ.get("COVID_HORIZON", "30")  # 30일 예측

default_args = {
    "owner": "mlops",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}

with DAG(
        dag_id="covid_batch_prediction",
        start_date=datetime(2025, 9, 1),
        schedule_interval="0 * * * *",  # 매시간 정각마다 실행
        catchup=False,
        default_args=default_args,
        tags=["mlflow", "covid", "batch", "prediction"],
        description="Hourly batch prediction using latest production model"
) as dag:
    env = {
        "MLFLOW_TRACKING_URI": MLFLOW_URI,
        "PROJECT_PATH": PROJECT_PATH,
        "PYTHONPATH": "/workspace",
        "AWS_ACCESS_KEY_ID": os.getenv("AWS_ACCESS_KEY_ID", ""),
        "AWS_SECRET_ACCESS_KEY": os.getenv("AWS_SECRET_ACCESS_KEY", ""),
        "AWS_DEFAULT_REGION": os.getenv("AWS_DEFAULT_REGION", "ap-northeast-2"),
        "S3_BUCKET": os.getenv("S3_BUCKET", ""),
        # Email 알림용
        "SMTP_HOST": os.getenv("SMTP_HOST", "smtp.gmail.com"),
        "SMTP_PORT": os.getenv("SMTP_PORT", "587"),
        "SMTP_USER": os.getenv("SMTP_USER", ""),
        "SMTP_PASSWORD": os.getenv("SMTP_PASSWORD", ""),
        "EMAIL_TO": os.getenv("EMAIL_TO", ""),
    }

    PRE = f"""cd {PROJECT_PATH} && export PYTHONPATH=/workspace"""

    # S3 경로 설정
    S3_BUCKET = os.getenv("S3_BUCKET", "")
    if S3_BUCKET:
        S3_OUTPUT = f"s3://{S3_BUCKET}/predictions/"
        S3_FEATURES = f"s3://{S3_BUCKET}/features/"
    else:
        S3_OUTPUT = "/workspace/data/predictions/"  # Fallback to local
        S3_FEATURES = "/workspace/data/features/"

    # 1. 최신 데이터 수집
    collect_latest = BashOperator(
        task_id="collect_latest_data",
        bash_command=f"""{PRE} && python -m src.pipeline.collect --run-name "batch_collect_$(date +%Y%m%d)" """,
        env=env,
    )

    # 2. 최신 데이터 전처리
    preprocess_latest = BashOperator(
        task_id="preprocess_latest",
        bash_command=f"""{PRE} && python -m src.pipeline.preprocess --from-latest --run-name "batch_preprocess_$(date +%Y%m%d)" """,
        env=env,
    )

    # 3. 피처 엔지니어링
    feature_engineering = BashOperator(
        task_id="feature_engineering",
        bash_command=f"""{PRE} && python -m src.pipeline.fe \
                --run-name "batch_fe_$(date +%Y%m%d)" \
                --target new_cases \
                --output {S3_FEATURES}
            """,
        env=env,
    )

    # 4. 배치 예측 수행 및 저장 (S3)
    batch_predict_and_save = BashOperator(
        task_id="batch_predict_and_save",
        bash_command=f"""{PRE} && \
                python -m src.pipeline.batch_predict \
                    --horizon {HORIZON} \
                    --model-stage Production \
                    --feature-path {S3_FEATURES} \
                    --output {S3_OUTPUT}
            """,
        env=env,
    )

    # 5. 이메일 알림 전송
    notify_success = BashOperator(
        task_id="notify_success",
        bash_command=f"""{PRE} && python -m src.utils.send_notification \
            --message "배치 예측이 성공적으로 완료되었습니다. Horizon: {HORIZON}일, Output: {S3_OUTPUT}" \
            --subject "COVID-19 배치 예측 완료" \
            --status success \
            --channel email
        """,
        env=env,
    )


    # 6. 실패 시 이메일 알림 (on_failure_callback 사용)
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
            "python3", "-m", "src.utils.send_notification",  # ⭐ python → python3
            "--message", message,
            "--subject", "COVID-19 배치 예측 실패 알림",
            "--status", "error",
            "--channel", "email"
        ], env=env, cwd=PROJECT_PATH)


    # 모든 Task에 failure callback 추가
    for task in [collect_latest, preprocess_latest, feature_engineering,
                 batch_predict_and_save]:
        task.on_failure_callback = send_failure_notification

    # Task 의존성
    collect_latest >> preprocess_latest >> feature_engineering >> batch_predict_and_save >> notify_success