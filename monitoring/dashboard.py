# monitoring/dashboard.py
"""
MLOps 모니터링 대시보드 (Streamlit) - 실험 이름 수정 버전
"""
import requests
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from mlflow.tracking import MlflowClient
import mlflow
from datetime import datetime, timedelta
from pathlib import Path
import os
import boto3
from io import BytesIO
from urllib.parse import urlparse

# 환경 변수
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://fastapi:8000")
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
S3_BUCKET = os.getenv("S3_BUCKET", "")
S3_PREDICTIONS_PATH = os.getenv("S3_PREDICTIONS_PATH", "predictions/")

# 페이지 설정
st.set_page_config(
    page_title="COVID-19 ML 모니터링",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# MLflow 클라이언트
mlflow.set_tracking_uri(MLFLOW_URI)
client = MlflowClient()

# 사이드바
st.sidebar.title("⚙️ 설정")
st.sidebar.info(f"MLflow URI: {MLFLOW_URI}")
if S3_BUCKET:
    st.sidebar.info(f"S3 Bucket: {S3_BUCKET}")

# ✅ 실제 존재하는 실험 이름으로 수정
experiment_names = [
    "covid_model_training_integrated",
    "covid_feature_engineering",
    "covid_data_preprocessing",
    "covid_prediction_realtime"
]
experiment_name = st.sidebar.selectbox("Experiment 선택", experiment_names)

if st.sidebar.button("🔄 새로고침"):
    st.rerun()

# 메인 타이틀
st.title("📊 COVID-19 예측 모델 모니터링 대시보드")
st.markdown("---")


# ==================== S3 Helper Functions ====================
def _read_csv_from_s3(s3_uri: str) -> pd.DataFrame:
    """S3에서 CSV 직접 읽기"""
    u = urlparse(s3_uri)
    bucket, key = u.netloc, u.path.lstrip("/")
    obj = boto3.client("s3").get_object(Bucket=bucket, Key=key)
    return pd.read_csv(BytesIO(obj["Body"].read()))


def load_latest_prediction_from_s3() -> pd.DataFrame:
    """S3에서 최신 배치 예측 결과 로드"""
    if not S3_BUCKET:
        return None

    try:
        s3 = boto3.client("s3")
        paginator = s3.get_paginator("list_objects_v2")

        csv_files = []
        for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=S3_PREDICTIONS_PATH):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith(".csv") and "batch_prediction_" in key:
                    csv_files.append({
                        "key": key,
                        "last_modified": obj["LastModified"],
                        "size": obj["Size"]
                    })

        if not csv_files:
            return None

        latest = max(csv_files, key=lambda x: x["last_modified"])
        st.sidebar.success(f"✅ S3에서 로드: {os.path.basename(latest['key'])}")

        return _read_csv_from_s3(f"s3://{S3_BUCKET}/{latest['key']}")

    except Exception as e:
        st.sidebar.error(f"S3 로드 실패: {e}")
        return None


def load_latest_prediction_from_mlflow() -> pd.DataFrame:
    """MLflow 아티팩트에서 최신 예측 결과 로드"""
    try:
        # ✅ 실제 실험 이름들 시도
        prediction_experiments = [
            "covid_model_training_integrated",
            "covid_prediction_realtime"
        ]

        for exp_name in prediction_experiments:
            try:
                exp = client.get_experiment_by_name(exp_name)
                if not exp:
                    continue

                runs = mlflow.search_runs(
                    [exp.experiment_id],
                    order_by=["start_time DESC"],
                    max_results=5
                )

                if runs.empty:
                    continue

                # 각 런에서 예측 CSV 찾기
                for _, run in runs.iterrows():
                    run_id = run["run_id"]
                    try:
                        artifact_paths = ["forecast", "predictions", "results"]

                        for path in artifact_paths:
                            try:
                                dir_path = mlflow.artifacts.download_artifacts(
                                    f"runs:/{run_id}/{path}"
                                )
                                csv_files = [f for f in os.listdir(dir_path)
                                             if f.endswith(".csv")]

                                if csv_files:
                                    csv_file = os.path.join(dir_path, sorted(csv_files)[-1])
                                    df = pd.read_csv(csv_file)
                                    st.sidebar.success(f"✅ MLflow에서 로드: {os.path.basename(csv_file)}")
                                    return df

                            except Exception:
                                continue

                    except Exception:
                        continue

            except Exception:
                continue

        return None

    except Exception as e:
        st.sidebar.error(f"MLflow 로드 실패: {e}")
        return None


def load_prediction_results() -> pd.DataFrame:
    """예측 결과 로드 (S3 → MLflow → 로컬 순)"""

    # 1순위: S3
    if S3_BUCKET:
        df = load_latest_prediction_from_s3()
        if df is not None:
            return df

    # 2순위: MLflow
    df = load_latest_prediction_from_mlflow()
    if df is not None:
        return df

    # 3순위: 로컬 경로
    predictions_dir = Path("/workspace/data/predictions")
    if not predictions_dir.exists():
        predictions_dir = Path("/tmp/predictions")

    if predictions_dir.exists():
        prediction_files = list(predictions_dir.glob("batch_prediction_*.csv"))
        if prediction_files:
            latest_file = max(prediction_files, key=lambda p: p.stat().st_mtime)
            st.sidebar.info(f"📁 로컬에서 로드: {latest_file.name}")
            return pd.read_csv(latest_file)

    return None


def get_latest_metrics(experiment_name):
    """최신 실험의 메트릭 가져오기"""
    try:
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment:
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["start_time DESC"],
                max_results=10
            )
            return runs
    except Exception as e:
        st.sidebar.error(f"Experiment 조회 실패: {e}")
    return []


# ==================== 1. 모델 성능 추적 ====================
st.header("1. 📈 모델 성능 추적")

col1, col2, col3, col4 = st.columns(4)

runs = get_latest_metrics(experiment_name)

if runs:
    latest_run = runs[0]
    metrics = latest_run.data.metrics

    with col1:
        # ✅ 여러 메트릭 이름 지원
        mae = (metrics.get('test_MAE') or
               metrics.get('best_model_test_mae') or
               metrics.get('random_forest_test_mae') or 0)
        st.metric("MAE", f"{mae:.2f}")

    with col2:
        rmse = (metrics.get('test_RMSE') or
                metrics.get('best_model_test_rmse') or
                metrics.get('random_forest_test_rmse') or 0)
        st.metric("RMSE", f"{rmse:.2f}")

    with col3:
        mape = (metrics.get('test_MAPE') or
                metrics.get('best_model_test_mape') or 0)
        st.metric("MAPE", f"{mape:.2f}%")

    with col4:
        r2 = (metrics.get('test_R2') or
              metrics.get('best_model_test_r2') or
              metrics.get('random_forest_test_r2') or 0)
        st.metric("R²", f"{r2:.4f}")
else:
    st.warning(f"Experiment '{experiment_name}'에서 실행 기록을 찾을 수 없습니다.")
    st.info("먼저 모델 훈련을 실행해주세요.")

# ==================== 2. 성능 변화 추적 ====================
st.header("2. 📉 성능 변화 추이")

if runs:
    metric_history = []
    for run in runs:
        metrics = run.data.metrics
        metric_history.append({
            'timestamp': datetime.fromtimestamp(run.info.start_time / 1000),
            'run_id': run.info.run_id[:8],
            'MAE': (metrics.get('test_MAE') or
                    metrics.get('best_model_test_mae') or
                    metrics.get('random_forest_test_mae')),
            'RMSE': (metrics.get('test_RMSE') or
                     metrics.get('best_model_test_rmse') or
                     metrics.get('random_forest_test_rmse')),
            'R2': (metrics.get('test_R2') or
                   metrics.get('best_model_test_r2') or
                   metrics.get('random_forest_test_r2'))
        })

    df_metrics = pd.DataFrame(metric_history)
    # None 값 제거
    df_metrics = df_metrics.dropna(subset=['MAE', 'R2'])

    if not df_metrics.empty and len(df_metrics) > 1:
        col1, col2 = st.columns(2)

        with col1:
            fig_mae = px.line(
                df_metrics,
                x='timestamp',
                y='MAE',
                title='MAE 변화 추이',
                markers=True,
                hover_data=['run_id']
            )
            fig_mae.update_layout(xaxis_title='시간', yaxis_title='MAE')
            st.plotly_chart(fig_mae, use_container_width=True)

        with col2:
            fig_r2 = px.line(
                df_metrics,
                x='timestamp',
                y='R2',
                title='R² 변화 추이',
                markers=True,
                hover_data=['run_id']
            )
            fig_r2.update_layout(xaxis_title='시간', yaxis_title='R²')
            st.plotly_chart(fig_r2, use_container_width=True)
    else:
        st.info("성능 추이를 보려면 최소 2개 이상의 실행이 필요합니다.")

# ==================== 3. 최신 배치 예측 결과 ====================
st.header("3. 🔮 최신 배치 예측 결과")

try:
    df_pred = load_prediction_results()

    if df_pred is not None:
        if 'date' in df_pred.columns:
            df_pred['date'] = pd.to_datetime(df_pred['date'])
            st.success(f"📊 예측 데이터 로드 성공 ({len(df_pred)} rows)")

        pred_col = None
        for col in ['predicted_new_cases', 'yhat', 'prediction']:
            if col in df_pred.columns:
                pred_col = col
                break

        if pred_col:
            fig_pred = px.line(
                df_pred,
                x='date',
                y=pred_col,
                title='향후 예측',
                markers=True
            )
            fig_pred.update_layout(
                xaxis_title='날짜',
                yaxis_title='예상 신규 확진자 수',
                hovermode='x unified'
            )
            st.plotly_chart(fig_pred, use_container_width=True)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("평균 예측값", f"{df_pred[pred_col].mean():.0f}")
            with col2:
                st.metric("최대 예측값", f"{df_pred[pred_col].max():.0f}")
            with col3:
                st.metric("최소 예측값", f"{df_pred[pred_col].min():.0f}")

            with st.expander("📋 예측 데이터 보기"):
                st.dataframe(df_pred, use_container_width=True)
        else:
            st.warning("예측 컬럼을 찾을 수 없습니다.")
            st.dataframe(df_pred.head())
    else:
        st.warning("예측 결과를 찾을 수 없습니다.")
        st.info("Airflow에서 `covid_batch_prediction` DAG를 실행하면 예측 결과가 생성됩니다.")

except Exception as e:
    st.error(f"예측 결과 로드 실패: {str(e)}")
    with st.expander("에러 상세"):
        import traceback

        st.code(traceback.format_exc())

# ==================== 4. 모델 레지스트리 ====================
st.header("4. 🏷️ 모델 레지스트리")

try:
    models = client.search_registered_models()

    if models:
        for model in models:
            with st.expander(f"📦 {model.name}"):
                versions = client.search_model_versions(f"name='{model.name}'")

                if versions:
                    version_data = []
                    for version in versions[:5]:
                        version_data.append({
                            'Version': version.version,
                            'Stage': version.current_stage,
                            'Status': version.status,
                            'Created': datetime.fromtimestamp(int(version.creation_timestamp) / 1000).strftime(
                                '%Y-%m-%d %H:%M')
                        })

                    df_versions = pd.DataFrame(version_data)
                    st.dataframe(df_versions, use_container_width=True)
                else:
                    st.info("등록된 버전이 없습니다.")
    else:
        st.info("등록된 모델이 없습니다.")
except Exception as e:
    st.error(f"모델 레지스트리 조회 실패: {str(e)}")

# ==================== 5. 시스템 헬스체크 ====================
st.header("5. 🏥 시스템 상태")

col1, col2, col3 = st.columns(3)

with col1:
    try:
        client.search_experiments()
        st.success("✅ MLflow 연결 정상")
    except Exception as e:
        st.error("❌ MLflow 연결 실패")

with col2:
    if runs:
        last_run_time = datetime.fromtimestamp(runs[0].info.start_time / 1000)
        hours_ago = (datetime.now() - last_run_time).total_seconds() / 3600

        if hours_ago < 24:
            st.success(f"🕐 마지막 실행: {hours_ago:.1f}시간 전")
        else:
            st.warning(f"⚠️ 마지막 실행: {hours_ago / 24:.1f}일 전")
    else:
        st.info("⚠️ 실행 기록 없음")

with col3:
    if S3_BUCKET:
        try:
            s3 = boto3.client("s3")
            s3.head_bucket(Bucket=S3_BUCKET)
            st.success("✅ S3 연결 정상")
        except Exception:
            st.error("❌ S3 연결 실패")
    else:
        st.info("📦 S3 미설정")

# ==================== 6. 실시간 예측 ====================
st.header("6. 🔮 실시간 예측")

col1, col2 = st.columns([2, 1])

with col1:
    prediction_date = st.date_input("예측 날짜 선택", datetime.now().date())

with col2:
    st.write("")
    st.write("")
    if st.button("🚀 예측 실행", type="primary"):
        try:
            response = requests.post(
                f"{FASTAPI_URL}/predict",
                json={"date": prediction_date.strftime("%Y-%m-%d")},
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                st.success("✅ 예측 완료!")
                st.metric(
                    f"{prediction_date} 예상 신규 확진자",
                    f"{result['predicted_new_cases']:,}명"
                )

                with st.expander("📊 예측 상세 정보"):
                    st.json(result)
            else:
                st.error(f"예측 실패: {response.text}")
        except Exception as e:
            st.error(f"API 호출 실패: {e}")

st.markdown("---")
st.caption("📊 COVID-19 MLOps Dashboard v2.0 | Powered by Streamlit & MLflow")

# 사이드바
st.sidebar.markdown("---")
st.sidebar.subheader("📖 가이드")
st.sidebar.markdown("""
**실험 목록:**
- `covid_model_training_integrated`: 모델 훈련
- `covid_feature_engineering`: 피처 생성
- `covid_data_preprocessing`: 데이터 전처리
- `covid_prediction_realtime`: 실시간 예측

**데이터 소스:**
- S3 버킷 (우선)
- MLflow 아티팩트
- 로컬 경로 (폴백)

**링크:**
- [MLflow UI](http://localhost:5000)
- [Airflow UI](http://localhost:8080)
- [FastAPI Docs](http://localhost:8000/docs)
""")