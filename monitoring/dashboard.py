# monitoring/dashboard.py
"""
MLOps 모니터링 대시보드 (Streamlit) - 수정 버전
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

# FastAPI 엔드포인트
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://fastapi:8000")

# 페이지 설정
st.set_page_config(
    page_title="COVID-19 ML 모니터링",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# MLflow 클라이언트
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(MLFLOW_URI)
client = MlflowClient()

# 사이드바
st.sidebar.title("⚙️ 설정")
st.sidebar.info(f"MLflow URI: {MLFLOW_URI}")

experiment_names = ["covid_ml_pipeline", "batch_prediction"]
experiment_name = st.sidebar.selectbox("Experiment 선택", experiment_names)

if st.sidebar.button("🔄 새로고침"):
    st.rerun()

# 메인 타이틀
st.title("📊 COVID-19 예측 모델 모니터링 대시보드")
st.markdown("---")


# Helper 함수
def get_latest_metrics(experiment_name):
    """최신 실험의 메트릭 가져오기"""
    try:
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment:
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["start_time DESC"],
                max_results=10
            )
            return runs
    except Exception as e:
        st.sidebar.error(f"Experiment 조회 실패: {e}")
    return []


# 1. 모델 성능 추적
st.header("1. 📈 모델 성능 추적")

col1, col2, col3, col4 = st.columns(4)

runs = get_latest_metrics(experiment_name)

if runs:
    latest_run = runs[0]
    metrics = latest_run.data.metrics

    with col1:
        mae = metrics.get('test_MAE', 0)
        st.metric("MAE", f"{mae:.2f}")
    with col2:
        rmse = metrics.get('test_RMSE', 0)
        st.metric("RMSE", f"{rmse:.2f}")
    with col3:
        mape = metrics.get('test_MAPE', 0)
        st.metric("MAPE", f"{mape:.2f}%")
    with col4:
        r2 = metrics.get('test_R2', 0)
        st.metric("R²", f"{r2:.4f}")
else:
    st.warning(f"Experiment '{experiment_name}'에서 실행 기록을 찾을 수 없습니다.")
    st.info("먼저 모델 훈련을 실행해주세요.")

# 2. 성능 변화 추적
st.header("2. 📉 성능 변화 추이")

if runs:
    # 최근 N개 실행의 메트릭 수집
    metric_history = []
    for run in runs:
        metric_history.append({
            'timestamp': datetime.fromtimestamp(run.info.start_time / 1000),
            'run_id': run.info.run_id[:8],
            'MAE': run.data.metrics.get('test_MAE', None),
            'RMSE': run.data.metrics.get('test_RMSE', None),
            'R2': run.data.metrics.get('test_R2', None)
        })

    df_metrics = pd.DataFrame(metric_history)

    if not df_metrics.empty and len(df_metrics) > 1:
        col1, col2 = st.columns(2)

        with col1:
            # MAE 추이
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
            # R² 추이
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

# 3. 최신 배치 예측 결과
st.header("3. 🔮 최신 배치 예측 결과")

try:
    predictions_dir = Path("/workspace/data/predictions")

    if predictions_dir.exists():
        prediction_files = list(predictions_dir.glob("batch_prediction_*.csv"))

        if prediction_files:
            latest_prediction = max(prediction_files, key=lambda p: p.stat().st_mtime)
            df_pred = pd.read_csv(latest_prediction)

            # 날짜 컬럼 확인 및 변환
            if 'date' in df_pred.columns:
                df_pred['date'] = pd.to_datetime(df_pred['date'])

            st.success(f"📁 최신 예측: {latest_prediction.name}")
            st.caption(
                f"생성 시간: {datetime.fromtimestamp(latest_prediction.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')}")

            # 예측 시각화
            if 'predicted_new_cases' in df_pred.columns:
                fig_pred = px.line(
                    df_pred,
                    x='date',
                    y='predicted_new_cases',
                    title='향후 30일 예측',
                    markers=True
                )
                fig_pred.update_layout(
                    xaxis_title='날짜',
                    yaxis_title='예상 신규 확진자 수',
                    hovermode='x unified'
                )
                st.plotly_chart(fig_pred, use_container_width=True)

                # 예측 통계
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("평균 예측값", f"{df_pred['predicted_new_cases'].mean():.0f}")
                with col2:
                    st.metric("최대 예측값", f"{df_pred['predicted_new_cases'].max():.0f}")
                with col3:
                    st.metric("최소 예측값", f"{df_pred['predicted_new_cases'].min():.0f}")

                # 데이터 테이블 (선택적)
                with st.expander("📋 예측 데이터 보기"):
                    st.dataframe(df_pred, use_container_width=True)
            else:
                st.warning("예측 컬럼을 찾을 수 없습니다.")
        else:
            st.warning("예측 결과 파일이 없습니다.")
            st.info("Airflow에서 `covid_batch_prediction` DAG를 실행하면 예측 결과가 생성됩니다.")
    else:
        st.info(f"예측 디렉토리가 없습니다: {predictions_dir}")
        st.info("배치 예측 DAG를 실행하면 예측 결과가 표시됩니다.")

except Exception as e:
    st.error(f"예측 결과 로드 실패: {str(e)}")
    with st.expander("에러 상세"):
        st.code(str(e))

# 4. 모델 레지스트리
st.header("4. 🏷️ 모델 레지스트리")

try:
    models = client.search_registered_models()

    if models:
        for model in models:
            with st.expander(f"📦 {model.name}"):
                versions = client.search_model_versions(f"name='{model.name}'")

                if versions:
                    version_data = []
                    for version in versions[:5]:  # 최근 5개만
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
        st.info("모델을 MLflow Registry에 등록하려면 훈련 후 `mlflow.register_model()`을 사용하세요.")
except Exception as e:
    st.error(f"모델 레지스트리 조회 실패: {str(e)}")

# 5. 시스템 헬스체크
st.header("5. 🏥 시스템 상태")

col1, col2, col3 = st.columns(3)

with col1:
    # MLflow 연결 상태
    try:
        client.search_experiments()
        st.success("✅ MLflow 연결 정상")
    except Exception as e:
        st.error("❌ MLflow 연결 실패")
        st.caption(str(e))

with col2:
    # 최근 파이프라인 실행
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
    # 데이터 신선도
    try:
        data_dir = Path("/workspace/data/processed")
        if data_dir.exists():
            processed_files = list(data_dir.glob("*.csv"))
            if processed_files:
                latest_data = max(processed_files, key=lambda p: p.stat().st_mtime)
                data_age = (datetime.now() - datetime.fromtimestamp(latest_data.stat().st_mtime)).days
                st.info(f"📅 데이터 업데이트: {data_age}일 전")
            else:
                st.warning("📅 처리된 데이터 없음")
        else:
            st.warning("📅 데이터 디렉토리 없음")
    except Exception as e:
        st.warning(f"📅 데이터 확인 실패")

# 6. 실시간 예측 (FastAPI 호출)
st.header("6. 🔮 실시간 예측")

col1, col2 = st.columns([2, 1])

with col1:
    prediction_date = st.date_input("예측 날짜 선택", datetime.now().date())

with col2:
    st.write("")  # 여백
    st.write("")
    if st.button("🚀 예측 실행", type="primary"):
        try:
            response = requests.post(
                f"{FASTAPI_URL}/predict",
                json={"date": prediction_date.strftime("%Y-%m-%d")}
            )

            if response.status_code == 200:
                result = response.json()
                st.success(f"✅ 예측 완료!")
                st.metric(
                    f"{prediction_date} 예상 신규 확진자",
                    f"{result['predicted_new_cases']:,}명"
                )
            else:
                st.error(f"예측 실패: {response.text}")
        except Exception as e:
            st.error(f"API 호출 실패: {e}")

# 경로 수정
predictions_dir = Path("/workspace/data/predictions")
# Docker 볼륨 마운트가 제대로 되었는지 확인
if not predictions_dir.exists():
    predictions_dir = Path("/tmp/predictions")  # 폴백 경로

st.markdown("---")
st.caption("📊 COVID-19 MLOps Dashboard v1.0 | Powered by Streamlit & MLflow")

# 사이드바 추가 정보
st.sidebar.markdown("---")
st.sidebar.subheader("📖 가이드")
st.sidebar.markdown("""
**사용 방법:**
1. Experiment 선택
2. 메트릭 확인
3. 예측 결과 확인
4. 🔄 새로고침으로 업데이트

**링크:**
- [MLflow UI](http://localhost:5000)
- [Airflow UI](http://localhost:8080)
""")