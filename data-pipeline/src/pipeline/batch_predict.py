# data-pipeline/src/pipeline/batch_predict.py
# -*- coding: utf-8 -*-
"""
배치 예측 스크립트: MLflow에서 프로덕션(또는 대체 스테이지/최신) 모델을 로드하여 미래 예측
- Production 비었을 때 Staging → 최신 버전 폴백
- 모델 버전의 Run 아티팩트(features/)에서 최신 피처 CSV 검색 → 실패 시 --feature-path 폴백
- 로컬 하드코딩 경로 제거 (/workspace/data/processed/features.csv)
- 로깅/예외 메시지 강화
"""
import boto3, tempfile
from urllib.parse import urlparse
from io import BytesIO, StringIO
import argparse
import logging
import os
import glob
from pathlib import Path
from datetime import datetime

import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)


# -----------------------------
# Model loading with stage fallback
# -----------------------------
def _pick_latest_version(client: MlflowClient, model_name: str, stage: str | None):
    """주어진 stage에서 최신 모델 버전 리스트를 얻고 첫 번째를 리턴. 없으면 None."""
    try:
        latest = client.get_latest_versions(model_name, [stage] if stage else None)
        return latest[0] if latest else None
    except Exception as e:
        logger.warning(f"[model] get_latest_versions({model_name}, stage={stage}) failed: {e}")
        return None


def load_model_with_fallback(model_name: str, preferred_stage: str | None = "Production"):
    """
    MLflow Registry에서 모델을 로드한다.
    우선순위: preferred_stage(기본 Production) → 'Staging' → stage=None(전체 최신)
    """
    client = MlflowClient()

    # 1) Preferred stage
    mv = _pick_latest_version(client, model_name, preferred_stage)
    picked_stage = preferred_stage

    # 2) Staging fallback
    if mv is None and preferred_stage and preferred_stage.lower() != "staging":
        logger.warning(f"[model] No versions in '{preferred_stage}'. Falling back to 'Staging'...")
        mv = _pick_latest_version(client, model_name, "Staging")
        picked_stage = "Staging"

    # 3) Latest of all stages
    if mv is None:
        logger.warning("[model] No versions in 'Staging'. Falling back to latest among all stages...")
        mv = _pick_latest_version(client, model_name, None)
        picked_stage = None

    if mv is None:
        raise RuntimeError(
            f"[model] No registered versions found for model '{model_name}'. "
            f"Check the model name or register/transition a version to a stage."
        )

    # URI는 stage 기반이 가장 간단하지만, stage=None일 때는 버전 명시로 로드
    if picked_stage:
        model_uri = f"models:/{model_name}/{picked_stage}"
        logger.info(f"[model] Loading model by stage: {model_name} (stage={picked_stage}) -> v{mv.version}")
    else:
        model_uri = f"models:/{model_name}/{mv.version}"
        logger.info(f"[model] Loading model by version: {model_name} (version={mv.version})")

    # 환경 호환성 이슈 완화: 우선 pyfunc 시도 → 실패 시 sklearn 로더 시도
    try:
        model = mlflow.pyfunc.load_model(model_uri)
        loader = "pyfunc"
    except Exception as e1:
        logger.warning(f"[model] pyfunc.load_model failed ({e1}). Trying mlflow.sklearn.load_model ...")
        try:
            model = mlflow.sklearn.load_model(model_uri)
            loader = "sklearn"
        except Exception as e2:
            logger.error(f"[model] sklearn.load_model also failed: {e2}")
            # `_loss` 같은 모듈 미존재시 친절한 힌트
            raise RuntimeError(
                f"Failed to load model '{model_name}' (v{mv.version}). "
                f"Environment mismatch or missing custom modules may exist. "
                f"Consider pinning sklearn version at training-time and/or logging with code_path. "
                f"Original errors: pyfunc={e1} | sklearn={e2}"
            ) from e2

    logger.info(f"[model] Loaded successfully via {loader}. Run ID: {mv.run_id}")
    return model, mv  # mv: ModelVersion (has .version, .run_id, ...)


# -----------------------------
# Feature loading from MLflow artifacts (with local fallback)
# -----------------------------
def load_features_from_mlflow_artifacts(model_name: str, model_version: int) -> pd.DataFrame:
    """모델 버전의 Run 아티팩트 아래 'features/'에서 가장 최신 CSV 하나를 읽어온다."""
    client = MlflowClient()
    mv = client.get_model_version(model_name, str(model_version))
    run_id = mv.run_id
    logger.info(f"[features] Searching artifacts in run: {run_id}")

    # features/ 디렉터리 다운로드 시도
    try:
        local_dir = mlflow.artifacts.download_artifacts(f"runs:/{run_id}/features")
    except Exception as e:
        raise FileNotFoundError(f"[features] 'features/' artifacts not found for run {run_id}: {e}")

    csv_candidates = sorted(glob.glob(os.path.join(local_dir, "*.csv")))
    if not csv_candidates:
        raise FileNotFoundError(f"[features] No CSV files under 'features/' for run {run_id}")

    picked = csv_candidates[-1]
    logger.info(f"[features] Picked CSV: {picked}")
    df = pd.read_csv(picked)
    # date 컬럼을 가능하면 datetime으로
    if "date" in df.columns:
        try:
            df["date"] = pd.to_datetime(df["date"])
        except Exception:
            pass
    return df


def load_features(feature_path: str | None, model_name: str, model_version: int) -> pd.DataFrame:
    """
    우선순위:
      1) 모델 버전 run의 features/ 아티팩트에서 최신 CSV
      2) --feature-path로 넘어온 CSV 경로(로컬/S3 로딩은 별도 처리 필요시 확장)
    """
    # 1) MLflow artifacts
    try:
        df = load_features_from_mlflow_artifacts(model_name, model_version)
        logger.info("[features] Loaded from MLflow artifacts.")
        return df
    except Exception as e:
        logger.warning(f"[features] MLflow artifact load failed: {e}")

    # 2) CLI 경로 폴백
    if feature_path:
        logger.info(f"[features] Falling back to feature_path: {feature_path}")
        if feature_path.startswith("s3://"):
            if feature_path.endswith(".csv"):
                df = _read_csv_from_s3(feature_path)
            else:
                df = _read_latest_csv_under_prefix(feature_path)
        else:
            df = pd.read_csv(feature_path)

        # ⭐ float64 → float32 변환
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = df[col].astype('float32')

        if "date" in df.columns:
            try:
                df["date"] = pd.to_datetime(df["date"])
            except Exception:
                pass
        return df

    raise FileNotFoundError(
        "[features] Could not load features from MLflow artifacts, and no --feature-path was provided."
    )

# -----------------------------
# Prediction
# -----------------------------
def make_predictions(model, df: pd.DataFrame, horizon: int = 30, target: str = "new_cases"):
    """✅ 수정: 실시간 예측 - 데이터 마지막 날부터 예측"""
    logger.info(f"[predict] Predicting next {horizon} days from latest data")

    # ✅ 'date' 컬럼 확인 및 정렬
    if 'date' not in df.columns:
        # 인덱스가 날짜인지 확인
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
            df.rename(columns={'index': 'date'}, inplace=True)
        else:
            raise ValueError("[predict] 'date' column not found and index is not DatetimeIndex")

    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # 입력 피처 집합
    ignore_cols = {target, "date", "y_next"}
    feature_cols = [c for c in df.columns if c not in ignore_cols and pd.api.types.is_numeric_dtype(df[c])]

    if not feature_cols:
        raise ValueError("[predict] No numeric feature columns available")

    # ✅ 마지막 관측 데이터
    last = df.iloc[-1]
    last_date = last["date"]

    logger.info(f"[predict] Last training date: {last_date.date()}")
    logger.info(f"[predict] Prediction start: {(last_date + pd.Timedelta(days=1)).date()}")
    logger.info(f"[predict] Prediction end: {(last_date + pd.Timedelta(days=horizon)).date()}")

    X = last[feature_cols].to_frame().T

    preds, dates = [], []

    def _predict_one(x_df):
        try:
            return float(model.predict(x_df)[0])
        except AttributeError:
            return float(model.predict(x_df).iloc[0])

    # ✅ 예측 시작: 마지막 날 다음날부터
    for i in range(1, horizon + 1):
        pred = _predict_one(X)
        pred_date = last_date + pd.Timedelta(days=i)

        preds.append(max(0, pred))  # 음수 예측 방지
        dates.append(pred_date)

    result = pd.DataFrame({
        "date": dates,
        "predicted_new_cases": preds,
        "prediction_timestamp": datetime.now(),
        "last_train_date": last_date,  # ✅ 학습 데이터 마지막 날 기록
    })

    logger.info(f"[predict] Generated {len(result)} predictions")
    logger.info(f"[predict] Date range: {result['date'].min().date()} ~ {result['date'].max().date()}")

    return result


# -----------------------------
# Save predictions (local or S3)
# -----------------------------
def save_predictions(predictions: pd.DataFrame, output_path: str | None):
    """예측 결과 저장 + MLflow 로깅"""
    if output_path is None:
        # 기본 로컬 저장소로 저장
        out_dir = Path("/tmp/predictions")
        out_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = str(out_dir / f"batch_prediction_{timestamp}.csv")

    if output_path.startswith("s3://"):
        # S3 업로드
        try:
            import boto3
            from io import StringIO

            s3_path = output_path.replace("s3://", "")
            bucket = s3_path.split("/")[0]
            key = "/".join(s3_path.split("/")[1:])
            if not key or key.endswith("/"):
                key = key.rstrip("/") + f"/batch_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

            logger.info(f"[save] Uploading to s3://{bucket}/{key}")
            buf = StringIO()
            predictions.to_csv(buf, index=False)
            boto3.client("s3").put_object(
                Bucket=bucket, Key=key, Body=buf.getvalue().encode("utf-8"), ContentType="text/csv"
            )
            saved_path = f"s3://{bucket}/{key}"
            logger.info(f"[save] ✅ Saved to {saved_path}")
        except Exception as e:
            logger.error(f"[save] ❌ S3 upload failed: {e}")
            raise
    else:
        # 로컬 저장
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(out_path, index=False)
        saved_path = str(out_path)
        logger.info(f"[save] Saved locally: {saved_path}")

    # MLflow 로깅(가능한 경우)
    try:
        with mlflow.start_run(run_name=f"batch_prediction_{datetime.now().strftime('%Y%m%d')}"):
            mlflow.log_param("horizon", len(predictions))
            mlflow.log_param("prediction_time", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            mlflow.log_param("output_path", saved_path)
            if "predicted_new_cases" in predictions.columns:
                mlflow.log_metric("mean_prediction", float(predictions["predicted_new_cases"].mean()))
                mlflow.log_metric("max_prediction", float(predictions["predicted_new_cases"].max()))
                mlflow.log_metric("min_prediction", float(predictions["predicted_new_cases"].min()))
            if not saved_path.startswith("s3://"):
                mlflow.log_artifact(saved_path)
    except Exception as e:
        logger.warning(f"[save] MLflow logging skipped: {e}")

    return saved_path

def _read_csv_from_s3(s3_uri: str) -> pd.DataFrame:
    u = urlparse(s3_uri)
    assert u.scheme == "s3"
    bucket, key = u.netloc, u.path.lstrip("/")
    obj = boto3.client("s3").get_object(Bucket=bucket, Key=key)
    return pd.read_csv(BytesIO(obj["Body"].read()))

def _read_latest_csv_under_prefix(prefix: str) -> pd.DataFrame:
    u = urlparse(prefix)
    assert u.scheme == "s3"
    bucket, key_prefix = u.netloc, u.path.lstrip("/")

    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    latest = None
    for page in paginator.paginate(Bucket=bucket, Prefix=key_prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].lower().endswith(".csv"):
                if latest is None or obj["LastModified"] > latest["LastModified"]:
                    latest = obj
    if not latest:
        raise FileNotFoundError(f"No CSV under {prefix}")
    return _read_csv_from_s3(f"s3://{bucket}/{latest['Key']}")

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Batch prediction pipeline")
    ap.add_argument("--model-name", type=str, default="covid_prediction_model", help="MLflow Model Registry name")
    ap.add_argument("--model-stage", type=str, default="Production", help="Preferred model stage (e.g., Production)")
    ap.add_argument("--horizon", type=int, default=30, help="Prediction horizon (days)")
    ap.add_argument("--feature-path", type=str, default=None, help="Fallback CSV path if MLflow artifacts missing")
    ap.add_argument("--output", type=str, default=None, help="Output path (local or s3://bucket/path/)")
    args = ap.parse_args()

    # MLflow Tracking URI
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))

    # 1) 모델 로드(스테이지 폴백 포함)
    model, mv = load_model_with_fallback(args.model_name, args.model_stage)

    # 2) 피처 로드(MLflow 아티팩트 → --feature-path)
    df = load_features(args.feature_path, args.model_name, mv.version)

    # 3) 예측
    preds = make_predictions(model, df, args.horizon)

    # 4) 저장
    saved_path = save_predictions(preds, args.output)

    # 5) 요약 로그
    logger.info("[done] Batch prediction completed.")
    logger.info(f"[done] Model: {args.model_name} v{mv.version} (stage={mv.current_stage or 'N/A'})")
    logger.info(f"[done] Horizon: {args.horizon} days")
    logger.info(f"[done] Output: {saved_path}")
    if "predicted_new_cases" in preds.columns:
        logger.info(
            f"[done] Stats → count={len(preds)}, mean={preds['predicted_new_cases'].mean():.2f}, "
            f"max={preds['predicted_new_cases'].max():.2f}, min={preds['predicted_new_cases'].min():.2f}"
        )


if __name__ == "__main__":
    main()
