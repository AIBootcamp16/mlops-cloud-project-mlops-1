# api/main.py
# -*- coding: utf-8 -*-
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from urllib.parse import urlparse

import os
import pandas as pd
import numpy as np
import mlflow
import boto3
from io import BytesIO

app = FastAPI(title="COVID-19 Prediction API")

# -------------------- ENV & GLOBALS --------------------
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(MLFLOW_URI)

# 동적 예측 창: "1"(기본) → 학습 최종 관측일+1 ~ +MAX_FORECAST_DAYS
AUTO_PREDICT_WINDOW = os.getenv("AUTO_PREDICT_WINDOW", "1").strip()

# 고정 모드에서만 쓰는 값 (AUTO_PREDICT_WINDOW="0"일 때)
TRAIN_END_DATE = os.getenv("TRAIN_END_DATE", "").strip()       # ""면 자동
PREDICT_START_DATE = os.getenv("PREDICT_START_DATE", "").strip()
PREDICT_END_DATE = os.getenv("PREDICT_END_DATE", "").strip()

MAX_FORECAST_DAYS = int(os.getenv("MAX_FORECAST_DAYS", "30"))

FEATURES_S3_PREFIX = os.getenv("FEATURES_S3_PREFIX", "").strip()
FEATURES_LOCAL_PATH = os.getenv("FEATURES_LOCAL_PATH", "/workspace/data/features/latest_features.csv").strip()

# --- NEW: 오늘까지 창 자동 확장 옵션 & 절대 상한 ---
ALLOW_EXTEND_TO_TODAY = os.getenv("ALLOW_EXTEND_TO_TODAY", "1").strip()
ABS_MAX_FORECAST_DAYS = int(os.getenv("ABS_MAX_FORECAST_DAYS", "365"))

# 모델/피처 공유 상태
model = None
feature_columns: list[str] = []
features_df_cached: Optional[pd.DataFrame] = None
target_history: list[float] = []

# 피처 구성 기본 스펙
LOOKBACKS = [1, 3, 7, 14]
ROLLS = [7, 14, 30]
TARGET = "new_cases"
EPS = 1e-9

# -------------------- IO HELPERS --------------------
def _read_csv_from_s3(s3_uri: str) -> pd.DataFrame:
    u = urlparse(s3_uri)
    if u.scheme != "s3":
        raise ValueError(f"Invalid S3 URI: {s3_uri}")
    bucket, key = u.netloc, u.path.lstrip("/")
    obj = boto3.client("s3").get_object(Bucket=bucket, Key=key)
    return pd.read_csv(BytesIO(obj["Body"].read()))

def _read_latest_csv_under_prefix(prefix: str) -> pd.DataFrame:
    u = urlparse(prefix)
    if u.scheme != "s3":
        raise ValueError(f"Invalid S3 prefix: {prefix}")
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

def load_features_df() -> pd.DataFrame:
    """피처 데이터 로드 (S3 prefix 우선 → 로컬 → fallback)."""
    if FEATURES_S3_PREFIX:
        try:
            print(f"[load_features] Trying S3: {FEATURES_S3_PREFIX}")
            if FEATURES_S3_PREFIX.endswith(".csv"):
                df = _read_csv_from_s3(FEATURES_S3_PREFIX)
            else:
                df = _read_latest_csv_under_prefix(FEATURES_S3_PREFIX)
            print(f"[load_features] S3 load success: {df.shape}")
            return df
        except Exception as e:
            print(f"[load_features] S3 load failed: {e}")

    try:
        print(f"[load_features] Trying local: {FEATURES_LOCAL_PATH}")
        df = pd.read_csv(FEATURES_LOCAL_PATH)
        print(f"[load_features] Local load success: {df.shape}")
        return df
    except Exception as e:
        print(f"[load_features] Local path load failed: {e}")

    print(f"[load_features] Trying fallback: /tmp/features.csv")
    return pd.read_csv("/tmp/features.csv")

# -------------------- HISTORY / FEATURE-NAMES --------------------
def reconstruct_target_history(df: pd.DataFrame) -> List[float]:
    """타겟 히스토리 재구성(결측/음수/inf 방어)."""
    print(f"[reconstruct_history] Starting with df shape: {df.shape}")

    if len(df) == 0:
        print("[reconstruct_history] WARNING: Empty dataframe, using dummy history")
        return [50000.0] * 100  # 안전한 더미

    hist: List[float] = []

    if TARGET in df.columns:
        print(f"[reconstruct_history] Found {TARGET} column")
        for i in range(len(df)):
            try:
                val = float(df.iloc[i][TARGET])
            except Exception:
                continue
            if np.isnan(val) or np.isinf(val):
                continue
            hist.append(max(0.0, val))

        if hist:
            print(f"[reconstruct_history] Collected {len(hist)} values")
            print(f"[reconstruct_history] Stats - min: {min(hist):.2f}, max: {max(hist):.2f}, mean: {np.mean(hist):.2f}")
            return hist

    # fallback: lag_1 기반 복원
    lag_col = f"{TARGET}_lag_1"
    if lag_col in df.columns:
        print(f"[reconstruct_history] Found {lag_col} column")
        for i in range(len(df)):
            try:
                val = float(df.iloc[i][lag_col])
            except Exception:
                continue
            if not np.isnan(val) and not np.isinf(val):
                hist.append(max(0.0, val))

        if hist:
            last_val = np.mean(hist[-7:]) if len(hist) >= 7 else hist[-1]
            hist.append(max(0.0, last_val))
            print(f"[reconstruct_history] From lag_1: {len(hist)} values, mean: {np.mean(hist):.2f}")
            return hist

    print("[reconstruct_history] WARNING: Cannot reconstruct, using dummy history")
    return [50000.0] * 100

def _consecutive_runs(arr: np.ndarray) -> tuple[float, float]:
    """증가/감소 연속 길이(마지막 원소 기준)를 계산"""
    if arr.size < 2:
        return 0.0, 0.0
    diff = np.diff(arr)
    inc = 0
    dec = 0
    for d in diff[::-1]:
        if d > 0:
            if dec > 0:
                break
            inc += 1
        elif d < 0:
            if inc > 0:
                break
            dec += 1
        else:
            break
    return float(inc), float(dec)

def _build_features_for_step(hist: List[float], feature_cols: List[str]) -> dict:
    """hist(현재까지의 예측 포함)로부터 시그니처에 맞는 피처를 생성해 dict로 반환"""
    x = {}
    arr = np.asarray(hist, dtype=float)
    if arr.size == 0:
        arr = np.array([0.0], dtype=float)

    # LAG
    for lag in (1, 3, 7, 14):
        name = f"{TARGET}_lag_{lag}"
        if name in feature_cols:
            idx = -lag
            x[name] = float(arr[idx]) if arr.size >= lag else float(arr[-1])

    # ROLLINGS
    for w in (7, 14, 30):
        win = arr[-w:] if arr.size >= w else arr
        mean_v = float(np.mean(win)) if win.size else 0.0
        std_v  = float(np.std(win)) if win.size else 0.0
        max_v  = float(np.max(win)) if win.size else 0.0
        min_v  = float(np.min(win)) if win.size else 0.0
        mapping = {
            f"{TARGET}_rolling_mean_{w}": mean_v,
            f"{TARGET}_rolling_std_{w}": std_v,
            f"{TARGET}_rolling_max_{w}": max_v,
            f"{TARGET}_rolling_min_{w}": min_v,
        }
        for cname, val in mapping.items():
            if cname in feature_cols:
                x[cname] = val

    # TREND / ACCELERATION
    for w in (7, 14, 30):
        cname = f"{TARGET}_trend_{w}"
        if cname in feature_cols:
            val = 0.0
            if arr.size > w:
                val = float((arr[-1] - arr[-1 - w]) / float(w))
            x[cname] = val
    for w in (7, 14):
        cname = f"{TARGET}_acceleration_{w}"
        if cname in feature_cols:
            val = 0.0
            if arr.size > 2 * w:
                t_now = (arr[-1] - arr[-1 - w]) / float(w)
                t_prev = (arr[-1 - w] - arr[-1 - 2 * w]) / float(w)
                val = float(t_now - t_prev)
            x[cname] = val

    # RATIO TO MA
    for w in (7, 14, 30):
        cname = f"{TARGET}_ratio_to_ma_{w}"
        if cname in feature_cols:
            win = arr[-w:] if arr.size >= w else arr
            ma = float(np.mean(win)) if win.size else 0.0
            x[cname] = float(arr[-1] / (ma + EPS)) if ma > 0 else 1.0

    # VOLATILITY (= rolling std)
    for w in (7, 14):
        cname = f"{TARGET}_volatility_{w}"
        if cname in feature_cols:
            win = arr[-w:] if arr.size >= w else arr
            x[cname] = float(np.std(win)) if win.size else 0.0

    # CONSECUTIVE RUNS
    need_inc = f"{TARGET}_consecutive_increase" in feature_cols
    need_dec = f"{TARGET}_consecutive_decrease" in feature_cols
    if need_inc or need_dec:
        inc, dec = _consecutive_runs(arr)
        if need_inc:
            x[f"{TARGET}_consecutive_increase"] = inc
        if need_dec:
            x[f"{TARGET}_consecutive_decrease"] = dec

    # IS_PEAK (최근 7일 최대와 같으면 peak로)
    if f"{TARGET}_is_peak" in feature_cols:
        win = arr[-7:] if arr.size >= 7 else arr
        x[f"{TARGET}_is_peak"] = float(arr[-1] >= (np.max(win) - EPS)) if win.size else 0.0

    return x

def _safe_cap(hist: List[float]) -> float:
    """출력 상한 계산(폭주 방지)."""
    arr = np.asarray(hist[-90:], dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 10_000.0
    p995 = float(np.quantile(arr, 0.995))
    recent_max = float(np.max(arr))
    return max(10_000.0, 1.5 * p995, 1.5 * recent_max)

def _try_load_feature_names_from_artifacts() -> Optional[List[str]]:
    """모델 최신 버전 런의 metadata/feature_names_*.json 폴백 로드."""
    try:
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        latest = client.get_latest_versions("covid_prediction_model", ["Production"])
        mv = latest[0] if latest else None
        if not mv:
            return None
        try:
            local_dir = mlflow.artifacts.download_artifacts(f"runs:/{mv.run_id}/metadata")
            for name in sorted(os.listdir(local_dir)):
                if name.startswith("feature_names_") and name.endswith(".json"):
                    import json
                    with open(os.path.join(local_dir, name), "r") as f:
                        return json.load(f)
        except Exception:
            pass
    except Exception:
        pass
    return None

def _extract_feature_columns_from_signature() -> Optional[List[str]]:
    try:
        info = mlflow.models.get_model_info("models:/covid_prediction_model/Production")
        sig = info.signature
        if sig and sig.inputs:
            cols = [i.name for i in sig.inputs.inputs]
            return [c for c in cols if c not in {TARGET, "date", "y_next"}]
    except Exception as e:
        print(f"[startup] signature parse failed: {e}")
    return None

# -------------------- STARTUP --------------------
@app.on_event("startup")
def load_model():
    global model, feature_columns, features_df_cached, target_history

    print("\n" + "=" * 50)
    print("STARTUP")
    print("=" * 50)

    # 1) 모델 로드
    model_uri = "models:/covid_prediction_model/Production"
    try:
        model = mlflow.pyfunc.load_model(model_uri)
        print("✓ Model loaded")
    except Exception as e:
        print(f"✗ Model load failed: {e}")
        raise

    # 2) 피처 로드
    try:
        features_df_cached = load_features_df()
        print(f"✓ Features loaded: {features_df_cached.shape}")
        if len(features_df_cached.columns) > 0:
            print(f"  Columns: {list(features_df_cached.columns[:5])}...")
    except Exception as e:
        print(f"✗ Features load failed: {e}")
        raise

    # 3) 날짜 컬럼 정규화
    date_col = None
    for col in ["date", "Date", "DATE", "ds"]:
        if col in features_df_cached.columns:
            date_col = col
            break

    if not date_col:
        raise ValueError("No date column found in features CSV!")

    print(f"  Found date column: {date_col}")
    print(f"  Date dtype before: {features_df_cached[date_col].dtype}")

    # 다양한 포맷 시도
    date_formats = [
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%d-%m-%Y",
        "%m/%d/%Y",
        "%Y%m%d",
        "%Y-%m-%d %H:%M:%S",
    ]
    converted = False
    for fmt in date_formats:
        try:
            features_df_cached[date_col] = pd.to_datetime(features_df_cached[date_col], format=fmt, errors="raise")
            print(f"  ✓ Converted with format: {fmt}")
            converted = True
            break
        except Exception:
            continue
    if not converted:
        print("  Trying automatic inference...")
        features_df_cached[date_col] = pd.to_datetime(features_df_cached[date_col], infer_datetime_format=True, errors="coerce")
        print("  ✓ Converted with automatic inference")

    # 'date'로 통일 + NaT 제거 + 정렬
    if date_col != "date":
        features_df_cached.rename(columns={date_col: "date"}, inplace=True)
        print(f"  Renamed {date_col} → date")

    nat_count = features_df_cached["date"].isna().sum()
    if nat_count > 0:
        print(f"  ⚠ NaT rows: {nat_count}, dropping...")
    features_df_cached = features_df_cached.dropna(subset=["date"]).copy()
    features_df_cached = features_df_cached.sort_values("date").reset_index(drop=True)

    if len(features_df_cached) == 0:
        raise ValueError("All dates were NaT after conversion!")

    print(f"  Date range: {features_df_cached['date'].min()} → {features_df_cached['date'].max()}")

    # 4) 동적/고정 예측 윈도우 계산
    last_train_date = pd.to_datetime(features_df_cached["date"]).max()
    app.state.train_end_dt = last_train_date

    if AUTO_PREDICT_WINDOW == "1":
        app.state.predict_start_dt = last_train_date + pd.Timedelta(days=1)
        app.state.predict_end_dt = last_train_date + pd.Timedelta(days=MAX_FORECAST_DAYS)
        print(f"✓ Predict window (auto): {app.state.predict_start_dt.date()} ~ {app.state.predict_end_dt.date()}")
    else:
        # 고정 모드: ENV 적용(비어있으면 자동 보정)
        fixed_train_end = pd.to_datetime(TRAIN_END_DATE) if TRAIN_END_DATE else last_train_date
        app.state.train_end_dt = fixed_train_end
        app.state.predict_start_dt = pd.to_datetime(PREDICT_START_DATE) if PREDICT_START_DATE else fixed_train_end + pd.Timedelta(days=1)
        app.state.predict_end_dt = pd.to_datetime(PREDICT_END_DATE) if PREDICT_END_DATE else fixed_train_end + pd.Timedelta(days=MAX_FORECAST_DAYS)
        print(f"✓ Predict window (fixed): {app.state.predict_start_dt.date()} ~ {app.state.predict_end_dt.date()}")

    # --- NEW: 오늘까지 창 자동 확장(절대 상한 포함) ---
    try:
        today = pd.Timestamp.today().normalize()
    except Exception:
        today = pd.to_datetime("today").normalize()

    if ALLOW_EXTEND_TO_TODAY == "1":
        hard_limit_end = app.state.train_end_dt + pd.Timedelta(days=ABS_MAX_FORECAST_DAYS)
        desired_end = min(today, hard_limit_end)
        if desired_end > app.state.predict_end_dt:
            old_end = app.state.predict_end_dt
            app.state.predict_end_dt = desired_end
            print(f"✓ Window extended to today: {old_end.date()} → {app.state.predict_end_dt.date()} "
                  f"(cap={ABS_MAX_FORECAST_DAYS}d)")

    # 5) feature_columns 설정(시그니처 → 아티팩트 → 숫자컬럼 폴백)
    feature_columns_local = _extract_feature_columns_from_signature()
    if not feature_columns_local:
        print("[startup] signature missing, trying artifact feature_names...")
        feature_columns_local = _try_load_feature_names_from_artifacts()
    if not feature_columns_local:
        numeric_cols = [c for c in features_df_cached.columns if pd.api.types.is_numeric_dtype(features_df_cached[c])]
        feature_columns_local = [c for c in numeric_cols if c not in {TARGET, "date", "y_next"}]
    feature_columns = feature_columns_local
    print(f"✓ feature_columns: {len(feature_columns)} columns")

    # 6) 히스토리 구성
    target_history_local = reconstruct_target_history(features_df_cached)
    target_history[:] = target_history_local if target_history is not None else target_history_local
    print(f"✓ History reconstructed ({len(target_history)})")

# -------------------- SCHEMAS --------------------
class PredictionRequest(BaseModel):
    date: str  # YYYY-MM-DD

# -------------------- ROUTES --------------------
@app.get("/")
def root():
    train_end = getattr(app.state, "train_end_dt", None)
    p_start = getattr(app.state, "predict_start_dt", None)
    p_end = getattr(app.state, "predict_end_dt", None)
    return {
        "message": "COVID-19 Prediction API",
        "status": "running",
        "training_period": f"~{(train_end.date() if train_end is not None else TRAIN_END_DATE or 'auto')}",
        "prediction_period": f"{(p_start.date() if p_start is not None else (PREDICT_START_DATE or 'auto'))} ~ {(p_end.date() if p_end is not None else (PREDICT_END_DATE or 'auto'))}",
        "max_forecast_days": MAX_FORECAST_DAYS,
        "mode": "auto" if AUTO_PREDICT_WINDOW == "1" else "fixed-env",
        "auto_extend_to_today": ALLOW_EXTEND_TO_TODAY == "1",
        "abs_max_forecast_days": ABS_MAX_FORECAST_DAYS,
    }

@app.post("/predict")
def predict(request: PredictionRequest):
    """예측 엔드포인트: 학습 마지막 날짜 이후의 target_date까지 재귀 예측."""
    print("\n" + "=" * 50)
    print(f"REQUEST: {request.date}")
    print("=" * 50)

    try:
        target_date = pd.to_datetime(request.date).normalize()
        if pd.isna(target_date):
            raise ValueError("Invalid date format. Use YYYY-MM-DD")

        last_train_date = getattr(app.state, "train_end_dt", None)
        predict_start = getattr(app.state, "predict_start_dt", None)
        predict_end = getattr(app.state, "predict_end_dt", None)

        if last_train_date is None or predict_start is None or predict_end is None:
            raise RuntimeError("Server not initialized properly. Missing date boundaries.")

        print(f"Last training date: {last_train_date.date()}")
        print(f"Target date: {target_date.date()}")

        # 오늘 & 절대상한 계산
        try:
            today = pd.Timestamp.today().normalize()
        except Exception:
            today = pd.to_datetime("today").normalize()
        hard_limit_end = last_train_date + pd.Timedelta(days=ABS_MAX_FORECAST_DAYS)

        # 시작일 이전은 그대로 차단
        if target_date < predict_start:
            raise ValueError(f"Target date must be on or after {predict_start.date()}")

        # 범위를 넘었지만, 오늘까지 확장 허용이면 창 확장
        if target_date > predict_end:
            if ALLOW_EXTEND_TO_TODAY == "1" and target_date <= today:
                new_end = min(today, hard_limit_end)
                if target_date <= new_end:
                    old_end = predict_end
                    app.state.predict_end_dt = new_end
                    predict_end = new_end
                    print(f"✓ Extended window for request: {old_end.date()} → {predict_end.date()}")
                else:
                    raise ValueError(
                        f"Target date exceeds absolute max horizon ({ABS_MAX_FORECAST_DAYS} days after {last_train_date.date()})"
                    )
            else:
                raise ValueError(f"Target date must be on or before {predict_end.date()}")

        # 예측 수평(일)
        days_ahead = int((target_date - last_train_date).days)
        max_horizon = int((predict_end - last_train_date).days)
        print(f"Forecast horizon: {days_ahead} days (max {max_horizon} days)")

        if days_ahead <= 0:
            raise ValueError(f"Target must be after {last_train_date.date()}")
        if days_ahead > max_horizon:
            raise ValueError(f"Forecast horizon ({days_ahead}) exceeds current window ({max_horizon} days)")

        # 재귀 예측
        hist = target_history.copy() if target_history else [50000.0] * 100
        corrections = 0

        for i in range(1, days_ahead + 1):
            # 최신 통계(윈도우 7/30) 갱신
            win7  = hist[-7:]  if len(hist) >= 7  else hist
            win30 = hist[-30:] if len(hist) >= 30 else hist
            recent_mean = float(np.mean(win7))  if win7  else 0.0
            recent_std  = float(np.std(win7))   if win7  else 0.0
            guard_std   = max(recent_std, 5.0)  # 하한
            cap         = _safe_cap(hist)

            # (A) 시그니처에 맞는 피처 생성
            feat_row = _build_features_for_step(hist, feature_columns or [])

            # (B) 모델 입력 정렬
            result_dict = {}
            cols = feature_columns or []
            for col in cols:
                val = feat_row.get(col, 0.0)
                if not np.isfinite(val):
                    val = 0.0
                result_dict[col] = np.float32(val)
            x_input = pd.DataFrame([result_dict]) if cols else pd.DataFrame([{}])

            # (C) 예측 + 강력 보정
            try:
                yhat = float(model.predict(x_input)[0])

                # 1) 유한·양수 보장
                if not np.isfinite(yhat) or yhat < 0:
                    corrections += 1
                    yhat = recent_mean

                # 2) 과도한 이탈 보정 (3σ 초과 → 중간값 쪽으로 당김)
                if abs(yhat - recent_mean) > 3 * guard_std:
                    corrections += 1
                    median30 = float(np.median(win30)) if win30 else recent_mean
                    yhat = 0.5 * yhat + 0.5 * median30

                # 3) 상한 클램프 + 모멘텀 스무딩
                yhat = float(np.clip(yhat, 0.0, cap))
                yhat = 0.7 * yhat + 0.3 * hist[-1]  # 급변 완화

                hist.append(yhat)

                if i % max(1, days_ahead // 10) == 0:
                    print(f"  {i}/{days_ahead} ({100 * i // days_ahead}%)  yhat={yhat:.2f} cap={cap:.0f}")

            except Exception as e:
                print(f"Prediction error at step {i}: {e}")
                fallback = recent_mean if np.isfinite(recent_mean) else (hist[-1] if hist else 0.0)
                hist.append(max(0.0, fallback))
                corrections += 1

        pred_val = hist[-1]
        final_val = max(0, int(round(pred_val)))

        print(f"Result: {final_val} (corrections: {corrections}/{days_ahead})")
        print("=" * 50 + "\n")

        return {
            "date": request.date,
            "predicted_new_cases": final_val,
            "last_training_date": str(last_train_date.date()),
            "forecast_horizon_days": days_ahead,
            "corrections": corrections,
            "prediction_period": f"{getattr(app.state,'predict_start_dt').date()} ~ {getattr(app.state,'predict_end_dt').date()}",
            "note": ("Window auto-extended to today"
                     if ALLOW_EXTEND_TO_TODAY == "1" else
                     ("Prediction within dynamic window" if AUTO_PREDICT_WINDOW == "1" else "Prediction within fixed window"))
        }

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    last_date = "unknown"
    if features_df_cached is not None and "date" in features_df_cached.columns and len(features_df_cached) > 0:
        try:
            last_date = str(pd.to_datetime(features_df_cached["date"]).max().date())
        except Exception:
            pass

    train_end = getattr(app.state, "train_end_dt", None)
    p_start = getattr(app.state, "predict_start_dt", None)
    p_end = getattr(app.state, "predict_end_dt", None)

    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "last_training_date": str(train_end.date()) if train_end is not None else last_date,
        "training_end_env": TRAIN_END_DATE or "auto",
        "prediction_period": f"{(p_start.date() if p_start is not None else (PREDICT_START_DATE or 'auto'))} ~ {(p_end.date() if p_end is not None else (PREDICT_END_DATE or 'auto'))}",
        "max_forecast_days": MAX_FORECAST_DAYS,
        "history_length": len(target_history) if target_history else 0,
        "recent_7day_mean": float(np.mean(target_history[-7:])) if target_history and len(target_history) >= 7 else 0.0,
        "recent_30day_mean": float(np.mean(target_history[-30:])) if target_history and len(target_history) >= 30 else 0.0,
        "mode": "auto" if AUTO_PREDICT_WINDOW == "1" else "fixed-env",
        "auto_extend_to_today": ALLOW_EXTEND_TO_TODAY == "1",
        "abs_max_forecast_days": ABS_MAX_FORECAST_DAYS,
    }

@app.get("/info")
def info():
    train_end = getattr(app.state, "train_end_dt", None)
    p_start = getattr(app.state, "predict_start_dt", None)
    p_end = getattr(app.state, "predict_end_dt", None)

    return {
        "project": "COVID-19 Prediction Model",
        "scenario": "Dynamic short-range forecasting from the last observed date",
        "training_data": {
            "period": f"~ {train_end.date() if train_end is not None else (TRAIN_END_DATE or 'auto')}",
            "description": "Latest engineered features up to the last observed date"
        },
        "prediction_period": {
            "start": str(p_start.date() if p_start is not None else (PREDICT_START_DATE or "auto")),
            "end": str(p_end.date() if p_end is not None else (PREDICT_END_DATE or "auto")),
            "description": "From the day after the last training date up to MAX_FORECAST_DAYS (auto-extended to today if enabled)"
        },
        "constraints": {
            "max_forecast_horizon": f"{MAX_FORECAST_DAYS} days",
            "absolute_cap_days": ABS_MAX_FORECAST_DAYS,
            "reason": "To maintain accuracy for short- to mid-term forecasts"
        },
        "usage": {
            "example": f"POST /predict {{\"date\": \"{(p_start.date() if p_start is not None else 'YYYY-MM-DD')}\"}}",
            "valid_range": f"{(p_start.date() if p_start is not None else (PREDICT_START_DATE or 'auto'))} ~ {(p_end.date() if p_end is not None else (PREDICT_END_DATE or 'auto'))}"
        },
        "mode": "auto" if AUTO_PREDICT_WINDOW == "1" else "fixed-env",
        "auto_extend_to_today": ALLOW_EXTEND_TO_TODAY == "1"
    }
