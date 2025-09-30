import mlflow.pyfunc
import pandas as pd
import numpy as np
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

run_id = "ad20f3bd0faa46598063eef3c8bf3bcb" #15fb849ce7a7465eb73ca187a2a0209a
log_model_name = "model_rf" # model_linear, model_rf, model_xgb
model_uri = f"runs:/{run_id}/{log_model_name}"   # replace "model" with your logged name
model = mlflow.sklearn.load_model(model_uri)

# ---------------------------path setting--------------------------- 
FEAT = Path(r"C:\AIBootCamp\project\mlops\workspace\mlops-cloud-project-mlops-1\data\feature\covid_features.csv")
OUTDIR = Path(r"C:\AIBootCamp\project\mlops\workspace\mlops-cloud-project-mlops-1\serves\outputs")

# ---------------------------data load---------------------------
DATE_FEATS = ["dow_sin","dow_cos","weekofyear","dayofyear","month_sin","month_cos"]
df_feat = pd.read_csv(FEAT)

# ---------------------------feature_engineering--------------------------- 
TARGET = "new_cases"
LOOKBACKS = [1, 7, 14]          # 래그
ROLLS = [7, 14, 28]             # 롤링
def add_time_features(df: pd.DataFrame, date_col="date"):
    if date_col not in df.columns: 
        return df
    df["date"] = pd.to_datetime(df[date_col], errors="coerce")
    df["dow"] = df["date"].dt.dayofweek
    df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)
    df["dayofyear"] = df["date"].dt.dayofyear
    df["dow_sin"] = np.sin(2*np.pi*df["dow"]/7); df["dow_cos"] = np.cos(2*np.pi*df["dow"]/7)
    if "month" in df.columns:
        df["month_sin"] = np.sin(2*np.pi*df["month"]/12); df["month_cos"] = np.cos(2*np.pi*df["month"]/12)
    return df

def add_lag_roll(df: pd.DataFrame, target: str, lags, rolls):
    df = df.sort_values("date").reset_index(drop=True)
    for l in lags:
        df[f"{target}_lag{l}"] = df[target].shift(l)
    for w in rolls:
        df[f"{target}_rollmean{w}"] = df[target].shift(1).rolling(w, min_periods=1).mean()
        df[f"{target}_rollstd{w}"]  = df[target].shift(1).rolling(w, min_periods=1).std()
    df[f"{target}_diff1"] = df[target].diff(1)
    df[f"{target}_pct"]   = df[target].pct_change().replace([np.inf, -np.inf], np.nan)
    df = df.bfill().ffill()
    return df

def select_future_aware_features(df: pd.DataFrame, target: str) -> list:
    """
    미래 시점에도 스스로 생성 가능한 피처만 사용:
    - TARGET의 lag/rolling/diff/pct
    - 날짜 기반 주기 피처 (DATE_FEATS)
    """
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    allowed = []
    for c in num_cols:
        if c in DATE_FEATS:
            allowed.append(c)
        elif c.startswith(f"{target}_"):
            allowed.append(c)
    # 타깃과 y_next는 제외
    allowed = [c for c in allowed if c != target and c != "y_next"]
    # 존재하는 컬럼만
    allowed = [c for c in allowed if c in df.columns]
    return sorted(list(dict.fromkeys(allowed)))
feat_list = select_future_aware_features(df_feat, TARGET)

# ---------------- 동적 재귀 예측 ----------------

def _date_feats(next_date):
    dow = next_date.dayofweek
    weekofyear = int(next_date.isocalendar().week)
    dayofyear  = int(next_date.timetuple().tm_yday)
    month = int(next_date.month)
    out = {
        "dow_sin": np.sin(2*np.pi*dow/7.0),
        "dow_cos": np.cos(2*np.pi*dow/7.0),
        "weekofyear": weekofyear,
        "dayofyear": dayofyear,
        "month_sin": np.sin(2*np.pi*month/12.0),
        "month_cos": np.cos(2*np.pi*month/12.0),
    }
    return out

def _roll_stats(seq, w):
    arr = np.array(seq[-w:], dtype=float)
    return float(np.mean(arr)), float(np.std(arr, ddof=0))

def recursive_forecast_dynamic(df_feat: pd.DataFrame,feat_list:list, target: str, model, pred_date: str) -> pd.DataFrame:
    """
    마지막 관측까지의 target 히스토리로부터 미래 H일을 동적으로 생성.
    - 매 스텝 예측값을 래그/롤링/증감/pct 계산에 주입
    - 날짜 주기 피처는 next_date로 갱신
    """
    pred_date = pd.to_datetime(pred_date)
    # print("pred_date type: ", pred_date.dtype)
    df_feat["date"] = pd.to_datetime(df_feat["date"])
    df = df_feat.sort_values("date").copy()
    last_date = df["date"].max()
    horizon = (pred_date - last_date).days
    # 타깃 히스토리
    hist = df[target].tolist()
    preds, dates = [], []

    max_lag = max(LOOKBACKS) if LOOKBACKS else 1

    for i in range(1, horizon+1):
        nd = last_date + pd.Timedelta(days=i)
        # 필요한 피처 벡터 구성
        feat_row = {}

        # 날짜 피처
        feat_row.update(_date_feats(nd))

        # 래그들
        for l in LOOKBACKS:
            feat_row[f"{target}_lag{l}"] = float(hist[-l]) if len(hist) >= l else float(hist[-1])

        # 롤링 통계
        for w in ROLLS:
            m, s = _roll_stats(hist, min(len(hist), w))
            feat_row[f"{target}_rollmean{w}"] = m
            feat_row[f"{target}_rollstd{w}"] = s

        # 증감/비율
        if len(hist) >= 2:
            diff1 = hist[-1] - hist[-2]
            pct   = (hist[-1] - (hist[-2] if hist[-2] != 0 else 1e-9)) / max(abs(hist[-2]), 1e-9)
        else:
            diff1, pct = 0.0, 0.0
        feat_row[f"{target}_diff1"] = float(diff1)
        feat_row[f"{target}_pct"]   = float(pct)

        # print("feat_row_columns:\n", feat_row.keys())

        # 모델이 실제로 쓰는 컬럼 순서에 맞춰 배열화
        x = np.array([[feat_row.get(c, 0.0) for c in feat_list]], dtype=np.float32)
        yhat = float(model.predict(x)[0])

        # 누적
        preds.append(yhat)
        dates.append(nd)
        hist.append(yhat)  # 예측값을 히스토리에 추가(다음 스텝에 사용)
        print(f"predict date:{nd}, predcit value:{yhat}")

    return pd.DataFrame({"date": dates, "yhat": preds})



def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    # df = pd.read_csv(in_path)
    # if "date" not in df.columns: 
    #     raise ValueError("Need 'date' column.")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = add_time_features(df)
    df = add_lag_roll(df, TARGET, LOOKBACKS, ROLLS)

   # 결측치 채우는 부분
    # num_cols = df.select_dtypes(include=[np.number]).columns
    # df[num_cols] = df[num_cols].interpolate("linear", limit_direction="both")
    # df = df.bfill().ffill()

    return df



# Predict
# input_data = "2025-09-29"
# print("input_data :", input_data)
# output_df = recursive_forecast_dynamic(df_feat, feat_list, TARGET, model, input_data)
# path = OUTDIR / "served_forecast.csv"
# output_df.to_csv(path, index =False)

app = FastAPI()

@app.get("/date")
def predict_date(date: str):
    input_data = date
    output_df = recursive_forecast_dynamic(df_feat, feat_list, TARGET, model, input_data)
    last_yhat = output_df.iloc[-1]["yhat"]
    yhat_int = int(last_yhat) 
    return {"value": yhat_int}