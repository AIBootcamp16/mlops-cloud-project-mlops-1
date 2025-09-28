from __future__ import annotations

import os, json, math, tempfile
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

# ========================= MLflow Safe Init =========================
def setup_mlflow_or_fallback(experiment_name: str, preferred_uri: Optional[str] = None) -> bool:
    """
    1) Try preferred_uri (or env MLFLOW_TRACKING_URI) first
    2) On failure, fallback to file:/root/mlruns
    Returns: registry_enabled (True if registry usable, False otherwise)
    """
    uri = preferred_uri or os.getenv("MLFLOW_TRACKING_URI", "").strip() or None

    def try_uri(u: str) -> bool:
        try:
            mlflow.set_tracking_uri(u)
            _ = mlflow.get_experiment_by_name(experiment_name)
            mlflow.set_experiment(experiment_name)
            return True
        except Exception:
            return False

    if uri and try_uri(uri):
        reg_enabled = uri.startswith("http")
        if not reg_enabled:
            print(f"[MLflow] Using '{uri}' (file store). Model registry disabled.")
        else:
            print(f"[MLflow] Connected to '{uri}'. Model registry enabled.")
        return reg_enabled

    local_dir = "/root/mlruns"
    os.makedirs(local_dir, exist_ok=True)
    local_uri = f"file:{local_dir}"
    mlflow.set_tracking_uri(local_uri)
    mlflow.set_experiment(experiment_name)
    print(f"[MLflow] Fallback to {local_uri}. Model registry disabled.")
    return False


# ========================= Minimal FE helpers =========================
DATE_FEATS = ["dow_sin","dow_cos","weekofyear","dayofyear","month_sin","month_cos"]

def add_time_features(df: pd.DataFrame, date_col="date") -> pd.DataFrame:
    df = df.copy()
    if date_col not in df.columns:
        for cand in ["Date","ds","DATE","날짜"]:
            if cand in df.columns:
                df.rename(columns={cand:"date"}, inplace=True)
                break
    if "date" not in df.columns:
        raise ValueError("Missing 'date' column after rename attempts.")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().all():
        raise ValueError("All 'date' values failed to parse. Check date format.")
    df["dow"] = df["date"].dt.dayofweek
    df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)
    df["dayofyear"] = df["date"].dt.dayofyear
    df["month"] = df["date"].dt.month
    df["dow_sin"] = np.sin(2*np.pi*df["dow"]/7); df["dow_cos"] = np.cos(2*np.pi*df["dow"]/7)
    df["month_sin"] = np.sin(2*np.pi*df["month"]/12); df["month_cos"] = np.cos(2*np.pi*df["month"]/12)
    return df

def add_lag_roll(df: pd.DataFrame, target: str, lags=(1,7,14), rolls=(7,14,28)) -> pd.DataFrame:
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found.")
    df = df.sort_values("date").reset_index(drop=True)
    for l in lags:
        df[f"{target}_lag{l}"] = df[target].shift(l)
    for w in rolls:
        df[f"{target}_rollmean{w}"] = df[target].shift(1).rolling(w, min_periods=1).mean()
        df[f"{target}_rollstd{w}"] = df[target].shift(1).rolling(w, min_periods=1).std()
    df[f"{target}_diff1"] = df[target].diff(1)
    df[f"{target}_pct"] = df[target].pct_change().replace([np.inf,-np.inf], np.nan)
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].interpolate("linear", limit_direction="both")
    df = df.bfill().ffill()
    return df

def select_future_aware_features(df: pd.DataFrame, target: str) -> List[str]:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    allowed = []
    for c in num_cols:
        if c in DATE_FEATS or c.startswith(f"{target}_"):
            allowed.append(c)
    allowed = [c for c in allowed if c not in {target,"y_next"} and c in df.columns]
    return sorted(list(dict.fromkeys(allowed)))


# ========================= Plot helpers =========================
def _lineplot(dates, series, title, fname):
    fig = plt.figure(); plt.plot(dates, series)
    plt.title(title); plt.xlabel("date"); plt.ylabel("value")
    mlflow.log_figure(fig, fname); plt.close(fig)

def _dualplot(dates, y_true, y_pred, title, fname):
    fig = plt.figure()
    plt.plot(dates, y_true, label="actual")
    plt.plot(dates, y_pred, label="pred")
    plt.legend(); plt.title(title); plt.xlabel("date"); plt.ylabel("value")
    mlflow.log_figure(fig, fname); plt.close(fig)

def _residplot(y_true, y_pred, title, fname):
    res = np.array(y_true) - np.array(y_pred)
    fig = plt.figure(); plt.scatter(y_pred, res, s=8); plt.axhline(0)
    plt.title(title); plt.xlabel("pred"); plt.ylabel("residuals")
    mlflow.log_figure(fig, fname); plt.close(fig)


# ========================= Dynamic Forecast =========================
def _roll_stats(seq: List[float], w: int) -> Tuple[float,float]:
    arr = np.array(seq[-w:], dtype=float)
    return float(np.mean(arr)), float(np.std(arr, ddof=0))

def _date_feats(ts: pd.Timestamp) -> Dict[str,float]:
    dow = ts.dayofweek
    weekofyear = int(ts.isocalendar().week)
    dayofyear = int(ts.timetuple().tm_yday)
    month = int(ts.month)
    return {
        "dow_sin": math.sin(2*math.pi*dow/7.0),
        "dow_cos": math.cos(2*math.pi*dow/7.0),
        "weekofyear": float(weekofyear),
        "dayofyear": float(dayofyear),
        "month_sin": math.sin(2*math.pi*month/12.0),
        "month_cos": math.cos(2*math.pi*month/12.0),
    }

def recursive_forecast_dynamic(df_feat: pd.DataFrame, target: str, feat_list: List[str], model,
                               horizon: int, lookbacks: List[int], rolls: List[int]) -> pd.DataFrame:
    df = df_feat.sort_values("date").copy()
    last_date = pd.to_datetime(df["date"].max())
    hist = df[target].astype(float).tolist()
    preds, dates = [], []
    for i in range(1, horizon+1):
        nd = last_date + pd.Timedelta(days=i)
        row: Dict[str,float] = {}
        row.update(_date_feats(nd))
        for l in lookbacks:
            row[f"{target}_lag{l}"] = float(hist[-l]) if len(hist) >= l else float(hist[-1])
        for w in rolls:
            m,s = _roll_stats(hist, min(len(hist), w))
            row[f"{target}_rollmean{w}"] = m
            row[f"{target}_rollstd{w}"] = s
        if len(hist) >= 2:
            diff1 = hist[-1] - hist[-2]
            base = hist[-2] if hist[-2] != 0 else 1e-9
            pct = (hist[-1] - base) / max(abs(base), 1e-9)
        else:
            diff1, pct = 0.0, 0.0
        row[f"{target}_diff1"] = float(diff1)
        row[f"{target}_pct"] = float(pct)
        x = np.array([[row.get(c,0.0) for c in feat_list]], dtype=np.float32)
        yhat = float(model.predict(x)[0])
        preds.append(yhat); dates.append(nd); hist.append(yhat)
    return pd.DataFrame({"date": dates, "yhat": preds})


# ========================= Metrics =========================
def _metrics(y_true, y_pred) -> Dict[str,float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred)/np.maximum(1e-9, np.abs(y_true))))*100.0)
    return {"rmse": rmse, "mae": mae, "r2": r2, "mape": mape}


# ========================= Trainer =========================
@dataclass
class TrainConfig:
    experiment_name: str = "covid_model_training_integrated"
    target_col: str = "new_cases"
    test_days: int = 60
    horizon: int = 30
    lookbacks: List[int] = None
    rolls: List[int] = None
    do_time_series_cv: bool = True
    cv_splits: int = 3
    random_state: int = 42

    def __post_init__(self):
        if self.lookbacks is None:
            self.lookbacks = [1,7,14]
        if self.rolls is None:
            self.rolls = [7,14,28]


class IntegratedCovidTrainer:
    def __init__(self, cfg: Optional[TrainConfig] = None, tracking_uri: Optional[str] = None):
        self.cfg = cfg or TrainConfig()
        self.registry_enabled = setup_mlflow_or_fallback(self.cfg.experiment_name, tracking_uri)

        self.models: Dict[str,Any] = {
            "random_forest": RandomForestRegressor(random_state=self.cfg.random_state, n_estimators=400, max_depth=12, min_samples_leaf=2, n_jobs=-1),
            "gradient_boosting": GradientBoostingRegressor(random_state=self.cfg.random_state, n_estimators=300, learning_rate=0.05, max_depth=3),
            "linear_regression": LinearRegression(),
            "ridge_regression": Ridge(random_state=self.cfg.random_state),
        }
        try:
            from xgboost import XGBRegressor
            self.models["xgboost"] = XGBRegressor(
                n_estimators=600, max_depth=6, learning_rate=0.05,
                subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
                tree_method="hist", random_state=self.cfg.random_state, n_jobs=-1
            )
        except Exception:
            pass

    def train(self, df_feat: pd.DataFrame, target: str) -> Dict[str,Any]:
        with mlflow.start_run(run_name="model_training_integrated"):
            if "date" not in df_feat.columns:
                raise ValueError("features_df must include 'date' column")
            feat_list = select_future_aware_features(df_feat, target)
            if not feat_list:
                raise ValueError("No future-aware features. FE must add lag/roll/diff/pct and date features.")
            mlflow.log_text("\n".join(feat_list), "features_used.txt")

            df = df_feat.sort_values("date").copy()
            df["y_next"] = df[target].shift(-1)
            df = df.dropna(subset=["y_next"]).reset_index(drop=True)

            X_all = df[feat_list].astype(np.float32)
            y_all = df["y_next"].astype(np.float32)
            dt_all = pd.to_datetime(df["date"]).reset_index(drop=True)

            cutoff = dt_all.max() - pd.Timedelta(days=self.cfg.test_days-1)
            train_mask = dt_all < cutoff
            test_mask  = ~train_mask

            X_train, X_test = X_all[train_mask].reset_index(drop=True), X_all[test_mask].reset_index(drop=True)
            y_train, y_test = y_all[train_mask].reset_index(drop=True), y_all[test_mask].reset_index(drop=True)
            dt_train, dt_test = dt_all[train_mask].reset_index(drop=True), dt_all[test_mask].reset_index(drop=True)

            mlflow.log_params({
                "target": target,
                "test_days": self.cfg.test_days,
                "horizon": self.cfg.horizon,
                "lookbacks": ",".join(map(str,self.cfg.lookbacks)),
                "rolls": ",".join(map(str,self.cfg.rolls)),
                "train_rows": len(X_train),
                "test_rows": len(X_test),
                "feature_count": len(feat_list),
                "models": list(self.models.keys()),
            })

            results: Dict[str,Any] = {}
            for name, model in self.models.items():
                try:
                    model.fit(X_train, y_train)
                    ytr = model.predict(X_train)
                    yte = model.predict(X_test)
                    m_tr = _metrics(y_train, ytr)
                    m_te = _metrics(y_test, yte)

                    for k,v in m_tr.items(): mlflow.log_metric(f"{name}_train_{k}", v)
                    for k,v in m_te.items(): mlflow.log_metric(f"{name}_test_{k}", v)

                    if self.cfg.do_time_series_cv:
                        try:
                            tscv = TimeSeriesSplit(n_splits=self.cfg.cv_splits)
                            cv = cross_val_score(model, X_train, y_train, cv=tscv, scoring="neg_root_mean_squared_error")
                            mlflow.log_metric(f"{name}_cv_rmse_mean", float((-cv).mean()))
                            mlflow.log_metric(f"{name}_cv_rmse_std", float((-cv).std()))
                        except Exception:
                            pass

                    _dualplot(dt_test, y_test, yte, f"{name}: actual vs pred (test)", f"{name}_test_pred.png")
                    _residplot(y_test, yte, f"{name}: residuals (test)", f"{name}_test_resid.png")

                    results[name] = {"model": model, "metrics": {"train": m_tr, "test": m_te}, "pred": {"train": ytr, "test": yte}}
                except Exception as e:
                    mlflow.log_text(str(e), f"{name}_error.txt")

            if not results:
                raise RuntimeError("No models trained successfully.")

            best = min(results, key=lambda n: results[n]["metrics"]["test"]["rmse"])
            best_model = results[best]["model"]
            mlflow.log_param("best_model_name", best)
            mlflow.log_metric("best_model_test_rmse", results[best]["metrics"]["test"]["rmse"])
            mlflow.log_metric("best_model_test_r2",   results[best]["metrics"]["test"]["r2"])

            comp = { n: {
                "test_rmse": r["metrics"]["test"]["rmse"],
                "test_mae":  r["metrics"]["test"]["mae"],
                "test_r2":   r["metrics"]["test"]["r2"],
                "train_rmse":r["metrics"]["train"]["rmse"],
                "train_r2":  r["metrics"]["train"]["r2"],
            } for n,r in results.items() }

            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                json.dump(comp, f, indent=2); mlflow.log_artifact(f.name, "analysis/model_comparison.json"); tmp=f.name
            import os; os.remove(tmp)

            dfp = pd.DataFrame({
                "split": ["train"]*len(y_train) + ["test"]*len(y_test),
                "actual": np.concatenate([y_train.values, y_test.values]),
                "predicted": np.concatenate([results[best]["pred"]["train"], results[best]["pred"]["test"]]),
            })
            dfp["residual"] = dfp["actual"] - dfp["predicted"]
            dfp["abs_residual"] = np.abs(dfp["residual"])
            with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
                dfp.to_csv(f.name, index=False); mlflow.log_artifact(f.name, "results/predictions_detailed.csv"); tmp=f.name
            import os; os.remove(tmp)

            try:
                if hasattr(best_model,"feature_importances_"):
                    imp = pd.DataFrame({"feature": X_train.columns, "importance": best_model.feature_importances_}).sort_values("importance", ascending=False)
                    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
                        imp.to_csv(f.name, index=False); mlflow.log_artifact(f.name, "analysis/feature_importance.csv"); tmp=f.name
                    os.remove(tmp)
            except Exception:
                pass

            try:
                sig = infer_signature(X_train, best_model.predict(X_train))
                if self.registry_enabled:
                    info = mlflow.sklearn.log_model(
                        sk_model=best_model,
                        name="best_model",
                        signature=sig,
                        input_example=X_train.head(3),
                        registered_model_name="covid_prediction_model",
                        serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
                    )
                    mlflow.log_text(info.model_uri, "model_uri.txt")
                else:
                    info = mlflow.sklearn.log_model(
                        sk_model=best_model,
                        name="best_model",
                        signature=sig,
                        input_example=X_train.head(3),
                    )
                    mlflow.log_text(info.model_uri, "model_uri_artifact_only.txt")
            except Exception as e2:
                mlflow.log_text(f"model_save_failed: {e2}", "model_save_failed.txt")

            try:
                df_fore = recursive_forecast_dynamic(df_feat, target, list(X_train.columns), best_model,
                                                     self.cfg.horizon, self.cfg.lookbacks, self.cfg.rolls)
                with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
                    df_fore.to_csv(f.name, index=False); mlflow.log_artifact(f.name, "forecast/forecast_future.csv"); tmp=f.name
                os.remove(tmp)
                _lineplot(df_fore["date"], df_fore["yhat"], f"{best} forecast (future)", f"{best}_forecast.png")
            except Exception as e:
                mlflow.log_text(str(e), "forecast_error.txt")

            summ = {
                "timestamp": datetime.now().isoformat(),
                "total_models_trained": len(results),
                "best_model": best,
                "best_performance": results[best]["metrics"]["test"],
            }
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                json.dump(summ, f, indent=2); mlflow.log_artifact(f.name, "reports/training_summary.json"); tmp=f.name
            os.remove(tmp)

            return {"best_model": best, "metrics": results[best]["metrics"]}