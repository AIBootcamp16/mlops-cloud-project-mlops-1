
# main.py - Orchestrate end-to-end COVID pipeline with MLflow + Integrated Trainer
# -*- coding: utf-8 -*-
import os, sys, argparse
from datetime import datetime

# make src importable whether run from project root or elsewhere
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from src.pipeline.mlflow_pipeline import MLflowCovidPipeline
from src.feature_engineering.feature_engineer import CovidFeatureEngineer
from src.modeling.integrated_trainer import IntegratedCovidTrainer, TrainConfig, add_time_features, add_lag_roll

def run_pipeline(args):
    # 1) Collect + Preprocess (or load from MLflow latest)
    pipeline = MLflowCovidPipeline()
    if args.skip_collect:
        print("[info] --skip-collect enabled -> using latest collection run in MLflow (or running full if none).")
        processed = pipeline.run_preprocessing_from_latest_collection()
    else:
        processed = pipeline.run_full_pipeline(collection_run_name="collection_v1",
                                               preprocessing_run_name="preprocessing_v1")

    # 2) Feature Engineering (rich FE)
    fe = CovidFeatureEngineer()
    features_df, target_series = fe.engineer_features(processed, target_column=args.target,
                                                      run_name="feature_engineering_v1")

    # 3) (Optional) augment with minimal future-aware FE for recursive forecasting
    #    — ensures required lag/roll/diff/pct exist in case upstream FE didn't add them
    features_with_min = features_df.copy()
    features_with_min[args.target] = target_series.values
    features_with_min = add_time_features(features_with_min, "date")
    features_with_min = add_lag_roll(features_with_min, args.target, lags=(1,7,14), rolls=(7,14,28))

    # 4) Train using the integrated trainer
    cfg = TrainConfig(target_col=args.target, test_days=args.test_days, horizon=args.horizon)
    trainer = IntegratedCovidTrainer(cfg, tracking_uri=args.tracking_uri)
    res = trainer.train(features_with_min, target=args.target)

    print("\n=== DONE ===")
    print("Best model:", res["best_model"])
    print("Test metrics:", res["metrics"]["test"])

def build_argparser():
    ap = argparse.ArgumentParser(description="End-to-end COVID pipeline runner (MLflow + Integrated Trainer)")
    ap.add_argument("--tracking-uri", type=str, default=None, help="MLflow tracking URI (e.g., http://mlflow:5000)")
    ap.add_argument("--target", type=str, default="new_cases")
    ap.add_argument("--test-days", type=int, default=60)
    ap.add_argument("--horizon", type=int, default=30)
    ap.add_argument("--skip-collect", action="store_true", help="Use latest MLflow collection instead of pulling anew")
    return ap

if __name__ == "__main__":
    args = build_argparser().parse_args()
    run_pipeline(args)
