
# main.py - Orchestrate end-to-end COVID pipeline with MLflow + Integrated Trainer
# -*- coding: utf-8 -*-
import os, sys, argparse, json, mlflow
from datetime import datetime
from typing import Optional
import pandas as pd

# make src importable whether run from project root or elsewhere
file_path = os.path.abspath(__file__)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = PROJECT_ROOT
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from pipeline.mlflow_pipeline import MLflowCovidPipeline
from feature_engineering.feature_engineer import CovidFeatureEngineer
from modeling.integrated_trainer import IntegratedCovidTrainer, TrainConfig, add_time_features, add_lag_roll
from config.settings import ProjectConfig

def run_pipeline(args, flag, feature_run_id : Optional[str] = None, config: Optional[ProjectConfig] =None):
    # 1) Collect + Preprocess (or load from MLflow latest)
    print("start pipeline : \n", file_path)
    if flag < 1 :
        pipeline = MLflowCovidPipeline()

        if args.skip_collect:
            print("[info] --skip-collect enabled -> using latest collection run in MLflow (or running full if none).")
            processed = pipeline.run_preprocessing_from_latest_collection()
        else:
            processed = pipeline.run_full_pipeline(collection_run_name="collection_v1",
                                                preprocessing_run_name="preprocessing_v1")

    # 2) Feature Engineering (rich FE)
    if flag < 2:
        fe = CovidFeatureEngineer()
        # print("[main]processed :", processed) checked
        features_df, target_series = fe.engineer_features(processed, target_column=args.target,
                                                        run_name="feature_engineering_v1")

    # 3) (Optional) augment with minimal future-aware FE for recursive forecasting
    #    — ensures required lag/roll/diff/pct exist in case upstream FE didn't add them
    if flag < 3:
        if feature_run_id != None:
            # feature 데이터 불러오기
            mlflow.set_tracking_uri(config.mlflow.TRACKING_URI)

            feature_path = mlflow.artifacts.download_artifacts(
                run_id=feature_run_id,
                artifact_path="features"  # 예: "model", "data.csv", "plots/"
            )
            target_path = mlflow.artifacts.download_artifacts(
                run_id=feature_run_id,
                artifact_path="targets"  # 예: "model", "data.csv", "plots/"
            )
            feature_files = os.listdir(feature_path)
            target_files = os.listdir(target_path)

            feature_csv_file = [f for f in feature_files if f.endswith(".csv")][0]
            target_csv_file = [f for f in target_files if f.endswith(".csv")][0]
            feature_csv_path = os.path.join(feature_path, feature_csv_file)
            target_csv_path = os.path.join(target_path, target_csv_file)

            features = pd.read_csv(feature_csv_path)
            targets = pd.read_csv(target_csv_path)
            features_df, target_series = features, targets
            print("[main]feature load success!")
        features_with_min = features_df.copy()
        features_with_min[args.target] = target_series.values
        features_with_min = add_time_features(features_with_min, "date")
        features_with_min = add_lag_roll(features_with_min, args.target, lags=(1,7,14), rolls=(7,14,28))

        # 4) Train using the integrated trainer
        cfg = TrainConfig(target_col=args.target, test_days=args.test_days, horizon=args.horizon)
        trainer = IntegratedCovidTrainer(cfg, tracking_uri=args.tracking_uri)
        res = trainer.train(features_with_min, target=args.target)
        if res:
            print("\n=== DONE ===")
            print("Best model:", res["best_model"])
            print("Test metrics:", res["metrics"]["test"])
            print("Model Run ID:", res["model_run_id"])
            print("Model Run Name:", res["model_run_name"])
        # if all(v is not None for v in [processed, features_df, target_series, features_with_min]):
        #     del processed, features_df, target_series, features_with_min

        return 0

def build_argparser():
    ap = argparse.ArgumentParser(description="End-to-end COVID pipeline runner (MLflow + Integrated Trainer)")
    ap.add_argument("--tracking-uri", type=str, default=None, help="MLflow tracking URI (e.g., http://mlflow:5000)")
    ap.add_argument("--target", type=str, default="new_cases")
    ap.add_argument("--test-days", type=int, default=60)
    ap.add_argument("--horizon", type=int, default=30)
    ap.add_argument("--skip-collect", action="store_true", help="Use latest MLflow collection instead of pulling anew")
    return ap

def manage_pipeline():
    config = ProjectConfig()
    json_path = config.mlflow.PREPATH
    args = build_argparser().parse_args()
    flag = 0
    cnt = 0
    while(cnt < 5):
        if os.path.exists(json_path) :
            with open(json_path, "r", encoding="utf-8") as f:
                predata = json.load(f)
            if "feature_run_id" in predata :
                feature_run_id = predata["feature_run_id"]
                flag = 2
            else :
                feature_run_id = None
        else :
            feature_run_id = None
        run_pipeline(args, flag, feature_run_id, config)
        cnt += 1
    
    

if __name__ == "__main__":
    manage_pipeline()